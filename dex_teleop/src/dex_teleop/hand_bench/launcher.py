"""Finger-tracking bench: a fixed right Sharpa hand above a table in OmniGibson.

The bench accepts an articulation source only (Quest HTS or a MANUS glove).
Wrist tracking is disabled by construction: the Franka that carries the Sharpa
hand is hidden and held by a ``NullJointController``, the action space is the
22 finger joints, and the runtime pins the wrist through ``FixedWristSource``.

With ``--record`` (implied by ``--emg``) the session is written with the same
HDF5 layout as the ARAT launcher: ``/data/demo_N`` finger actions and simulator
state, ``/hand_tracking`` native-rate articulation, wrist, and retargeting
streams, ``/human_hand_pose``, ``/action_timing``, and, with ``--emg``, the
OYMotion wristband under ``/emg`` with per-action ``/synchronization`` ranges.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from dex_teleop.arat.camera_rig import camera_rig_names, layout_camera_ids, load_camera_rig
from dex_teleop.emg import ActionTimingSample, EmgSession, emg_staging_path
from dex_teleop.hand_bench.scene import (
    CAMERA_LAYOUT,
    DEFAULT_CAMERA_RIG,
    TABLE_NAME,
    HandBenchScene,
    build_hand_bench_config,
    hand_pose_error,
    hidden_prim_names,
    load_hand_bench_scene,
    reset_joint_positions,
    validate_runtime_assets,
)
from dex_teleop.omnigibson.hand_pose_recording import HumanHandPoseSample
from dex_teleop.omnigibson.hand_tracking_recording import HandTrackingRecordingSession
from dex_teleop.omnigibson.launcher import (
    GracefulShutdown,
    _camera_layout_panel_dock,
    _configure_camera_layout,
    _file_fingerprint,
    _finalize_recording,
    _show_robot_end_effectors,
    _shutdown_omnigibson,
    _window_name,
    recording_staging_path,
)
from dex_teleop.retargeting import SUPPORTED_RETARGETERS, create_hand_retargeter
from dex_teleop.runtime import MultiSourceTrackingWorker
from dex_teleop.tracking import (
    FIXED_WRIST_SOURCE_NAME,
    FixedWristSource,
    HandObservationFuser,
    HTSSource,
    SourceUnavailableError,
)
from dex_teleop.tracking.manus import (
    ManusIntegratedSource,
    default_manus_bridge_path,
    discover_manus_sdk,
)
from dex_teleop.types import Handedness


LOGGER = logging.getLogger(__name__)
HAND_SOURCES = ("quest", "hts", "manus")
HAND_MODEL = "sharpa"
HAND_SIDE = "right"
SCREENSHOT_SETTLE_FRAMES = 90
ARM_HOLD_TOLERANCE_RAD = 0.05
WAIT_MESSAGE_INTERVAL_SECONDS = 5.0
DEFAULT_RECORDING_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "recordings"

# Options of the ARAT launcher that select, calibrate, or tune wrist tracking.
# They are rejected up front with an explanation instead of argparse's generic
# "unrecognized arguments" message.
REJECTED_WRIST_OPTIONS = (
    "--source",
    "--wrist-source",
    "--record-wrist-source",
    "--record-hand-source",
    "--vive-calibration",
    "--articulation-wrist-calibration",
    "--auto-anchor",
    "--position-sensitivity",
    "--tracking-yaw-deg",
    "--manus-hand-motion",
    "--manus-core-calibration",
    "--manus-expected-tracker-id",
    "--manus-expected-tracker-user-id",
    "--manus-tracker-diagnostics",
    "--no-manus-tracker-diagnostics",
    "--manus-tracker-stale-timeout",
)


def _canonical_hand_source(name: str) -> str:
    return "quest" if name in {"quest", "hts"} else name


def reject_wrist_options(argv: list[str]) -> None:
    """Refuse wrist-tracking options: the bench holds the wrist in the simulator."""

    offending = sorted(
        {
            option
            for token in argv
            for option in REJECTED_WRIST_OPTIONS
            if token == option or token.startswith(option + "=")
        }
    )
    if offending:
        raise SystemExit(
            "The hand bench disables wrist tracking; not accepted: "
            + ", ".join(offending)
            + ". Only finger tracking is configurable: "
            "--hand-source quest|hts|manus (plus the HTS network and MANUS glove options)."
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hand-source",
        choices=HAND_SOURCES,
        default="quest",
        help="Finger articulation provider (quest and hts are aliases for Quest HTS; default: quest)",
    )
    parser.add_argument("--retargeter", choices=SUPPORTED_RETARGETERS, default="adaptive")
    parser.add_argument(
        "--finger-target-scale",
        type=float,
        default=1.0,
        help="Scale applied to the retargeted finger targets, in (0, 1] (default: 1.0)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTS listen address")
    parser.add_argument("--port", type=int, default=9000, help="HTS listen port")
    parser.add_argument("--protocol", choices=("udp", "tcp"), default="udp")
    parser.add_argument(
        "--manus-sdk-root",
        default=str(Path.home() / "Desktop/emg/manus"),
        help="MANUS install/discovery root; the bridge also searches ~/manus_setup",
    )
    parser.add_argument(
        "--manus-mode",
        choices=("integrated", "remote"),
        default="integrated",
        help="Integrated gloves-only sidecar (default) or Remote Windows-Core client",
    )
    parser.add_argument("--manus-core-host", help="Remote Core selector: IP, exact name, or sorted index")
    parser.add_argument("--manus-loopback-only", action="store_true")
    parser.add_argument("--manus-bridge", help="Pre-built matching MANUS bridge executable")
    parser.add_argument("--manus-calibration", help="MANUS .mcal file for the right glove")
    parser.add_argument("--manus-settings-dir", help="Writable MANUS SDK settings directory")
    parser.add_argument("--manus-log-dir", help="Writable MANUS SDK log directory")
    parser.add_argument("--manus-startup-timeout", type=float, default=35.0)
    parser.add_argument("--manus-connect-timeout", type=int, default=15)
    parser.add_argument("--manus-glove-timeout", type=int, default=15)
    parser.add_argument("--manus-reconnect-timeout", type=int, default=15)
    parser.add_argument("--manus-discovery-wait", type=int, default=1)
    parser.add_argument(
        "--camera-rig",
        choices=camera_rig_names(),
        default=DEFAULT_CAMERA_RIG,
        help=f"Camera rig with a 'teleop' layout (default: {DEFAULT_CAMERA_RIG})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after this many simulator steps; 0 runs until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--maximum-frame-age",
        type=float,
        default=0.5,
        help="Oldest acceptable articulation receipt age in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--initial-frame-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the first hand frame before failing (default: 30)",
    )
    parser.add_argument(
        "--stale-frame-policy",
        choices=("error", "hold"),
        default="error",
        help="Fail on a stale articulation stream (default) or hold the last finger pose",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Load the bench without any tracking source; the hand stays open",
    )
    parser.add_argument(
        "--show-arm",
        action="store_true",
        help="Keep the Franka and its pedestal visible (debugging the hidden mount)",
    )
    parser.add_argument(
        "--screenshot-dir",
        help="Render every camera of the layout's toggle cycle to PNG in this directory, then exit",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "Record finger actions, simulator state, native-rate hand tracking streams, and "
            "human hand poses to HDF5 (implied by --emg)"
        ),
    )
    parser.add_argument(
        "--recording-path",
        help=f"HDF5 destination (default: {DEFAULT_RECORDING_ROOT}/hand_bench_<timestamp>.hdf5)",
    )
    parser.add_argument(
        "--emg",
        action="store_true",
        help="Record synchronized native-rate OYMotion EMG and show it in a docked Kit window",
    )
    parser.add_argument(
        "--emg-device",
        help="OYMotion device name substring or Bluetooth address (auto-selects when exactly one is found)",
    )
    parser.add_argument(
        "--emg-adapter",
        default="hci0",
        help="Linux Bluetooth adapter used by the Synchroni SDK (default: hci0)",
    )
    parser.add_argument(
        "--emg-sdk-path",
        help="Path containing the Synchroni SDK's sensor package if it is not installed in behavior_dex",
    )
    parser.add_argument(
        "--emg-hpf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable or disable the firmware 0.5 Hz HPF (default: enabled)",
    )
    parser.add_argument(
        "--emg-lpf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable or disable the firmware 80 Hz LPF (default: enabled)",
    )
    parser.add_argument(
        "--emg-notch",
        choices=("off", "50", "60", "both"),
        default=None,
        help="Explicit firmware mains-notch selection (default: 60)",
    )
    parser.add_argument("--emg-scan-ms", type=int, default=5000)
    parser.add_argument("--emg-connect-timeout", type=float, default=45.0)
    parser.add_argument(
        "--no-emg-display",
        action="store_true",
        help="Record EMG without creating the docked waveform window",
    )
    parser.add_argument(
        "--visualize-decoder",
        action="store_true",
        help="Decode EMG with VEMG2Pose and show the hand in a docked window",
    )
    parser.add_argument(
        "--display",
        choices=("decoder",),
        help="Display an optional panel; 'decoder' is an alias for --visualize-decoder",
    )
    parser.add_argument(
        "--emg2pose-root",
        help="emg2pose checkout (defaults to the emg2pose directory beside this repository)",
    )
    parser.add_argument("--emg2pose-checkpoint", help="VEMG2Pose checkpoint; defaults inside --emg2pose-root")
    parser.add_argument("--emg2pose-device", help="Torch decoder device: auto, cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument(
        "--emg2pose-inference-hz",
        type=float,
        help="Target decoded-hand update rate (default: 5 Hz; 10-15 Hz is practical on CUDA)",
    )
    return parser


def default_recording_path(now: float | None = None) -> Path:
    """Timestamped default so successive bench sessions never overwrite each other."""

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    return DEFAULT_RECORDING_ROOT / f"hand_bench_{stamp}.hdf5"


def _validate_args(args) -> None:
    if not 0.0 < args.finger_target_scale <= 1.0 or not math.isfinite(args.finger_target_scale):
        raise SystemExit("--finger-target-scale must be in (0, 1]")
    if (
        args.steps < 0
        or not math.isfinite(args.maximum_frame_age)
        or args.maximum_frame_age <= 0
        or not math.isfinite(args.initial_frame_timeout)
        or args.initial_frame_timeout <= 0
    ):
        raise SystemExit("--steps must be >= 0 and the frame age/timeout options must be positive")
    if (
        not math.isfinite(args.manus_startup_timeout)
        or args.manus_startup_timeout <= 0
        or args.manus_connect_timeout < 0
        or args.manus_glove_timeout < 0
        or args.manus_reconnect_timeout < 0
        or args.manus_discovery_wait <= 0
    ):
        raise SystemExit("MANUS timeouts are invalid")
    manus_selected = _canonical_hand_source(args.hand_source) == "manus"
    manus_only = (
        args.manus_bridge,
        args.manus_calibration,
        args.manus_core_host,
        args.manus_settings_dir,
        args.manus_log_dir,
    )
    if not manus_selected and (
        any(option is not None for option in manus_only)
        or args.manus_mode != "integrated"
        or args.manus_loopback_only
    ):
        raise SystemExit("MANUS mode, bridge, host, calibration, and directory options require --hand-source manus")
    if args.manus_mode == "integrated" and (args.manus_core_host is not None or args.manus_loopback_only):
        raise SystemExit("--manus-core-host and --manus-loopback-only require --manus-mode remote")
    if args.view_only and args.screenshot_dir is not None:
        raise SystemExit("--screenshot-dir already implies a tracking-free load; drop --view-only")
    if args.display == "decoder":
        args.visualize_decoder = True
    if args.emg:
        args.record = True
    if not args.emg and (
        args.emg_device is not None
        or args.emg_sdk_path is not None
        or args.no_emg_display
        or args.visualize_decoder
        or args.emg_hpf is not None
        or args.emg_lpf is not None
        or args.emg_notch is not None
        or args.emg2pose_root is not None
        or args.emg2pose_checkpoint is not None
        or args.emg2pose_device is not None
        or args.emg2pose_inference_hz is not None
    ):
        raise SystemExit("EMG device, SDK, filter, display, and decoder options require --emg")
    if args.visualize_decoder and args.no_emg_display:
        raise SystemExit("--visualize-decoder requires the EMG display; remove --no-emg-display")
    if not args.visualize_decoder and (
        args.emg2pose_root is not None
        or args.emg2pose_checkpoint is not None
        or args.emg2pose_device is not None
        or args.emg2pose_inference_hz is not None
    ):
        raise SystemExit("emg2pose path and device options require --visualize-decoder")
    if args.emg2pose_inference_hz is not None and args.emg2pose_inference_hz <= 0:
        raise SystemExit("--emg2pose-inference-hz must be positive")
    if args.emg_scan_ms <= 0 or args.emg_connect_timeout <= 0:
        raise SystemExit("EMG scan and connection timeouts must be positive")
    if args.recording_path is not None and not args.record:
        raise SystemExit("--recording-path requires --record or --emg")
    if args.record and (args.view_only or args.screenshot_dir is not None):
        raise SystemExit("--record and --emg need live finger tracking; drop --view-only / --screenshot-dir")


def _force_data_path() -> Path:
    data_root = Path(__file__).resolve().parents[4] / "datasets"
    configured = os.environ.get("OMNIGIBSON_DATA_PATH")
    if configured is not None and Path(configured).expanduser().resolve() != data_root.resolve():
        raise SystemExit(
            "OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat; unset it or set it to " f"{data_root}"
        )
    os.environ["OMNIGIBSON_DATA_PATH"] = str(data_root)
    return data_root


def create_articulation_source(args):
    """Construct the selected finger-tracking receiver (never a wrist provider)."""

    source = _canonical_hand_source(args.hand_source)
    if source == "quest":
        return HTSSource(host=args.host, port=args.port, protocol=args.protocol)
    if source == "manus":
        return ManusIntegratedSource(
            bridge_executable=args.manus_bridge,
            sdk_root=args.manus_sdk_root,
            mode=args.manus_mode,
            core_host=args.manus_core_host,
            loopback_only=args.manus_loopback_only,
            startup_timeout=args.manus_startup_timeout,
            connect_timeout=args.manus_connect_timeout,
            glove_timeout=args.manus_glove_timeout,
            reconnect_timeout=args.manus_reconnect_timeout,
            discovery_wait=args.manus_discovery_wait,
            required_handedness=Handedness.RIGHT,
            settings_dir=args.manus_settings_dir,
            log_dir=args.manus_log_dir,
            right_calibration=args.manus_calibration,
            require_tracker_for_wrist=False,
        )
    raise SystemExit(f"Unsupported hand source {args.hand_source!r}")


def create_worker(args, articulation_source=None) -> MultiSourceTrackingWorker:
    """Fuse the finger source with a fixed wrist and the selected Sharpa retargeter."""

    source_key = _canonical_hand_source(args.hand_source)
    inner = create_articulation_source(args) if articulation_source is None else articulation_source
    fixed = FixedWristSource(inner)
    retargeter = create_hand_retargeter(args.retargeter, "sharpa", hand_side="right")
    return MultiSourceTrackingWorker(
        sources={source_key: fixed, FIXED_WRIST_SOURCE_NAME: fixed},
        articulation_source=source_key,
        wrist_source=FIXED_WRIST_SOURCE_NAME,
        retargeter=retargeter,
        handedness=Handedness.RIGHT,
        fuser=HandObservationFuser(interpolate_wrist=False),
        retargeter_name=args.retargeter,
    )


def _hide_mount_prims(robot, scene: HandBenchScene, *, show_arm: bool) -> tuple[str, ...]:
    """Hide the Franka links and the palm-mounted camera pods; return what was hidden."""

    import omnigibson as og
    import omnigibson.lazy as lazy

    prefixes = scene.robot.hidden_prim_prefixes
    links = []
    child_prims = []
    for link_name, link in robot.links.items():
        if hidden_prim_names((link_name,), prefixes):
            links.append((link_name, link))
            continue
        # Fixed-joint children merged into a link (the ARAT wrist-camera pods
        # under right_hand_C_MC) are plain Xform prims rather than robot links.
        for child in link.prim.GetChildren():
            if hidden_prim_names((child.GetName(),), prefixes):
                child_prims.append((f"{link_name}/{child.GetName()}", child))
    names = tuple(name for name, _ in links) + tuple(name for name, _ in child_prims)
    if show_arm:
        return names
    for _, link in links:
        link.visible = False
    with og.sim.editing_usd():
        for _, child in child_prims:
            lazy.pxr.UsdGeom.Imageable(child).MakeInvisible()
    return names


def _validate_bench(env, robot, table, scene: HandBenchScene) -> None:
    """Assert the table height, the held hand pose, and that nothing touches the table."""

    from omnigibson.utils.usd_utils import RigidContactAPI

    expected = scene.robot.expected_hand_pose
    _, aabb_max = table.aabb
    tabletop = float(aabb_max[2])
    if abs(tabletop - scene.table.tabletop_height) > scene.table.height_tolerance_m:
        raise RuntimeError(
            f"Tabletop is at {tabletop:.4f} m, expected {scene.table.tabletop_height:.4f} m; "
            "update table.position or table.tabletop_height in hand_bench.yaml"
        )
    link = robot.links.get(expected.link)
    if link is None:
        raise RuntimeError(f"Robot has no link {expected.link!r}; available: {sorted(robot.links)}")
    position, quaternion = link.get_position_orientation()
    position = position.cpu().numpy()
    quaternion = quaternion.cpu().numpy()
    translation, degrees = hand_pose_error(position, quaternion, expected)
    if translation > expected.position_tolerance_m or degrees > expected.orientation_tolerance_deg:
        raise RuntimeError(
            f"{expected.link} settled at position {np.round(position, 4).tolist()} / orientation_xyzw "
            f"{np.round(quaternion, 5).tolist()}, {translation * 1000:.1f} mm and {degrees:.2f} deg from the expected "
            "hand pose; re-solve arm_joint_positions or update expected_hand_pose in hand_bench.yaml"
        )
    arm_indices = robot.arm_control_idx[robot.arm_names[0]]
    measured_arm = robot.get_joint_positions()[arm_indices].cpu().numpy()
    arm_error = float(np.max(np.abs(measured_arm - np.asarray(scene.robot.arm_joint_positions))))
    if arm_error > ARM_HOLD_TOLERANCE_RAD:
        raise RuntimeError(
            f"The hidden Franka drifted {math.degrees(arm_error):.1f} deg from its held configuration"
        )
    if RigidContactAPI.is_in_contact(
        scene_idx=env.scene.idx,
        query_set=[robot],
        with_set=[table],
        ignore_set=None,
        current_only=True,
    ):
        raise RuntimeError("The hidden Franka or the Sharpa hand is touching the table; raise the hand pose")
    print(
        f"Hand bench ready: {expected.link} at {np.round(position, 3).tolist()} "
        f"({translation * 1000:.1f} mm / {degrees:.2f} deg from nominal), tabletop at {tabletop:.3f} m"
    )


def _capture_layout_cameras(env, camera_rig_name: str, output_dir: Path) -> list[Path]:
    """Render every camera in the layout's toggle cycles (and viewports) to PNG."""

    import omnigibson as og
    from PIL import Image

    rig = load_camera_rig(camera_rig_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(SCREENSHOT_SETTLE_FRAMES):
        og.sim.render()
    paths = []
    for camera_id in layout_camera_ids(rig, CAMERA_LAYOUT):
        sensor = env.external_sensors[rig.camera(camera_id)["sensor_name"]]
        observation, _ = sensor.get_obs()
        rgb = observation["rgb"]
        array = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        array = np.ascontiguousarray(array[..., :3]).astype(np.uint8)
        path = output_dir / f"{camera_id}.png"
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


def _hold_reset_pose(robot, reset_positions) -> None:
    robot.set_joint_positions(reset_positions)
    robot.keep_still()


def create_emg_session(args, recording_path: Path) -> EmgSession:
    """Configure the OYMotion sidecar exactly as the ARAT launcher does."""

    return EmgSession(
        emg_staging_path(recording_path),
        device=args.emg_device,
        adapter=args.emg_adapter,
        sdk_path=args.emg_sdk_path,
        scan_ms=args.emg_scan_ms,
        filter_hpf=True if args.emg_hpf is None else args.emg_hpf,
        filter_lpf=True if args.emg_lpf is None else args.emg_lpf,
        filter_notch=args.emg_notch or "60",
        visualize_decoder=args.visualize_decoder,
        emg2pose_root=args.emg2pose_root,
        emg2pose_checkpoint=args.emg2pose_checkpoint,
        decoder_device=args.emg2pose_device or "auto",
        decoder_hand=HAND_SIDE,
        decoder_inference_hz=args.emg2pose_inference_hz or 5.0,
    )


def set_recording_metadata(session: HandTrackingRecordingSession, worker: MultiSourceTrackingWorker, args) -> None:
    """Attach provenance for the finger source, the fixed wrist, and the retargeter."""

    source_key = worker.articulation_source
    fixed_source = worker.sources[source_key]
    if not isinstance(fixed_source, FixedWristSource):
        raise TypeError("The hand bench worker must wrap its finger source in FixedWristSource")
    articulation: dict[str, object] = {"registry_key": source_key}
    if source_key == "quest":
        articulation.update(
            {
                "provider": "Quest Hand Tracking Streamer",
                "endpoint": f"{args.protocol}://{args.host}:{args.port}",
            }
        )
    elif source_key == "manus":
        try:
            resolved_sdk_root = str(discover_manus_sdk(args.manus_sdk_root, mode=args.manus_mode).root)
        except SourceUnavailableError:
            resolved_sdk_root = "unavailable"
        articulation.update(
            {
                **fixed_source.articulation_source.recording_metadata(),
                "requested_sdk_root": str(Path(args.manus_sdk_root).expanduser()),
                "resolved_sdk_root": resolved_sdk_root,
                "bridge": _file_fingerprint(args.manus_bridge or default_manus_bridge_path(args.manus_mode)),
                "glove_calibration": _file_fingerprint(args.manus_calibration),
            }
        )
    session.set_stream_metadata(worker.articulation_streams[source_key], articulation)
    session.set_stream_metadata(
        worker.wrist_streams[worker.wrist_source],
        {
            "registry_key": worker.wrist_source,
            "provider": "FixedWristSource: wrist held by the simulator, device wrist discarded",
            "articulation_source": source_key,
            "position": fixed_source.position.tolist(),
            "quaternion_xyzw": fixed_source.quaternion_xyzw.tolist(),
        },
    )
    session.set_stream_metadata(
        worker.control_wrist_stream,
        {
            "provider": "same-callback identity pairing",
            "native_stream": worker.wrist_streams[worker.wrist_source],
            "maximum_skew_seconds": 0.0,
            "interpolation": "none; exact source frame/time identity",
            "articulation_to_wrist": None,
        },
    )
    suffix = f"{HAND_MODEL}_{HAND_SIDE}_dexpilot.yaml" if args.retargeter == "dexpilot" else f"{HAND_MODEL}.yaml"
    config_path = Path(__file__).resolve().parents[1] / "retargeting" / "configs" / suffix
    session.set_stream_metadata(
        worker.retargeting_stream,
        {
            "backend": args.retargeter,
            "hand_model": HAND_MODEL,
            "handedness": HAND_SIDE,
            "configuration": _file_fingerprint(config_path),
        },
    )


def _create_emg_monitor(env, emg_session: EmgSession, args):
    """Dock the waveform (and optional decoder) panels where the camera rig says."""

    import omnigibson as og
    from omnigibson.macros import gm

    from dex_teleop.emg.ui import create_emg_monitor

    default_parent = "DockSpace" if gm.HEADLESS else _window_name(env._arat_camera_viewports["main"])
    emg_parent, emg_position, emg_ratio = _camera_layout_panel_dock(
        env, "emg", default_parent=default_parent, default_position="right", default_ratio=0.35
    )
    decoder_parent, decoder_position, decoder_ratio = _camera_layout_panel_dock(
        env, "decoder", default_parent="OYMotion EMG", default_position="bottom", default_ratio=0.5
    )
    monitor = create_emg_monitor(
        emg_session,
        dock_parent_name=emg_parent,
        dock_position=emg_position,
        dock_ratio=emg_ratio,
        enabled=not args.no_emg_display,
        visualize_decoder=args.visualize_decoder,
        decoder_hand=HAND_SIDE,
        decoder_dock_parent_name=decoder_parent,
        decoder_dock_position=decoder_position,
        decoder_dock_ratio=decoder_ratio,
    )
    if monitor is not None:
        monitor.update(force=True)
        og.sim.render()
    return monitor


def run_bench(
    scene: HandBenchScene,
    args,
    worker: MultiSourceTrackingWorker | None,
    shutdown: GracefulShutdown,
    *,
    recording_path: Path | None = None,
    emg_session: EmgSession | None = None,
) -> None:
    import omnigibson as og
    import omnigibson.lazy as lazy
    import torch as th
    from omnigibson.macros import gm
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    from dex_teleop.omnigibson.sharpa_finger_adapter import (
        SharpaFingerActionAdapter,
        SharpaFingerAdapterConfig,
    )

    if recording_path is not None and worker is None:
        raise ValueError("Recording requires a tracking worker")
    if emg_session is not None and recording_path is None:
        raise ValueError("EMG acquisition requires a recording path")
    screenshot_dir = None if args.screenshot_dir is None else Path(args.screenshot_dir).expanduser()
    headless = os.environ.get("OMNIGIBSON_HEADLESS") == "1"
    config = build_hand_bench_config(
        scene,
        camera_rig_name=args.camera_rig,
        viewport_name=None if headless else "Viewport",
        sensor_modalities=("rgb",) if screenshot_dir is not None else None,
    )
    print(f"\nLoading the Sharpa hand bench (camera rig: {args.camera_rig})")
    env = og.Environment(configs=config)
    shutdown.install()

    recording_env = None
    staging_path = None
    hand_pose_episodes: list[list[HumanHandPoseSample]] | None = None
    action_timing_episodes: list[list[ActionTimingSample]] | None = None
    hand_tracking_session: HandTrackingRecordingSession | None = None
    if recording_path is not None:
        from omnigibson.envs import HDF5CollectionWrapper

        recording_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = recording_staging_path(recording_path)
        # JoyLo's state/action trajectory wrapper, as in the ARAT launcher.
        recording_env = HDF5CollectionWrapper(
            env=env,
            output_path=str(staging_path),
            viewport_camera_path=None,
            only_successes=False,
            flush_every_n_traj=1,
            keep_checkpoint_rollback_data=True,
        )
        env = recording_env
        hand_pose_episodes = []
        action_timing_episodes = []
        hand_tracking_session = HandTrackingRecordingSession()
        set_recording_metadata(hand_tracking_session, worker, args)
        worker.set_recording_session(hand_tracking_session)
        print(
            f"Recording to {recording_path} (staging: {staging_path.name}): /data finger actions and state, "
            "/hand_tracking native streams, /human_hand_pose, /action_timing"
            + (", /emg and /synchronization" if emg_session is not None else "")
        )

    reset_positions = th.tensor(reset_joint_positions(scene), dtype=th.float32)

    def begin_episode(robot=None) -> None:
        env.reset()
        if hand_pose_episodes is not None:
            hand_pose_episodes.append([])
        if action_timing_episodes is not None:
            action_timing_episodes.append([])
        if hand_tracking_session is not None:
            hand_tracking_session.begin_episode()
        if robot is not None:
            _hold_reset_pose(robot, reset_positions)
            _show_robot_end_effectors(robot)

    begin_episode()
    robot = env.robots[0]
    table = env.scene.object_registry("name", TABLE_NAME)
    if table is None:
        raise RuntimeError("Hand bench scene did not load its table")
    _hold_reset_pose(robot, reset_positions)
    for _ in range(10):
        og.sim.step()
    _show_robot_end_effectors(robot)
    hidden = _hide_mount_prims(robot, scene, show_arm=args.show_arm)
    print(
        f"{'Kept visible for debugging' if args.show_arm else 'Hidden'}: "
        f"{len(hidden)} mount prims (Franka links, pedestal, wrist-camera pods)"
    )
    _validate_bench(env, robot, table, scene)
    _configure_camera_layout(env, args.camera_rig, CAMERA_LAYOUT, robot)

    if screenshot_dir is not None:
        for path in _capture_layout_cameras(env, args.camera_rig, screenshot_dir):
            print(f"Wrote {path}")
        return

    steps = 0
    if worker is None:
        print("View-only hand bench; press B to switch views, Ctrl+C to exit.")
        while (args.steps <= 0 or steps < args.steps) and not shutdown.requested:
            _hold_reset_pose(robot, reset_positions)
            og.sim.step()
            steps += 1
        return

    emg_monitor = None
    primary_error: BaseException | None = None
    applied = 0
    try:
        if emg_session is not None:
            emg_monitor = _create_emg_monitor(env, emg_session, args)
        source_label = _canonical_hand_source(args.hand_source)
        adapter = SharpaFingerActionAdapter(
            robot,
            config=SharpaFingerAdapterConfig(control_hz=30.0, finger_target_scale=args.finger_target_scale),
        )
        control = {"engaged": True, "pending": None}

        def request_toggle():
            control["pending"] = "toggle"

        def request_reset():
            control["pending"] = "reset"

        if not gm.HEADLESS:
            KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.SPACE, request_toggle)
            KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.R, request_reset)
        reset_hint = "R starts a new episode with an open hand" if recording_env is not None else "R reopens the hand"
        print(
            f"Waiting for the first {source_label} hand frame (wrist tracking disabled; fingers only). "
            f"Press SPACE to pause/resume, {reset_hint}, B to switch views; Ctrl+C exits."
        )
        started = time.monotonic()
        last_wait_message = started
        live = False
        holding_stale = False

        def idle_step() -> None:
            og.sim.step()
            if emg_monitor is not None:
                emg_monitor.update()

        while (args.steps <= 0 or steps < args.steps) and not shutdown.requested:
            pending = control["pending"]
            control["pending"] = None
            if pending == "toggle":
                control["engaged"] = not control["engaged"]
                print(
                    "Finger tracking resumed"
                    if control["engaged"]
                    else "Finger tracking paused; the hand holds its pose"
                )
            elif pending == "reset":
                worker.reset()
                begin_episode(robot)
                adapter.reset()
                print(
                    f"Episode {len(hand_pose_episodes)} started; hand reopened"
                    if hand_pose_episodes is not None
                    else "Hand reopened; the next frame drives the fingers again"
                )
            worker.check_health()
            if emg_session is not None:
                emg_session.check_health()
            snapshot = worker.snapshot()
            now = time.monotonic()
            if snapshot is None:
                if now - started > args.initial_frame_timeout:
                    raise SourceUnavailableError(
                        f"No {source_label} hand articulation received within {args.initial_frame_timeout:.0f}s"
                    )
                if now - last_wait_message >= WAIT_MESSAGE_INTERVAL_SECONDS:
                    print(f"Still waiting for {source_label} hand tracking...")
                    last_wait_message = now
                _hold_reset_pose(robot, reset_positions)
                idle_step()
                steps += 1
                continue
            if not live:
                live = True
                print(f"Hand tracking live: the Sharpa fingers follow the {source_label} articulation")
            frame_age = now - snapshot.observation.articulation.receipt_timestamp
            if frame_age > args.maximum_frame_age:
                if args.stale_frame_policy == "error":
                    raise SourceUnavailableError(
                        f"{source_label} articulation is stale ({frame_age:.3f}s > {args.maximum_frame_age:.3f}s)"
                    )
                if not holding_stale:
                    print(f"{source_label} articulation is stale; holding the last finger pose")
                    holding_stale = True
                idle_step()
                steps += 1
                continue
            if holding_stale:
                print(f"{source_label} articulation resumed")
                holding_stale = False
            if not control["engaged"]:
                idle_step()
                steps += 1
                continue
            action = adapter.action(snapshot)
            sim_time_before_s = float(og.sim.current_time)
            action_apply_monotonic_ns = time.monotonic_ns()
            env.step(action)
            step_return_monotonic_ns = time.monotonic_ns()
            sim_time_after_s = float(og.sim.current_time)
            if action_timing_episodes is not None:
                action_timing_episodes[-1].append(
                    ActionTimingSample(
                        action_apply_monotonic_ns=action_apply_monotonic_ns,
                        step_return_monotonic_ns=step_return_monotonic_ns,
                        sim_time_before_s=sim_time_before_s,
                        sim_time_after_s=sim_time_after_s,
                    )
                )
            if hand_pose_episodes is not None:
                # The fixed wrist never carries HTS raw values, so raw_available stays false.
                hand_pose_episodes[-1].append(HumanHandPoseSample.capture(snapshot.frame))
            if hand_tracking_session is not None:
                selection = snapshot.action_selection()
                if selection is None:
                    raise RuntimeError("Snapshot is missing hand-tracking recording provenance")
                hand_tracking_session.append_action_selection(selection)
            if emg_monitor is not None:
                emg_monitor.update()
            applied += 1
            steps += 1
        measured = adapter.measured_fingers
        print(
            f"Hand bench stopped after {steps} steps; {applied} finger actions applied; "
            f"measured finger joints span {min(measured.values()):.2f} to {max(measured.values()):.2f} rad"
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if emg_monitor is not None:
            try:
                emg_monitor.close()
            except BaseException as error:
                cleanup_errors.append(("EMG monitor", error))
        recorder_error: BaseException | None = None
        finalization_error: BaseException | None = None
        if recording_env is not None:
            try:
                worker.set_recording_session(None, timeout=10.0)
                hand_tracking_session.close()
            except BaseException as error:
                recorder_error = error
            if recorder_error is None:
                try:
                    _finalize_recording(
                        recording_env=recording_env,
                        staging_path=staging_path,
                        output_path=recording_path,
                        hand_pose_episodes=hand_pose_episodes,
                        action_timing_episodes=action_timing_episodes,
                        emg_session=emg_session,
                        hand_tracking_session=hand_tracking_session,
                        # Without an episode there is no trajectory to publish; keep the
                        # staging file for diagnosis instead of replacing a prior recording.
                        publish=primary_error is None or bool(hand_pose_episodes),
                    )
                except BaseException as error:
                    finalization_error = error
            else:
                try:
                    recording_env.save_data()
                except BaseException as error:
                    finalization_error = error
        messages = [f"Could not close {label}: {error}" for label, error in cleanup_errors]
        if recorder_error is not None:
            messages.append(f"Could not quiesce the hand-tracking recorder: {recorder_error}")
        if finalization_error is not None:
            messages.append(f"Recording finalization failed: {finalization_error}")
        if primary_error is not None:
            for message in messages:
                primary_error.add_note(message)
                LOGGER.error(message)
        elif recorder_error is not None:
            failure = RuntimeError(f"Could not quiesce the hand-tracking recorder: {recorder_error}")
            for message in messages:
                if not message.startswith("Could not quiesce"):
                    failure.add_note(message)
            raise failure from recorder_error
        elif finalization_error is not None:
            for message in messages:
                if not message.startswith("Recording finalization failed"):
                    finalization_error.add_note(message)
            raise finalization_error
        elif cleanup_errors:
            failure = RuntimeError("Hand bench resources failed to close cleanly")
            for message in messages:
                failure.add_note(message)
            raise failure from cleanup_errors[0][1]


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Kit exits the process without flushing block-buffered pipes; keep the
    # status lines when stdout is redirected to a log file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    reject_wrist_options(argv)
    args = _parser().parse_args(argv)
    _validate_args(args)
    scene = load_hand_bench_scene()
    _force_data_path()
    validate_runtime_assets(scene)
    camera_rig = load_camera_rig(args.camera_rig)
    layout = camera_rig.layout(CAMERA_LAYOUT)

    import omnigibson as og
    from omnigibson.macros import gm

    # Rigid bodies only (no fluids or cloth), so CPU PhysX keeps the GPU footprint
    # small enough to share the card with other renders.
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = False
    gm.USE_PBR_MATERIALS = True
    gm.GUI_VIEWPORT_ONLY = bool(layout.get("workspace", {}).get("viewport_only", False))
    # The bench always has a robot, so it reuses Kit's main viewport like ARAT
    # teleoperation instead of OmniGibson's viewer-camera render product.
    gm.RENDER_VIEWER_CAMERA = False
    logging.basicConfig(level=logging.INFO)

    tracking_enabled = not args.view_only and args.screenshot_dir is None
    recording_path = None
    if args.record and tracking_enabled:
        recording_path = Path(args.recording_path).expanduser() if args.recording_path else default_recording_path()
    worker = None
    emg_session = None
    shutdown = GracefulShutdown()
    launcher_error: BaseException | None = None
    try:
        if args.emg:
            # Establish the Bluetooth connection before the long Isaac Sim startup.
            emg_session = create_emg_session(args, recording_path)
            print(f"Starting OYMotion EMG acquisition (staging: {emg_session.output_path.name})")
            emg_session.start(timeout=args.emg_connect_timeout)
            print(
                f"EMG ready: {emg_session.metadata.get('device_name', 'OYMotion')} | "
                f"{emg_session.channel_count} ch @ {emg_session.sample_rate_hz:g} Hz"
            )
            print(f"EMG firmware filters: {emg_session.metadata.get('filter_configuration', 'unknown')}")
            if args.visualize_decoder:
                print(
                    f"EMG2Pose target: {emg_session.metadata.get('decoder_inference_hz_target', 5.0):g} Hz | "
                    f"device: {emg_session.metadata.get('decoder_device', 'unknown')}"
                )
        if tracking_enabled:
            worker = create_worker(args)
            worker.start()
        run_bench(scene, args, worker, shutdown, recording_path=recording_path, emg_session=emg_session)
    except KeyboardInterrupt:
        print("\nStopping the Sharpa hand bench")
    except BaseException as error:
        launcher_error = error
        if isinstance(error, Exception):
            print("\nHand bench failed before shutdown:", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.stderr.flush()
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        for label, close in (
            ("tracking worker", None if worker is None else worker.close),
            ("EMG session", None if emg_session is None else emg_session.close),
            ("OmniGibson", lambda: _shutdown_omnigibson(og)),
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as error:
                cleanup_errors.append((label, error))
        messages = [f"Could not close {label}: {error}" for label, error in cleanup_errors]
        if launcher_error is not None:
            for message in messages:
                launcher_error.add_note(message)
                LOGGER.error(message)
        elif cleanup_errors:
            failure = RuntimeError("Hand bench resources failed to close cleanly")
            for message in messages:
                failure.add_note(message)
            raise failure from cleanup_errors[0][1]


if __name__ == "__main__":
    main()

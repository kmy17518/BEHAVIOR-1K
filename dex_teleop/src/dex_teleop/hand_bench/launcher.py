"""Finger-tracking bench: a fixed right Sharpa hand above a table in OmniGibson.

The bench accepts an articulation source only (Quest HTS or a MANUS glove).
Wrist tracking is disabled by construction: the Franka that carries the Sharpa
hand is hidden and held by a ``NullJointController``, the action space is the
22 finger joints, and the runtime pins the wrist through ``FixedWristSource``.
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
from dex_teleop.omnigibson.launcher import (
    GracefulShutdown,
    _configure_camera_layout,
    _show_robot_end_effectors,
    _shutdown_omnigibson,
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
from dex_teleop.tracking.manus import ManusIntegratedSource
from dex_teleop.types import Handedness


LOGGER = logging.getLogger(__name__)
HAND_SOURCES = ("quest", "hts", "manus")
SCREENSHOT_SETTLE_FRAMES = 90
ARM_HOLD_TOLERANCE_RAD = 0.05
WAIT_MESSAGE_INTERVAL_SECONDS = 5.0

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
    return parser


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


def run_bench(scene: HandBenchScene, args, worker: MultiSourceTrackingWorker | None, shutdown: GracefulShutdown) -> None:
    import omnigibson as og
    import omnigibson.lazy as lazy
    import torch as th
    from omnigibson.macros import gm
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    from dex_teleop.omnigibson.sharpa_finger_adapter import (
        SharpaFingerActionAdapter,
        SharpaFingerAdapterConfig,
    )

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
    env.reset()
    robot = env.robots[0]
    table = env.scene.object_registry("name", TABLE_NAME)
    if table is None:
        raise RuntimeError("Hand bench scene did not load its table")
    reset_positions = th.tensor(reset_joint_positions(scene), dtype=th.float32)
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
    print(
        f"Waiting for the first {source_label} hand frame (wrist tracking disabled; fingers only). "
        "Press SPACE to pause/resume, R to reopen the hand, B to switch views; Ctrl+C exits."
    )
    started = time.monotonic()
    last_wait_message = started
    live = False
    holding_stale = False
    applied = 0
    while (args.steps <= 0 or steps < args.steps) and not shutdown.requested:
        pending = control["pending"]
        control["pending"] = None
        if pending == "toggle":
            control["engaged"] = not control["engaged"]
            print("Finger tracking resumed" if control["engaged"] else "Finger tracking paused; the hand holds its pose")
        elif pending == "reset":
            worker.reset()
            _hold_reset_pose(robot, reset_positions)
            adapter.reset()
            print("Hand reopened; the next frame drives the fingers again")
        worker.check_health()
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
            og.sim.step()
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
            og.sim.step()
            steps += 1
            continue
        if holding_stale:
            print(f"{source_label} articulation resumed")
            holding_stale = False
        if not control["engaged"]:
            og.sim.step()
            steps += 1
            continue
        env.step(adapter.action(snapshot))
        applied += 1
        steps += 1
    measured = adapter.measured_fingers
    print(
        f"Hand bench stopped after {steps} steps; {applied} finger actions applied; "
        f"measured finger joints span {min(measured.values()):.2f} to {max(measured.values()):.2f} rad"
    )


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
    worker = None
    shutdown = GracefulShutdown()
    launcher_error: BaseException | None = None
    try:
        if tracking_enabled:
            worker = create_worker(args)
            worker.start()
        run_bench(scene, args, worker, shutdown)
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

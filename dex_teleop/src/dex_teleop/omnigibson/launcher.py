"""One-process ARAT BehaviorTask launcher for landmark hand teleoperation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from dex_teleop.arat import AratTask, AratTaskCatalog
from dex_teleop.arat.camera_rig import (
    apply_camera_lens_models,
    build_camera_sensor_configs,
    camera_rig_names,
    layout_camera_ids,
    load_camera_rig,
)
from dex_teleop.arat.eval import AratSessionScorer, load_rubrics
from dex_teleop.arat.eval.report import (
    format_item_summary,
    format_session_summary,
    make_results_dir,
    write_item_result,
    write_session_result,
)
from dex_teleop.arat.reset_poses import (
    DEFAULT_RESET_POSE,
    add_reset_pose_argument,
    reset_joint_positions,
)
from dex_teleop.arat.scene import (
    ROBOT_DATASET_NAME,
    ROBOT_END_EFFECTOR,
    ROBOT_NAME,
    ROBOT_MODEL,
    get_task_scene_data,
    get_task_scene_path,
    validate_runtime_assets,
)
from dex_teleop.emg import (
    ActionTimingSample,
    EmgSession,
    emg_staging_path,
    merge_emg_recording,
    write_action_timing_episodes,
)
from dex_teleop.omnigibson.evaluation_trace import build_step_evaluation, write_evaluation_episodes
from dex_teleop.omnigibson.hand_pose_recording import HumanHandPoseSample, write_hand_pose_episodes
from dex_teleop.retargeting import LandmarkRetargeter
from dex_teleop.runtime import TrackingRetargetingWorker
from dex_teleop.tracking import HTSSource
from dex_teleop.tracking.ovxr import create_ovxr_source
from dex_teleop.types import Handedness


LOGGER = logging.getLogger(__name__)
ROBOT_COMPOSED_MODEL = f"{ROBOT_MODEL}_{ROBOT_END_EFFECTOR}"
ROBOT_PRIM_PATH = f"/controllable__{ROBOT_COMPOSED_MODEL}__{ROBOT_NAME}"
CAMERA_TOGGLE_DEBOUNCE_SECONDS = 0.25
DEFAULT_RECORDING_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "recordings"


class GracefulShutdown:
    """Turn SIGINT into a stop request so recording can finish before Kit shuts down."""

    def __init__(self) -> None:
        self.requested = False

    def install(self) -> None:
        # OmniGibson installs its own handler every time the simulator launches.
        # Install ours only after Environment construction so it wins.
        signal.signal(signal.SIGINT, self)

    def __call__(self, _signum, _frame) -> None:
        if not self.requested:
            print("\nCtrl+C received; finishing the current simulator operation and saving the recording...", flush=True)
        self.requested = True


def _parser(catalog: AratTaskCatalog) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", choices=tuple(catalog.tasks))
    selection.add_argument("--subscale", choices=tuple(catalog.subscales))
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument(
        "--camera-rig",
        choices=camera_rig_names(),
        help="Override the task catalog's camera rig for this launch",
    )
    parser.add_argument("--source", choices=("hts", "ovxr"), default="hts")
    parser.add_argument("--hand-model", choices=("shadow", "sharpa", "wuji"), default="sharpa")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--protocol", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--auto-anchor", action="store_true")
    parser.add_argument(
        "--recording-path",
        help=(
            "Record to this HDF5 file (single-task only); defaults to "
            "dex_teleop/outputs/recordings/<task>.hdf5"
        ),
    )
    parser.add_argument(
        "--record-hand-poses",
        action="store_true",
        help=(
            "Store action-aligned pre-retargeting wrist poses and 21 hand landmarks "
            "in the recording HDF5"
        ),
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
        help="Record EMG without creating the native OmniGibson waveform dock",
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
    parser.add_argument(
        "--emg2pose-device",
        help="Torch decoder device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--emg2pose-inference-hz",
        type=float,
        help="Target decoded-hand update rate (default: 5 Hz; 10-15 Hz is practical on CUDA)",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Load the selected saved scene without a robot or hand-tracking source",
    )
    add_reset_pose_argument(parser)
    parser.add_argument(
        "--assisted-grasp",
        action="store_true",
        help=(
            "Weld an object to the hand while the teleoperated fingers hold it in opposition "
            "(OmniGibson assisted grasping, driven by the fingers actually touching the object "
            "instead of the robot definition's fixed finger pairs)"
        ),
    )
    parser.add_argument(
        "--visualize-arm-markers",
        "--visualize-arm-workspace",
        dest="visualize_arm_markers",
        action="store_true",
        help=(
            "Start the independently toggleable commanded-target and measured-EEF XYZ frames visible "
            "(--visualize-arm-workspace is a deprecated alias)"
        ),
    )
    parser.add_argument("--steps", type=int, default=0, help="Single-task step limit; 0 runs until interrupted")
    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=3600,
        help="Fixed duration for every activity selected through --subscale",
    )
    parser.add_argument("--maximum-frame-age", type=float, default=0.25)
    parser.add_argument(
        "--stale-frame-policy",
        choices=("error", "hold"),
        default="error",
        help="Error on a stale HTS frame, or hold the robot until fresh tracking resumes",
    )
    parser.add_argument("--initial-frame-timeout", type=float, default=10.0)
    parser.add_argument(
        "--tracking-yaw-deg",
        type=float,
        default=0.0,
        help="Rotate tracking translation and orientation about world Z before anchoring",
    )
    parser.add_argument(
        "--position-sensitivity",
        type=float,
        default=1.5,
        help="Scale tracked wrist translation about the engagement anchor (default: 1.5)",
    )
    parser.add_argument(
        "--wrist-flip-output",
        help="Enable wrist-flip diagnostics and write trace.csv, events.json, and summary.md here",
    )
    parser.add_argument("--wrist-flip-threshold-deg", type=float, default=90.0)
    parser.add_argument("--wrist-pair-skew-threshold-ms", type=float, default=20.0)
    parser.add_argument("--wrist-joint-rotation-threshold-deg", type=float, default=150.0)
    parser.add_argument("--wrist-joint-rotation-window-s", type=float, default=5.0)
    parser.add_argument("--wrist-joint-limit-margin-deg", type=float, default=5.0)
    parser.add_argument("--wrist-flip-post-event-seconds", type=float, default=2.0)
    parser.add_argument(
        "--continue-after-wrist-flip",
        action="store_true",
        help="Keep teleoperating after a detected or manually marked wrist-flip event",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Disable the ARAT 0-3 scorer (scoring is on by default during teleoperation)",
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[3] / "outputs" / "arat_results"),
        help="Base directory for per-item and per-session ARAT result JSON files",
    )
    return parser


def _print_catalog(catalog: AratTaskCatalog) -> None:
    for subscale, activities in catalog.subscales.items():
        print(f"{subscale} ({len(activities)} tasks)")
        for activity in activities:
            task = catalog.tasks[activity]
            print(f"  {activity}: {task.label} [layout={task.layout}]")


def _create_source(args):
    if args.source == "hts":
        return HTSSource(host=args.host, port=args.port, protocol=args.protocol)
    if args.source == "ovxr":
        return create_ovxr_source()
    raise ValueError(f"Unsupported tracking source {args.source!r}")


def build_environment_config(
    task: AratTask,
    *,
    view_only: bool = False,
    assisted_grasp: bool = False,
    reset_pose: str = DEFAULT_RESET_POSE,
) -> dict:
    """Build an environment from the selected version-1 saved scene."""

    scene_file = get_task_scene_data(task, include_task_metadata=not view_only)
    robot_pose = None if view_only else scene_file["metadata"]["task"]["robot_poses"]["robot"][0]
    camera_rig = load_camera_rig(task.camera_rig)
    camera_layout = "view_only" if view_only else "teleop"
    # Sharing Kit's main viewport prevents VisionSensor from creating an
    # undocked window per camera. The configured layout assigns them below.
    viewport_name = None if os.environ.get("OMNIGIBSON_HEADLESS") == "1" else "Viewport"
    config = {
        # OmniGibson's default rendering frequency is 30 Hz, so the action
        # period must be a multiple of 1 / 30 seconds.
        "env": {
            "action_frequency": 30.0,
            "automatic_reset": False,
            "external_sensors": build_camera_sensor_configs(
                camera_rig,
                camera_layout,
                robot_prim_path=ROBOT_PRIM_PATH,
                viewport_name=viewport_name,
            ),
        },
        "scene": {
            "type": "Scene",
            "scene_file": scene_file,
            "use_floor_plane": True,
            "floor_plane_visible": True,
            "floor_plane_color": [0.5, 0.5, 0.5],
            "use_skybox": True,
            "include_robots": False,
        },
        "objects": [],
        "robots": [] if view_only else [
            {
                "model": ROBOT_MODEL,
                "dataset_name": ROBOT_DATASET_NAME,
                "end_effector": ROBOT_END_EFFECTOR,
                "name": ROBOT_NAME,
                "position": list(robot_pose["position"]),
                "orientation": list(robot_pose["orientation"]),
                "grasping_direction": "upper",
                # Assisted mode enables OmniGibson's weld machinery and its state
                # serialization (recordings/replay); the launcher's supervisor makes the
                # grasp/release decisions, so the built-in handling stays disabled.
                "grasping_mode": "assisted" if assisted_grasp else "physical",
                "disable_grasp_handling": assisted_grasp,
                "obs_modalities": ["rgb"],
                "action_normalize": False,
                "fixed_base": True,
                "self_collisions": False,
                "reset_joint_pos": reset_joint_positions(reset_pose),
                "controller_config": {
                    "arm_0": {
                        "name": "InverseKinematicsController",
                        "mode": "absolute_pose",
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "mode": "independent",
                        "motor_type": "position",
                        "inverted": False,
                        "command_input_limits": None,
                        "command_output_limits": None,
                    },
                },
            }
        ],
        "task": {"type": "DummyTask"} if view_only else {
            "type": "BehaviorTask",
            "activity_name": task.activity,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": False,
            "use_presampled_robot_pose": True,
            "highlight_task_relevant_objects": False,
            "termination_config": {"max_steps": 50000},
            "reward_config": {"r_potential": 1.0},
            "include_obs": False,
        },
    }
    return config


def _validate_loaded_apparatus(env, task: AratTask) -> None:
    if task.subscale == "gross_movement":
        mannequin = env.scene.object_registry("name", "mannequin")
        if mannequin is None or mannequin.category != "mannequin" or mannequin.model != "nphsfp":
            raise RuntimeError("ARAT gross-movement scene did not load mannequin/nphsfp")
        # The saved gross-movement layout contains only the mannequin, but the
        # normal teleoperation path adds the intended robot to the live scene.
        allowed_names = {"mannequin", *(robot.name for robot in env.robots)}
        unexpected = set(env.scene.object_registry.get_dict("name")).difference(allowed_names)
        if unexpected:
            raise RuntimeError(f"ARAT gross-movement scene loaded unexpected objects: {sorted(unexpected)}")
        return
    table = env.scene.object_registry("name", "table")
    if table is None or table.category != "breakfast_table" or table.model != "nvoqyl":
        raise RuntimeError("ARAT scene did not load the resized breakfast_table/nvoqyl")
    box = env.scene.object_registry("name", "arat_box")
    if box is None or box.category != "arat_box" or box.model != "aratbx":
        raise RuntimeError("ARAT scene did not load the articulated arat_box/aratbx")


# Cap for how fast PhysX resolves penetration on graspable objects (m/s). A welded object
# can be dragged inside static geometry; without a cap the accumulated penetration converts
# into an enormous ejection velocity the moment the weld releases.
OBJECT_MAX_DEPENETRATION_VELOCITY = 3.0


def _limit_depenetration_velocity(env, maximum: float = OBJECT_MAX_DEPENETRATION_VELOCITY) -> list[str]:
    """Author physxRigidBody:maxDepenetrationVelocity on every dynamic scene object."""

    import omnigibson as og
    import omnigibson.lazy as lazy

    limited = []
    with og.sim.editing_usd():
        for obj in env.scene.objects:
            if obj in env.robots or getattr(obj, "fixed_base", False) or getattr(obj, "kinematic_only", False):
                continue
            links = getattr(obj, "links", None) or {}
            touched = False
            for link in links.values():
                api = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(link.prim)
                if not api:
                    continue
                api.CreateMaxDepenetrationVelocityAttr().Set(float(maximum))
                touched = True
            if touched:
                limited.append(obj.name)
    return limited


def _reset_arat_box(env) -> None:
    """Match the articulated-box setup used by run_hand_teleop_hts_arat.py."""

    box = env.scene.object_registry("name", "arat_box")
    if box is None:
        return
    box.joints["front_cover_joint"].friction = 20.0
    for joint in box.joints.values():
        joint.set_pos(0.0)
        joint.set_vel(0.0)


def _show_robot_end_effectors(robot) -> None:
    """Restore end-effector visuals hidden by OmniGibson during robot initialization."""

    for arm_name in robot.arm_names:
        robot.links[robot.eef_link_names[arm_name]].visible = True


def _hide_skybox_from_camera() -> None:
    """Keep the prior setup's dome lighting while rendering a black background."""

    import omnigibson as og
    import omnigibson.lazy as lazy

    skybox = og.sim.skybox
    if skybox is None:
        return
    dome_prim = skybox.light_link.prim
    with og.sim.editing_usd():
        attr = dome_prim.GetAttribute("visibleInPrimaryRay")
        if not attr:
            attr = dome_prim.CreateAttribute("visibleInPrimaryRay", lazy.pxr.Sdf.ValueTypeNames.Bool)
        attr.Set(False)


def _create_and_dock_camera_viewport(
    name: str,
    camera_path: str,
    dock_position,
    dock_ratio: float,
    resolution: tuple[int, int],
    dock_parent_name: str = "DockSpace",
    fill_frame: bool = False,
):
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.utils.ui_utils import dock_window

    viewports = {
        viewport.name: viewport for viewport in lazy.omni.kit.viewport.window.get_viewport_window_instances()
    }
    viewport = viewports.get(name)
    if viewport is None:
        with og.sim.editing_usd():
            viewport = lazy.omni.kit.viewport.utility.create_viewport_window(name=name)
        og.sim.render()

    dock_window(
        space=lazy.omni.ui.Workspace.get_window(dock_parent_name),
        name=viewport.name,
        location=dock_position,
        ratio=dock_ratio,
    )
    viewport.viewport_api.set_active_camera(camera_path)
    _set_viewport_resolution(viewport, resolution, fill_frame=fill_frame)
    og.sim.render()
    return viewport


def _set_viewport_resolution(viewport, resolution: tuple[int, int], *, fill_frame: bool) -> None:
    """Keep Kit's viewport widget and backing render texture on the same aspect ratio."""

    # ViewportAPI.set_texture_resolution() changes Hydra's backing texture but
    # leaves ViewportWidget.full_resolution at the window's initial 1280x720.
    # ImageWithProvider then continues presenting a 16:9 image even when Hydra
    # renders a square texture. Drive the widget so both pieces are updated.
    viewport.viewport_api.fill_frame = False
    viewport.viewport_widget.fill_frame = False
    viewport.viewport_widget.resolution = resolution
    if fill_frame:
        viewport.viewport_widget.fill_frame = True
        viewport.viewport_api.fill_frame = True


def _window_name(window) -> str:
    """Return the Kit workspace name for a viewport or an ``omni.ui.Window``."""

    name = getattr(window, "name", None) or getattr(window, "title", None)
    if not name:
        raise RuntimeError(f"Cannot resolve Kit window name for {window!r}")
    return name


def _set_workspace_windows_visible(window_names: tuple[str, ...], visible: bool) -> None:
    """Show or hide named Kit windows and allow its dock tree to reflow."""

    import omnigibson as og
    import omnigibson.lazy as lazy

    for name in window_names:
        window = lazy.omni.ui.Workspace.get_window(name)
        if window is not None:
            window.visible = visible
            og.app.update()


def _create_and_dock_empty_window(
    name: str,
    dock_position,
    dock_ratio: float,
    dock_parent_name: str,
):
    """Create a blank reserved panel in a YAML-defined viewport slot."""

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.utils.ui_utils import dock_window

    window = lazy.omni.ui.Window(name, width=320, height=240)
    with window.frame:
        lazy.omni.ui.Rectangle(style={"background_color": 0xFF111111})
    dock_window(
        space=lazy.omni.ui.Workspace.get_window(dock_parent_name),
        name=window.title,
        location=dock_position,
        ratio=dock_ratio,
    )
    og.sim.render()
    return window


def _camera_layout_panel_dock(
    env,
    panel_id: str,
    *,
    default_parent: str,
    default_position: str,
    default_ratio: float,
) -> tuple[str, str, float]:
    """Resolve a utility-panel dock target from the active camera layout."""

    panel = getattr(env, "_arat_camera_layout", {}).get("panels", {}).get(panel_id, {})
    dock = panel.get("dock", {})
    parent = dock.get("parent", default_parent)
    layout_windows = getattr(env, "_arat_camera_windows", {})
    if parent in layout_windows:
        parent = _window_name(layout_windows[parent])
    return parent, dock.get("position", default_position), float(dock.get("ratio", default_ratio))


def _configure_camera_layout(env, camera_rig_name: str, layout_name: str, robot=None) -> None:
    """Apply one YAML viewport layout and register its camera-cycle keys."""

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.macros import gm
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    camera_rig = load_camera_rig(camera_rig_name)
    layout = camera_rig.layout(layout_name)
    fill_viewports = layout.get("workspace", {}).get("fill_viewports", False)
    sensors = env.external_sensors or {}
    camera_ids = layout_camera_ids(camera_rig, layout_name)
    expected = {camera_rig.camera(camera_id)["sensor_name"] for camera_id in camera_ids}
    missing = expected.difference(sensors)
    if missing:
        raise RuntimeError(f"ARAT camera layout is missing sensors: {sorted(missing)}")

    camera_paths = {
        camera_id: sensors[camera_rig.camera(camera_id)["sensor_name"]].prim_path
        for camera_id in camera_ids
    }
    apply_camera_lens_models(camera_rig, sensors, camera_ids)
    for camera_id in camera_ids:
        parent = camera_rig.camera(camera_id)["parent"]
        if parent["frame"] != "robot_link":
            continue
        if robot is None:
            raise RuntimeError(f"Camera {camera_id!r} requires a robot but layout {layout_name!r} has none")
        expected_parent = robot.links[parent["link"]].prim_path
        if not camera_paths[camera_id].startswith(f"{expected_parent}/"):
            raise RuntimeError(
                f"ARAT camera {camera_id!r} is not parented to {parent['link']}: {camera_paths[camera_id]}"
            )

    env._arat_camera_layout = layout
    env._arat_camera_rig_name = camera_rig_name
    env._arat_camera_windows = {}
    if gm.HEADLESS:
        return

    if robot is None:
        main_viewport = og.sim.viewer_camera._viewport
    else:
        # Normal teleoperation launches without OmniGibson's viewer-camera
        # sensor. Reuse Kit's initially hidden main viewport directly so no
        # default-camera render product or startup view remains active.
        viewports = {
            viewport.name: viewport for viewport in lazy.omni.kit.viewport.window.get_viewport_window_instances()
        }
        main_viewport = viewports.get("Viewport")
        if main_viewport is None:
            raise RuntimeError("ARAT camera layout could not find Kit's main Viewport")
        with og.sim.editing_usd():
            main_viewport.visible = True

    main_camera_id = layout["viewports"]["main"]["camera"]
    main_calibration = camera_rig.calibration(main_camera_id)
    main_viewport.viewport_api.set_active_camera(camera_paths[main_camera_id])
    _set_viewport_resolution(
        main_viewport,
        (main_calibration["image_width"], main_calibration["image_height"]),
        fill_frame=fill_viewports,
    )

    camera_viewports = {"main": main_viewport}
    layout_windows = {"main": main_viewport}
    empty_windows = {}
    pending_viewports = {
        viewport_id: viewport
        for viewport_id, viewport in layout["viewports"].items()
        if viewport_id != "main"
    }
    dock_positions = {
        "left": lazy.omni.ui.DockPosition.LEFT,
        "right": lazy.omni.ui.DockPosition.RIGHT,
        "top": lazy.omni.ui.DockPosition.TOP,
        "bottom": lazy.omni.ui.DockPosition.BOTTOM,
    }
    while pending_viewports:
        configured = []
        for viewport_id, viewport_config in pending_viewports.items():
            dock = viewport_config["dock"]
            dock_parent = dock["parent"]
            if dock_parent != "DockSpace" and dock_parent not in layout_windows:
                continue
            dock_parent_name = (
                "DockSpace" if dock_parent == "DockSpace" else _window_name(layout_windows[dock_parent])
            )
            if viewport_config.get("empty", False):
                window = _create_and_dock_empty_window(
                    name=viewport_config["window_name"],
                    dock_position=dock_positions[dock["position"]],
                    dock_ratio=dock["ratio"],
                    dock_parent_name=dock_parent_name,
                )
                empty_windows[viewport_id] = window
            else:
                camera_id = viewport_config["camera"]
                calibration = camera_rig.calibration(camera_id)
                window = _create_and_dock_camera_viewport(
                    name=viewport_config["window_name"],
                    camera_path=camera_paths[camera_id],
                    dock_position=dock_positions[dock["position"]],
                    dock_ratio=dock["ratio"],
                    resolution=(calibration["image_width"], calibration["image_height"]),
                    dock_parent_name=dock_parent_name,
                    fill_frame=fill_viewports,
                )
                camera_viewports[viewport_id] = window
            layout_windows[viewport_id] = window
            configured.append(viewport_id)
        if not configured:
            raise RuntimeError(f"Camera layout {layout_name!r} has cyclic viewport docking dependencies")
        for viewport_id in configured:
            del pending_viewports[viewport_id]

    env._arat_camera_viewports = camera_viewports
    env._arat_camera_windows = layout_windows
    env._arat_empty_camera_windows = empty_windows
    toggle_descriptions = []
    for toggle in layout.get("toggles", []):
        key_name = toggle["key"].upper()
        keyboard_key = getattr(lazy.carb.input.KeyboardInput, key_name, None)
        if keyboard_key is None:
            raise RuntimeError(f"Camera layout {layout_name!r} uses unsupported keyboard key {key_name!r}")
        viewport = camera_viewports[toggle["viewport"]]
        cycle = tuple(toggle["cameras"])
        state = {"index": 0, "last_toggle": float("-inf")}

        def cycle_camera(
            *,
            _viewport=viewport,
            _viewport_id=toggle["viewport"],
            _cycle=cycle,
            _state=state,
            _key=key_name,
        ):
            now = time.monotonic()
            if now - _state["last_toggle"] < CAMERA_TOGGLE_DEBOUNCE_SECONDS:
                return
            _state["last_toggle"] = now
            _state["index"] = (_state["index"] + 1) % len(_cycle)
            camera_id = _cycle[_state["index"]]
            calibration = camera_rig.calibration(camera_id)
            _viewport.viewport_api.set_active_camera(camera_paths[camera_id])
            _set_viewport_resolution(
                _viewport,
                (calibration["image_width"], calibration["image_height"]),
                fill_frame=fill_viewports,
            )
            print(f"Camera {_viewport_id} [{_key}]: {camera_id}")

        KeyboardEventHandler.add_keyboard_callback(keyboard_key, cycle_camera)
        toggle_descriptions.append(f"{key_name} cycles {toggle['viewport']} ({' / '.join(cycle)})")

    for _ in range(3):
        og.sim.render()
    print(f"Camera layout ready: {camera_rig_name}/{layout_name}")
    if toggle_descriptions:
        print("Camera controls: " + "; ".join(toggle_descriptions))


def _create_goal_status_ui(env):
    """Create a JoyLo-style overlay for the task's natural-language BDDL goals."""

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.macros import gm

    goal_conditions = env.task.activity_natural_language_goal_conditions
    if gm.HEADLESS:
        return None, []

    main_viewport = env._arat_camera_viewports["main"]
    main_viewport.dock_tab_bar_visible = False
    og.sim.render()
    overlay_window = lazy.omni.ui.Window(
        main_viewport.name,
        width=0,
        height=0,
        flags=lazy.omni.ui.WINDOW_FLAGS_NO_TITLE_BAR
        | lazy.omni.ui.WINDOW_FLAGS_NO_SCROLLBAR
        | lazy.omni.ui.WINDOW_FLAGS_NO_RESIZE,
    )
    labels = []
    with overlay_window.frame:
        with lazy.omni.ui.ZStack():
            lazy.omni.ui.Spacer()
            with lazy.omni.ui.VStack(alignment=lazy.omni.ui.Alignment.LEFT_TOP, spacing=0):
                lazy.omni.ui.Spacer(height=50)
                for condition in goal_conditions:
                    with lazy.omni.ui.HStack(height=20):
                        lazy.omni.ui.Spacer(width=50)
                        label = lazy.omni.ui.Label(
                            condition,
                            alignment=lazy.omni.ui.Alignment.LEFT_CENTER,
                            style={
                                "color": 0xFF0000FF,
                                "font_size": 25,
                                "margin": 0,
                                "padding": 0,
                                ":selected": {"color": 0xFF00FF00},
                            },
                        )
                        labels.append(label)
    og.sim.render()
    return overlay_window, labels


def _update_goal_status_labels(labels, goal_status) -> None:
    """Color satisfied BDDL goals green and unsatisfied goals red."""

    for index in goal_status["satisfied"]:
        if 0 <= index < len(labels):
            labels[index].selected = True
    for index in goal_status["unsatisfied"]:
        if 0 <= index < len(labels):
            labels[index].selected = False


def _reset_goal_status_labels(labels) -> None:
    for label in labels:
        label.selected = False


def default_recording_path(task: AratTask) -> Path:
    return DEFAULT_RECORDING_ROOT / f"{task.activity}.hdf5"


def recording_staging_path(output_path: str | Path, *, process_id: int | None = None) -> Path:
    """Return a same-directory temporary path suitable for atomic publication."""

    output_path = Path(output_path)
    process_id = os.getpid() if process_id is None else process_id
    return output_path.with_name(f".{output_path.name}.{process_id}.in_progress")


def _finalize_recording(
    recording_env,
    staging_path: Path,
    output_path: Path,
    evaluation_episodes: list[list[dict]] | None = None,
    hand_pose_episodes: list[list[HumanHandPoseSample]] | None = None,
    action_timing_episodes: list[list[ActionTimingSample]] | None = None,
    emg_session: EmgSession | None = None,
) -> None:
    """Close and augment the HDF5 file, then atomically publish it."""

    recording_env.save_data()
    if evaluation_episodes is not None:
        write_evaluation_episodes(staging_path, evaluation_episodes)
    if hand_pose_episodes is not None:
        write_hand_pose_episodes(staging_path, hand_pose_episodes)
    if action_timing_episodes is not None:
        write_action_timing_episodes(staging_path, action_timing_episodes)
    if emg_session is not None:
        emg_session.close()
        merge_emg_recording(staging_path, emg_session.output_path)
    os.replace(staging_path, output_path)
    if emg_session is not None:
        emg_session.remove_staging_file()
    print(f"Recording saved: {output_path}")


def _run_viewer(task: AratTask, step_limit: int, *, shutdown: GracefulShutdown | None = None) -> None:
    import omnigibson as og

    print(f"\nLoading {task.activity}: {task.label} (scene: {get_task_scene_path(task).name})")
    env = og.Environment(configs=build_environment_config(task, view_only=True))
    if shutdown is not None:
        shutdown.install()
    env.reset()
    _validate_loaded_apparatus(env, task)
    _reset_arat_box(env)
    _hide_skybox_from_camera()
    _configure_camera_layout(env, task.camera_rig, "view_only")
    print("Scene viewer ready; use the configured camera toggle keys or Ctrl+C to exit.")
    steps = 0
    while (step_limit <= 0 or steps < step_limit) and not (shutdown is not None and shutdown.requested):
        og.sim.step()
        steps += 1


def _run_task(
    task: AratTask,
    worker: TrackingRetargetingWorker,
    args,
    step_limit: int,
    *,
    enable_score: bool = False,
    placeholder_goals: bool = True,
    results_dir: Path | None = None,
    shutdown: GracefulShutdown | None = None,
    emg_session: EmgSession | None = None,
):
    import torch as th
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.envs import HDF5CollectionWrapper
    from omnigibson.macros import gm
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    from dex_teleop.omnigibson.sharpa_adapter import SharpaActionAdapter, SharpaAdapterConfig

    print(f"\nLoading {task.activity}: {task.label} (layout: {task.layout})")
    env = og.Environment(
        configs=build_environment_config(
            task,
            assisted_grasp=args.assisted_grasp,
            reset_pose=args.reset_pose,
        )
    )
    if shutdown is not None:
        shutdown.install()
    recording_path = Path(args.recording_path).expanduser() if args.recording_path else default_recording_path(task)
    staging_path = recording_staging_path(recording_path)
    # This is the same state/action trajectory wrapper used by JoyLo. ARAT
    # does not instantiate OmniGibson's viewer-camera sensor, so leave its
    # viewport optimization disabled and keep the custom camera layout.
    recording_env = HDF5CollectionWrapper(
        env=env,
        output_path=str(staging_path),
        viewport_camera_path=None,
        only_successes=False,
        flush_every_n_traj=1,
        keep_checkpoint_rollback_data=True,
    )
    env = recording_env
    print(f"Recording teleoperation to {recording_path} (staging: {staging_path.name})")
    evaluation_episodes = []
    hand_pose_episodes = [] if args.record_hand_poses or emg_session is not None else None
    action_timing_episodes = [] if emg_session is not None else None
    if hand_pose_episodes is not None:
        print("Recording action-aligned human hand poses under human_hand_pose/demo_N")
    if action_timing_episodes is not None:
        print("Recording action and simulator clock boundaries under action_timing/demo_N")
    wrist_recorder = None
    emg_monitor = None
    arm_marker_visualizer = None

    try:
        env.reset()
        evaluation_episodes.append([])
        if hand_pose_episodes is not None:
            hand_pose_episodes.append([])
        if action_timing_episodes is not None:
            action_timing_episodes.append([])
        _validate_loaded_apparatus(env, task)
        _reset_arat_box(env)
        _hide_skybox_from_camera()
        robot = env.robots[0]
        _show_robot_end_effectors(robot)
        tracking_yaw_half = math.radians(args.tracking_yaw_deg) / 2.0
        tracking_to_world_quaternion_xyzw = (
            0.0,
            0.0,
            math.sin(tracking_yaw_half),
            math.cos(tracking_yaw_half),
        )
        adapter = SharpaActionAdapter(
            robot,
            config=SharpaAdapterConfig(
                control_hz=30.0,
                position_sensitivity=args.position_sensitivity,
                tracking_to_world_quaternion_xyzw=tracking_to_world_quaternion_xyzw,
            ),
        )
        print(f"Wrist position sensitivity: {args.position_sensitivity:g}x")
        if args.tracking_yaw_deg:
            print(f"Tracking frame yaw: {args.tracking_yaw_deg:g} degrees")
        if not gm.HEADLESS:
            from dex_teleop.omnigibson.workspace_visualization import ArmPoseMarkerVisualizer

            marker_draw = lazy.isaacsim.util.debug_draw._debug_draw.acquire_debug_draw_interface()
            arm_marker_visualizer = ArmPoseMarkerVisualizer(
                marker_draw,
                target_visible=args.visualize_arm_markers,
                eef_visible=args.visualize_arm_markers,
            )
            initial_marker_state = "visible" if args.visualize_arm_markers else "hidden"
            print(
                f"Arm pose markers start {initial_marker_state}: target origin=magenta, measured EEF origin=cyan, "
                "with pose-oriented wrist rings and axes X=red/Y=green/Z=blue. "
                "Press T to toggle the target marker and E to toggle the EEF marker."
            )
        if args.wrist_flip_output is not None:
            from dex_teleop.diagnostics import WristFlipDiagnosticConfig, WristFlipRecorder

            wrist_recorder = WristFlipRecorder(
                args.wrist_flip_output,
                WristFlipDiagnosticConfig(
                    flip_threshold_degrees=args.wrist_flip_threshold_deg,
                    pair_skew_threshold_ms=args.wrist_pair_skew_threshold_ms,
                    post_event_seconds=args.wrist_flip_post_event_seconds,
                    continue_after_event=args.continue_after_wrist_flip,
                    tracking_yaw_degrees=args.tracking_yaw_deg,
                    joint_rotation_threshold_degrees=args.wrist_joint_rotation_threshold_deg,
                    joint_rotation_window_seconds=args.wrist_joint_rotation_window_s,
                    joint_limit_margin_threshold_degrees=args.wrist_joint_limit_margin_deg,
                ),
            )
            print(f"Wrist-flip diagnostics: {wrist_recorder.output_dir}")
            diagnostic_arm_indices = robot.arm_control_idx[adapter.arm_name]
            joint_lower_limits, joint_upper_limits = robot.joint_position_limits
            diagnostic_arm_lower_limits = joint_lower_limits[diagnostic_arm_indices].cpu().numpy().copy()
            diagnostic_arm_upper_limits = joint_upper_limits[diagnostic_arm_indices].cpu().numpy().copy()
        else:
            diagnostic_arm_indices = None
            diagnostic_arm_lower_limits = None
            diagnostic_arm_upper_limits = None
        grasp_supervisor = None
        if args.assisted_grasp:
            from dex_teleop.arat.eval.hand_model import HandSemantics
            from dex_teleop.omnigibson.assisted_grasp import AssistedGraspSupervisor

            grasp_supervisor = AssistedGraspSupervisor(
                robot,
                HandSemantics.sharpa(args.hand),
                dt=1.0 / 30.0,  # the env's action period (action_frequency 30 Hz)
                announce=print,
            )
            print(
                "Assisted grasping enabled: thumb-involved opposing contacts weld the object to the "
                "hand; grasping digits then hold their pose until the hand opens past its grasp posture"
            )
            limited = _limit_depenetration_velocity(env)
            print(
                f"Capped depenetration velocity at {OBJECT_MAX_DEPENETRATION_VELOCITY} m/s for: "
                f"{', '.join(limited) if limited else 'no dynamic objects'}"
            )
        evaluator = None
        if enable_score:
            from dex_teleop.arat.eval.live import LiveAratEvaluator

            rubric = load_rubrics()[task.activity]
            evaluator = LiveAratEvaluator(env, robot, task, rubric)
            print(
                f"ARAT scoring enabled for {task.activity}: "
                f"{evaluator.config.time_limit_s:.0f}s engaged-time limit, <{evaluator.config.score3_time_s:.0f}s for a 3"
            )
        _configure_camera_layout(env, task.camera_rig, "teleop", robot)
        if emg_session is not None:
            from dex_teleop.emg.ui import create_emg_monitor

            if not args.no_emg_display:
                # JoyLo's viewport-only workspace hides these lower tabs. Bring
                # their shared dock back only when it has live EMG content.
                _set_workspace_windows_visible(("Content", "Console"), True)
                og.sim.render()
            default_emg_parent = (
                "DockSpace" if gm.HEADLESS else _window_name(env._arat_camera_viewports["main"])
            )
            emg_dock_parent, emg_dock_position, emg_dock_ratio = _camera_layout_panel_dock(
                env,
                "emg",
                default_parent=default_emg_parent,
                default_position="right",
                default_ratio=0.5,
            )
            decoder_dock_parent, decoder_dock_position, decoder_dock_ratio = _camera_layout_panel_dock(
                env,
                "decoder",
                default_parent="OYMotion EMG",
                default_position="bottom",
                default_ratio=0.5,
            )
            emg_monitor = create_emg_monitor(
                emg_session,
                dock_parent_name=emg_dock_parent,
                dock_position=emg_dock_position,
                dock_ratio=emg_dock_ratio,
                enabled=not args.no_emg_display,
                visualize_decoder=args.visualize_decoder,
                decoder_hand=args.hand,
                decoder_dock_parent_name=decoder_dock_parent,
                decoder_dock_position=decoder_dock_position,
                decoder_dock_ratio=decoder_dock_ratio,
            )
            env._arat_emg_monitor = emg_monitor
            if emg_monitor is not None:
                emg_monitor.update(force=True)
                og.sim.render()
        goal_status_window, goal_status_labels = _create_goal_status_ui(env)
        goal_conditions = env.task.activity_natural_language_goal_conditions
        # Keep the Kit UI object alive for the full environment lifetime.
        env._arat_goal_status_window = goal_status_window
        reset_positions = th.tensor(reset_joint_positions(args.reset_pose), dtype=th.float32)
        control = {
            "engaged": bool(args.auto_anchor),
            "anchor_after": time.monotonic() if args.auto_anchor else None,
            "pending_command": None,
            "pending_marker_toggles": set(),
            "last_marker_toggle": {"target": float("-inf"), "eef": float("-inf")},
        }

        def start_or_anchor():
            adapter.request_anchor()
            if arm_marker_visualizer is not None:
                arm_marker_visualizer.reset()
            if wrist_recorder is not None:
                wrist_recorder.reset_segment()
            control["engaged"] = True
            control["anchor_after"] = time.monotonic()
            print("Teleoperation engaged; the next fresh hand frame anchors the wrist")

        def reset():
            nonlocal previous_goal_status
            control["engaged"] = False
            control["anchor_after"] = None
            env.reset()
            evaluation_episodes.append([])
            if hand_pose_episodes is not None:
                hand_pose_episodes.append([])
            if action_timing_episodes is not None:
                action_timing_episodes.append([])
            robot.set_joint_positions(reset_positions)
            robot.keep_still()
            _show_robot_end_effectors(robot)
            _validate_loaded_apparatus(env, task)
            _reset_arat_box(env)
            if grasp_supervisor is not None:
                grasp_supervisor.reset()
            _reset_goal_status_labels(goal_status_labels)
            previous_goal_status = None
            if evaluator is not None:
                evaluator.reset()
                print("ARAT scorer reset; the item restarts from zero task time")
            adapter.request_anchor()
            if arm_marker_visualizer is not None:
                arm_marker_visualizer.reset()
            if wrist_recorder is not None:
                wrist_recorder.reset_segment()
            print("Robot reset; press SPACE to engage teleoperation")

        def request_start_or_anchor():
            # Keyboard callbacks can run from inside env.step(). Defer mutations of
            # adapter and episode state until the current action has finished.
            control["pending_command"] = "start_or_anchor"

        def request_reset():
            control["pending_command"] = "reset"

        def request_marker_toggle(marker: str):
            now = time.monotonic()
            if now - control["last_marker_toggle"][marker] >= 0.5:
                control["last_marker_toggle"][marker] = now
                control["pending_marker_toggles"].add(marker)

        def mark_wrist_flip():
            if wrist_recorder is not None:
                wrist_recorder.mark_manual_event()
                print("Wrist flip marked; the next simulator step will be tagged")

        if not gm.HEADLESS:
            KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.SPACE, request_start_or_anchor)
            KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.R, request_reset)
            if arm_marker_visualizer is not None:
                KeyboardEventHandler.add_keyboard_callback(
                    lazy.carb.input.KeyboardInput.T, lambda: request_marker_toggle("target")
                )
                KeyboardEventHandler.add_keyboard_callback(
                    lazy.carb.input.KeyboardInput.E, lambda: request_marker_toggle("eef")
                )
            if wrist_recorder is not None:
                KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.F, mark_wrist_flip)
        diagnostic_key = "; press F when a flip is visible" if wrist_recorder is not None else ""
        marker_keys = "; press T/E to toggle target/EEF frames" if arm_marker_visualizer is not None else ""
        print(f"Press SPACE to start or re-anchor; press R to reset{marker_keys}{diagnostic_key}; Ctrl+C exits.")
        if args.auto_anchor:
            print("Auto-anchor enabled; the first fresh frame anchors the wrist.")

        steps = 0
        logged_goal_termination = False
        previous_goal_status = None
        holding_stale_frame = False
        stale_hold_started = None
        while (step_limit <= 0 or steps < step_limit) and not (shutdown is not None and shutdown.requested):
            pending_command = control["pending_command"]
            control["pending_command"] = None
            if pending_command == "reset":
                reset()
            elif pending_command == "start_or_anchor":
                start_or_anchor()
            if arm_marker_visualizer is not None:
                pending_marker_toggles = control["pending_marker_toggles"]
                if "target" in pending_marker_toggles:
                    visible = arm_marker_visualizer.toggle_target()
                    print(f"Commanded target frame {'shown' if visible else 'hidden'}")
                if "eef" in pending_marker_toggles:
                    visible = arm_marker_visualizer.toggle_eef()
                    print(f"Measured EEF frame {'shown' if visible else 'hidden'}")
                pending_marker_toggles.clear()
            worker.check_health()
            if emg_session is not None:
                emg_session.check_health()
            if not control["engaged"]:
                robot.set_joint_positions(reset_positions)
                robot.keep_still()
                og.sim.step()
                if arm_marker_visualizer is not None and arm_marker_visualizer.update_due():
                    from omnigibson.utils import transform_utils as T

                    base_position, base_quaternion = robot.get_position_orientation()
                    eef_position, eef_quaternion = robot.eef_links[adapter.arm_name].get_position_orientation()
                    eef_relative_position, eef_relative_quaternion = T.relative_pose_transform(
                        eef_position,
                        eef_quaternion,
                        base_position,
                        base_quaternion,
                    )
                    # While disengaged, holding the measured pose is the effective arm command.
                    arm_marker_visualizer.update(
                        target_position_robot=eef_relative_position.cpu().numpy(),
                        target_quaternion_robot_xyzw=eef_relative_quaternion.cpu().numpy(),
                        actual_position_world=eef_position.cpu().numpy(),
                        actual_quaternion_world_xyzw=eef_quaternion.cpu().numpy(),
                        base_position_world=base_position.cpu().numpy(),
                        base_quaternion_world_xyzw=base_quaternion.cpu().numpy(),
                    )
                if emg_monitor is not None:
                    emg_monitor.update()
                steps += 1
                continue
            snapshot = worker.snapshot()
            anchor_after = control["anchor_after"]
            if snapshot is not None:
                frame_age = time.monotonic() - snapshot.frame.receipt_timestamp
                if frame_age > args.maximum_frame_age:
                    if args.stale_frame_policy == "error":
                        # Preserve the worker's standard typed stale-frame exception and message.
                        worker.snapshot(maximum_age=args.maximum_frame_age)
                    if not holding_stale_frame:
                        LOGGER.warning(
                            "HTS frame is stale (%.3fs > %.3fs); holding the robot until tracking resumes",
                            frame_age,
                            args.maximum_frame_age,
                        )
                        holding_stale_frame = True
                        stale_hold_started = time.monotonic()
                    if anchor_after is not None and time.monotonic() - anchor_after > args.initial_frame_timeout:
                        raise RuntimeError(
                            f"No fresh {args.hand}-hand frame arrived within "
                            f"{args.initial_frame_timeout:.1f}s after anchoring"
                        )
                    robot.keep_still()
                    og.sim.step()
                    if emg_monitor is not None:
                        emg_monitor.update()
                    steps += 1
                    continue
                if holding_stale_frame:
                    assert stale_hold_started is not None
                    print(f"Fresh HTS tracking resumed after holding for {time.monotonic() - stale_hold_started:.3f}s")
                    holding_stale_frame = False
                    stale_hold_started = None
            if snapshot is None or (anchor_after is not None and snapshot.frame.receipt_timestamp <= anchor_after):
                if anchor_after is not None and time.monotonic() - anchor_after > args.initial_frame_timeout:
                    raise RuntimeError(
                        f"No fresh {args.hand}-hand frame arrived within {args.initial_frame_timeout:.1f}s after anchoring"
                    )
                robot.keep_still()
                og.sim.step()
                if emg_monitor is not None:
                    emg_monitor.update()
                steps += 1
                continue
            control["anchor_after"] = None
            frozen_fingers = grasp_supervisor.frozen_fingers if grasp_supervisor is not None else None
            source_diagnostics = None
            if wrist_recorder is not None or hand_pose_episodes is not None:
                diagnostics_for_frame = getattr(worker.source, "diagnostics_for_frame", None)
                if diagnostics_for_frame is not None:
                    source_diagnostics = diagnostics_for_frame(snapshot.frame)
            action = adapter.action(snapshot, frozen_fingers=frozen_fingers)
            action_diagnostics = None
            if wrist_recorder is not None or arm_marker_visualizer is not None:
                # Read this before env.step(): Kit keyboard callbacks are dispatched
                # while stepping and a re-anchor intentionally clears adapter state.
                action_diagnostics = adapter.last_wrist_diagnostics
                if action_diagnostics is None:
                    raise RuntimeError("Sharpa adapter did not expose wrist diagnostics after building an action")
            sim_time_before_s = float(og.sim.current_time)
            action_apply_monotonic_ns = time.monotonic_ns()
            _, _, terminated, truncated, info = env.step(action)
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
                hand_pose_episodes[-1].append(HumanHandPoseSample.capture(snapshot.frame, source_diagnostics))
            marker_update_due = arm_marker_visualizer is not None and arm_marker_visualizer.update_due()
            if wrist_recorder is not None or marker_update_due:
                from omnigibson.utils import transform_utils as T

                base_position, base_quaternion = robot.get_position_orientation()
                eef_position, eef_quaternion = robot.eef_links[adapter.arm_name].get_position_orientation()
            if wrist_recorder is not None:
                arm_joint_positions = robot.get_joint_positions()
                eef_relative_position, eef_relative_quaternion = T.relative_pose_transform(
                    eef_position, eef_quaternion, base_position, base_quaternion
                )
                wrist_recorder.observe(
                    step=steps,
                    snapshot=snapshot,
                    source_diagnostics=source_diagnostics,
                    action_diagnostics=action_diagnostics,
                    eef_position_after=eef_relative_position.cpu().numpy(),
                    eef_quaternion_after_xyzw=eef_relative_quaternion.cpu().numpy(),
                    arm_joint_positions=arm_joint_positions[diagnostic_arm_indices].cpu().numpy(),
                    arm_joint_lower_limits=diagnostic_arm_lower_limits,
                    arm_joint_upper_limits=diagnostic_arm_upper_limits,
                )
            if marker_update_due:
                filtered_quaternion = T.axisangle2quat(
                    th.as_tensor(action_diagnostics.filtered_axis_angle, dtype=th.float32)
                )
                arm_marker_visualizer.update(
                    target_position_robot=action_diagnostics.filtered_position,
                    target_quaternion_robot_xyzw=filtered_quaternion.cpu().numpy(),
                    actual_position_world=eef_position.cpu().numpy(),
                    actual_quaternion_world_xyzw=eef_quaternion.cpu().numpy(),
                    base_position_world=base_position.cpu().numpy(),
                    base_quaternion_world_xyzw=base_quaternion.cpu().numpy(),
                )
            if grasp_supervisor is not None:
                grasp_supervisor.step(adapter.last_live_fingers, adapter.measured_fingers)
            goal_status = info["done"]["goal_status"]
            _update_goal_status_labels(goal_status_labels, goal_status)
            if goal_status != previous_goal_status:
                print(
                    f"BDDL goals satisfied: {len(goal_status['satisfied'])}/"
                    f"{len(env.task.activity_natural_language_goal_conditions)}"
                )
                previous_goal_status = {
                    "satisfied": list(goal_status["satisfied"]),
                    "unsatisfied": list(goal_status["unsatisfied"]),
                }
            arat_trace = None
            new_events = ()
            if evaluator is not None:
                arat_snapshot = evaluator.step(engaged=True)
                if arat_snapshot is not None:
                    arat_trace = evaluator.evaluation_trace(arat_snapshot)
                new_events = evaluator.consume_new_events()
            evaluation_episodes[-1].append(build_step_evaluation(goal_conditions, goal_status, arat_trace))
            if emg_monitor is not None:
                emg_monitor.update()
            for event in new_events:
                detail = f" {event.detail}" if event.detail else ""
                print(f"[ARAT t={event.t:5.1f}s] {event.name}{detail}")
            if terminated and not logged_goal_termination:
                if placeholder_goals:
                    LOGGER.warning(
                        "%s uses the intentionally always-successful placeholder goal; termination is ignored",
                        task.activity,
                    )
                else:
                    # The scorer confirms completion (release + settle) before the item ends
                    print(f"BDDL goal for {task.activity} satisfied; awaiting scorer confirmation")
                logged_goal_termination = True
            if truncated:
                raise RuntimeError(f"{task.activity} reached its BehaviorTask timeout")
            steps += 1
            if evaluator is not None:
                if evaluator.finished:
                    break
            if wrist_recorder is not None and wrist_recorder.should_stop:
                print(
                    "Wrist-flip post-event window captured; stopping automatically. "
                    "Use --continue-after-wrist-flip to keep running."
                )
                break

        if evaluator is None:
            return None
        result = evaluator.finalize()
        print(format_item_summary(result))
        if results_dir is not None:
            result_path = write_item_result(result, results_dir)
            print(f"Wrote {result_path}")
        return result
    finally:
        try:
            if emg_monitor is not None:
                emg_monitor.close()
            if arm_marker_visualizer is not None:
                arm_marker_visualizer.close()
            if wrist_recorder is not None:
                wrist_recorder.close()
                print(f"Wrote wrist-flip report to {wrist_recorder.summary_path}")
        finally:
            _finalize_recording(
                recording_env,
                staging_path,
                recording_path,
                evaluation_episodes,
                hand_pose_episodes,
                action_timing_episodes,
                emg_session,
            )


def main(argv: list[str] | None = None) -> None:
    catalog = AratTaskCatalog()
    args = _parser(catalog).parse_args(argv)
    if args.display == "decoder":
        args.visualize_decoder = True
    if args.list_tasks:
        _print_catalog(catalog)
        return
    if args.task is None and args.subscale is None:
        raise SystemExit("Specify --task or --subscale (or use --list-tasks)")
    if not args.view_only and args.hand_model != "sharpa":
        raise SystemExit(
            f"Landmark retargeting supports {args.hand_model}, but initial OmniGibson execution supports only Sharpa"
        )
    if not args.view_only and args.hand != "right":
        raise SystemExit("Initial OmniGibson execution supports only the right-hand Sharpa robot")
    if (
        args.steps < 0
        or args.steps_per_task <= 0
        or args.maximum_frame_age <= 0
        or args.initial_frame_timeout <= 0
        or not math.isfinite(args.tracking_yaw_deg)
        or not math.isfinite(args.position_sensitivity)
        or args.position_sensitivity <= 0
    ):
        raise SystemExit(
            "Step limits, frame timeouts, tracking yaw, or position sensitivity are invalid "
            "(single-task --steps may be 0)"
        )

    tasks = catalog.resolve(args.task, args.subscale)
    if args.camera_rig is not None:
        tasks = tuple(replace(task, camera_rig=args.camera_rig) for task in tasks)
    if args.assisted_grasp and args.view_only:
        raise SystemExit("--assisted-grasp is only supported during teleoperation, not with --view-only")
    if args.visualize_arm_markers and args.view_only:
        raise SystemExit("--visualize-arm-markers is only supported during teleoperation, not with --view-only")
    if args.recording_path is not None and args.view_only:
        raise SystemExit("--recording-path is only supported during teleoperation, not with --view-only")
    if args.record_hand_poses and args.view_only:
        raise SystemExit("--record-hand-poses is only supported during teleoperation, not with --view-only")
    if args.emg and args.view_only:
        raise SystemExit("--emg is only supported during teleoperation, not with --view-only")
    if args.emg and len(tasks) != 1:
        raise SystemExit("--emg requires a single --task so one wristband stream maps to one recording")
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
    ):
        raise SystemExit("EMG device, SDK, display, and decoder options require --emg")
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
    if args.recording_path is not None and len(tasks) != 1:
        raise SystemExit("--recording-path requires a single --task; record subscale activities to separate files")
    if args.wrist_flip_output is not None and len(tasks) != 1:
        raise SystemExit("--wrist-flip-output requires a single --task")
    if args.wrist_flip_output is not None and args.source != "hts":
        raise SystemExit("Raw wrist-flip diagnostics currently require --source hts")
    if (
        not 0.0 < args.wrist_flip_threshold_deg <= 180.0
        or args.wrist_pair_skew_threshold_ms < 0.0
        or args.wrist_flip_post_event_seconds < 0.0
    ):
        raise SystemExit("Wrist diagnostic thresholds are outside their valid ranges")
    data_root = Path(__file__).resolve().parents[4] / "datasets"
    configured_data_root = os.environ.get("OMNIGIBSON_DATA_PATH")
    if configured_data_root is not None and Path(configured_data_root).expanduser().resolve() != data_root.resolve():
        raise SystemExit(
            "OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat; unset it or set it to "
            f"{data_root}"
        )
    os.environ["OMNIGIBSON_DATA_PATH"] = str(data_root)
    validate_runtime_assets(tasks)

    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    gm.USE_PBR_MATERIALS = True
    layout_name = "view_only" if args.view_only else "teleop"
    gm.GUI_VIEWPORT_ONLY = all(
        load_camera_rig(task.camera_rig)
        .layout(layout_name)
        .get("workspace", {})
        .get("viewport_only", False)
        for task in tasks
    )
    # View-only retains OmniGibson's viewer-camera viewport as the main window;
    # its active camera is selected by the YAML layout. Normal teleoperation
    # reuses Kit's main viewport without creating a viewer-camera render product.
    gm.RENDER_VIEWER_CAMERA = args.view_only
    logging.basicConfig(level=logging.INFO)

    enable_score = not args.no_score and not args.view_only and not catalog.placeholder_goals
    if not args.view_only and not args.no_score and catalog.placeholder_goals:
        LOGGER.warning("ARAT scoring disabled: the task catalog still declares placeholder goals")
    results_dir = make_results_dir(Path(args.results_dir)) if enable_score else None
    if results_dir is not None:
        print(f"ARAT results directory: {results_dir}")
    session = None
    if enable_score and args.subscale is not None:
        session = AratSessionScorer({args.subscale.lower(): tuple(task.activity for task in tasks)})

    worker = None
    emg_session = None
    shutdown = GracefulShutdown()
    try:
        if args.emg:
            recording_path = (
                Path(args.recording_path).expanduser() if args.recording_path else default_recording_path(tasks[0])
            )
            emg_session = EmgSession(
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
                decoder_hand=args.hand,
                decoder_inference_hz=args.emg2pose_inference_hz or 5.0,
            )
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
        if not args.view_only:
            source = _create_source(args)
            retargeter = LandmarkRetargeter.from_hand_model(args.hand_model, hand_side=args.hand)
            worker = TrackingRetargetingWorker(source, retargeter, Handedness(args.hand))
            worker.start()
        ran_any = False
        for task in tasks:
            if session is not None and not session.should_administer(task.activity):
                print(f"Skipping {task.activity}: ARAT protocol short-circuit")
                continue
            if ran_any:
                og.clear()
            ran_any = True
            limit = args.steps if len(tasks) == 1 else args.steps_per_task
            if args.view_only:
                _run_viewer(task, limit, shutdown=shutdown)
            else:
                result = _run_task(
                    task,
                    worker,
                    args,
                    limit,
                    enable_score=enable_score,
                    placeholder_goals=catalog.placeholder_goals,
                    results_dir=results_dir,
                    shutdown=shutdown,
                    emg_session=emg_session,
                )
                if session is not None and result is not None:
                    session.record(task.activity, result.score)
            if shutdown.requested:
                break
        if session is not None:
            print(format_session_summary(session))
            session_path = write_session_result(session, results_dir)
            print(f"Wrote {session_path}")
    except KeyboardInterrupt:
        print("\nStopping ARAT scene viewer" if args.view_only else "\nStopping ARAT teleoperation")
    except Exception:
        print("\nARAT launcher failed before shutdown:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if worker is not None:
            worker.close()
        if emg_session is not None:
            emg_session.close()
        og.shutdown()

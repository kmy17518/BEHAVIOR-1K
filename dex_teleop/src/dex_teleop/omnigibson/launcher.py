"""One-process ARAT BehaviorTask launcher for landmark hand teleoperation."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from dex_teleop.arat import AratTask, AratTaskCatalog
from dex_teleop.arat.eval import AratSessionScorer, load_rubrics
from dex_teleop.arat.eval.report import (
    format_item_summary,
    format_session_summary,
    make_results_dir,
    write_item_result,
    write_session_result,
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
from dex_teleop.omnigibson.evaluation_trace import build_step_evaluation, write_evaluation_episodes
from dex_teleop.retargeting import LandmarkRetargeter
from dex_teleop.runtime import TrackingRetargetingWorker
from dex_teleop.tracking import HTSSource
from dex_teleop.tracking.ovxr import create_ovxr_source
from dex_teleop.types import Handedness


LOGGER = logging.getLogger(__name__)
# Matches the sharpa_right default_joint_pos in the robot definition (franka.yaml).
RESET_JOINT_POSITIONS = [0.0, -1.30, 0.0, -2.87, 0.0, 2.00, 0.75] + [0.0] * 22
ROBOT_POSITION = [-0.5685013461, -0.1084822643, 0.0156103678]
TABLE_LAYOUT_TRANSLATION = [-0.0885013461, -0.1084822643, 0.1766103678]
CAMERA_IMAGE_HEIGHT = 480
CAMERA_IMAGE_WIDTH = 640
TELEOP_CAMERA_COLUMN_RATIO = 0.25
TELEOP_CAMERA_STACK_RATIO = 0.5
VIEW_ONLY_CAMERA_DOCK_RATIO = 0.25
# Symmetric 30-degree outward orbits about the shared shoulder-camera center-ray
# anchor [-0.1685013461, -0.1084822643, 1.0566103678].
LEFT_SHOULDER_CAMERA_POSITION = [-0.7387919357, 0.9421831684, 1.8487674722]
LEFT_SHOULDER_CAMERA_ORIENTATION = [-0.1164232160, 0.4585405560, 0.8539201738, -0.2168098939]
RIGHT_SHOULDER_CAMERA_POSITION = [-0.7387919357, -1.1591476970, 1.8487674722]
RIGHT_SHOULDER_CAMERA_ORIENTATION = [0.4585405560, -0.1164232160, -0.2168098939, 0.8539201738]
WRIST_CAMERA_LINK = "panda_link7"
ROBOT_COMPOSED_MODEL = f"{ROBOT_MODEL}_{ROBOT_END_EFFECTOR}"
# This transform puts the camera on the palmar side of the hand, under the heel of the
# palm (7 cm along the palm normal, 6 cm back toward the wrist at the start pose). Its
# optical axis looks out of the palm with a slight pitch toward the fingertips, so the
# approach target stays centered and the fingers enter the frame as they close. The
# parent is panda_link7, so it follows the Franka wrist rigidly.
WRIST_CAMERA_POSITION = [-0.01646253, 0.02611968, 0.17699993]
WRIST_CAMERA_ORIENTATION = [0.90889782, 0.31708947, -0.08921809, -0.25573269]
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


def _camera_sensor_config(
    name: str,
    relative_prim_path: str,
    position: list,
    orientation: list,
    pose_frame: str,
) -> dict:
    # Sharing the main viewport prevents VisionSensor from automatically
    # creating three extra windows. The intended windows are docked below.
    viewport_name = None if os.environ.get("OMNIGIBSON_HEADLESS") == "1" else "Viewport"
    return {
        "sensor_type": "VisionSensor",
        "name": name,
        "relative_prim_path": relative_prim_path,
        "modalities": [],
        "sensor_kwargs": {
            "viewport_name": viewport_name,
            "image_height": CAMERA_IMAGE_HEIGHT,
            "image_width": CAMERA_IMAGE_WIDTH,
            "horizontal_aperture": 20.955,
            "focal_length": 17.0,
        },
        "position": position,
        "orientation": orientation,
        "pose_frame": pose_frame,
        "include_in_obs": False,
    }


def _external_camera_configs(*, include_wrist: bool) -> list[dict]:
    cameras = [
        _camera_sensor_config(
            name="arat_left_shoulder_camera",
            relative_prim_path="/arat_left_shoulder_camera",
            position=LEFT_SHOULDER_CAMERA_POSITION,
            orientation=LEFT_SHOULDER_CAMERA_ORIENTATION,
            pose_frame="scene",
        ),
        _camera_sensor_config(
            name="arat_right_shoulder_camera",
            relative_prim_path="/arat_right_shoulder_camera",
            position=RIGHT_SHOULDER_CAMERA_POSITION,
            orientation=RIGHT_SHOULDER_CAMERA_ORIENTATION,
            pose_frame="scene",
        ),
    ]
    if include_wrist:
        cameras.append(
            _camera_sensor_config(
                name="arat_wrist_camera",
                relative_prim_path=(
                    f"/controllable__{ROBOT_COMPOSED_MODEL}__{ROBOT_NAME}/"
                    f"{WRIST_CAMERA_LINK}/arat_wrist_camera"
                ),
                position=WRIST_CAMERA_POSITION,
                orientation=WRIST_CAMERA_ORIENTATION,
                pose_frame="parent",
            )
        )
    return cameras


def _parser(catalog: AratTaskCatalog) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", choices=tuple(catalog.tasks))
    selection.add_argument("--subscale", choices=tuple(catalog.subscales))
    parser.add_argument("--list-tasks", action="store_true")
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
        "--view-only",
        action="store_true",
        help="Load the selected saved scene without a robot or hand-tracking source",
    )
    parser.add_argument(
        "--assisted-grasp",
        action="store_true",
        help=(
            "Weld an object to the hand while the teleoperated fingers hold it in opposition "
            "(OmniGibson assisted grasping, driven by the fingers actually touching the object "
            "instead of the robot definition's fixed finger pairs)"
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
    parser.add_argument("--initial-frame-timeout", type=float, default=10.0)
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


def build_environment_config(task: AratTask, *, view_only: bool = False, assisted_grasp: bool = False) -> dict:
    """Build an environment from the selected version-1 saved scene."""

    config = {
        # OmniGibson's default rendering frequency is 30 Hz, so the action
        # period must be a multiple of 1 / 30 seconds.
        "env": {
            "action_frequency": 30.0,
            "automatic_reset": False,
            "external_sensors": _external_camera_configs(include_wrist=not view_only),
        },
        "scene": {
            "type": "Scene",
            "scene_file": get_task_scene_data(task, include_task_metadata=not view_only),
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
                "position": ROBOT_POSITION,
                "orientation": [0.0, 0.0, 0.0, 1.0],
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
                "reset_joint_pos": RESET_JOINT_POSITIONS,
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
            "use_presampled_robot_pose": False,
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


def _configure_overview_camera() -> None:
    import torch as th
    import omnigibson as og
    import omnigibson.utils.transform_utils as T

    camera_position = th.tensor([-1.0455796193, 0.4291464976, 1.5382929949], dtype=th.float32)
    camera_target = th.tensor(
        [
            -0.08 + TABLE_LAYOUT_TRANSLATION[0],
            TABLE_LAYOUT_TRANSLATION[1],
            0.68 + TABLE_LAYOUT_TRANSLATION[2],
        ],
        dtype=th.float32,
    )
    forward = th.nn.functional.normalize(camera_target - camera_position, dim=0)
    right = th.nn.functional.normalize(th.linalg.cross(forward, th.tensor([0.0, 0.0, 1.0])), dim=0)
    up = th.linalg.cross(right, forward)
    camera_orientation = T.mat2quat(th.stack((right, up, -forward), dim=1))
    og.sim.viewer_camera.active_camera_path = og.sim.viewer_camera.prim_path
    og.sim.viewer_camera.set_position_orientation(camera_position, camera_orientation)
    og.sim.enable_viewer_camera_teleoperation()


def _create_and_dock_camera_viewport(
    name: str,
    camera_path: str,
    dock_position,
    dock_ratio: float,
    dock_parent_name: str = "DockSpace",
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
    viewport.viewport_api.set_texture_resolution((CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT))
    og.sim.render()
    return viewport


def _configure_camera_layout(env, robot=None) -> None:
    """Create left-shoulder, right-shoulder, and wrist camera views."""

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.macros import gm

    sensors = env.external_sensors or {}
    expected = {"arat_left_shoulder_camera", "arat_right_shoulder_camera"}
    if robot is not None:
        expected.add("arat_wrist_camera")
    missing = expected.difference(sensors)
    if missing:
        raise RuntimeError(f"ARAT camera layout is missing sensors: {sorted(missing)}")

    if robot is not None:
        wrist_camera_path = sensors["arat_wrist_camera"].prim_path
        expected_wrist_parent = robot.links[WRIST_CAMERA_LINK].prim_path
        if not wrist_camera_path.startswith(f"{expected_wrist_parent}/"):
            raise RuntimeError(
                f"ARAT wrist camera is not parented to {WRIST_CAMERA_LINK}: {wrist_camera_path}"
            )

    if gm.HEADLESS:
        return

    if robot is None:
        # View-only intentionally has no robot, so retain the movable overview
        # in the main viewport and show both shoulder views beside it.
        _configure_overview_camera()
        main_viewport = og.sim.viewer_camera._viewport
        main_view = "movable overview"

        left_viewport = _create_and_dock_camera_viewport(
            name="ARAT Left Shoulder",
            camera_path=sensors["arat_left_shoulder_camera"].prim_path,
            dock_position=lazy.omni.ui.DockPosition.LEFT,
            dock_ratio=VIEW_ONLY_CAMERA_DOCK_RATIO,
        )
        right_viewport = _create_and_dock_camera_viewport(
            name="ARAT Right Shoulder",
            camera_path=sensors["arat_right_shoulder_camera"].prim_path,
            dock_position=lazy.omni.ui.DockPosition.RIGHT,
            dock_ratio=VIEW_ONLY_CAMERA_DOCK_RATIO,
        )
        camera_viewports = {
            "left_shoulder": left_viewport,
            "main": main_viewport,
            "right_shoulder": right_viewport,
        }
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
        main_viewport.viewport_api.set_active_camera(sensors["arat_left_shoulder_camera"].prim_path)
        main_viewport.viewport_api.set_texture_resolution((CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT))
        with og.sim.editing_usd():
            main_viewport.visible = True

        right_viewport = _create_and_dock_camera_viewport(
            name="ARAT Right Shoulder",
            camera_path=sensors["arat_right_shoulder_camera"].prim_path,
            dock_position=lazy.omni.ui.DockPosition.LEFT,
            dock_ratio=TELEOP_CAMERA_COLUMN_RATIO,
        )
        wrist_viewport = _create_and_dock_camera_viewport(
            name="ARAT Franka Wrist",
            camera_path=wrist_camera_path,
            dock_position=lazy.omni.ui.DockPosition.BOTTOM,
            dock_ratio=TELEOP_CAMERA_STACK_RATIO,
            dock_parent_name=right_viewport.name,
        )
        camera_viewports = {
            "main": main_viewport,
            "left_shoulder": main_viewport,
            "right_shoulder": right_viewport,
            "wrist": wrist_viewport,
        }

    env._arat_camera_viewports = camera_viewports
    for _ in range(3):
        og.sim.render()
    if robot is None:
        print(f"Camera layout ready: left shoulder | {main_view} | right shoulder")
    else:
        print("Camera layout ready: left shoulder main | right shoulder + Franka wrist in left column")


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
) -> None:
    """Close the HDF5 file, then atomically replace the requested output."""

    recording_env.save_data()
    if evaluation_episodes is not None:
        write_evaluation_episodes(staging_path, evaluation_episodes)
    os.replace(staging_path, output_path)
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
    _configure_camera_layout(env)
    print("Scene viewer ready. The center overview remains camera-teleoperable; Ctrl+C exits.")
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
):
    import torch as th
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.envs import HDF5CollectionWrapper
    from omnigibson.macros import gm
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    from dex_teleop.omnigibson.sharpa_adapter import SharpaActionAdapter, SharpaAdapterConfig

    print(f"\nLoading {task.activity}: {task.label} (layout: {task.layout})")
    env = og.Environment(configs=build_environment_config(task, assisted_grasp=args.assisted_grasp))
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

    try:
        env.reset()
        evaluation_episodes.append([])
        _validate_loaded_apparatus(env, task)
        _reset_arat_box(env)
        _hide_skybox_from_camera()
        robot = env.robots[0]
        _show_robot_end_effectors(robot)
        adapter = SharpaActionAdapter(robot, config=SharpaAdapterConfig(control_hz=30.0))
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
        _configure_camera_layout(env, robot)
        goal_status_window, goal_status_labels = _create_goal_status_ui(env)
        goal_conditions = env.task.activity_natural_language_goal_conditions
        # Keep the Kit UI object alive for the full environment lifetime.
        env._arat_goal_status_window = goal_status_window
        reset_positions = th.tensor(RESET_JOINT_POSITIONS, dtype=th.float32)
        control = {
            "engaged": bool(args.auto_anchor),
            "anchor_after": time.monotonic() if args.auto_anchor else None,
        }

        def start_or_anchor():
            adapter.request_anchor()
            control["engaged"] = True
            control["anchor_after"] = time.monotonic()
            print("Teleoperation engaged; the next fresh hand frame anchors the wrist")

        def reset():
            nonlocal previous_goal_status
            control["engaged"] = False
            control["anchor_after"] = None
            env.reset()
            evaluation_episodes.append([])
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
            print("Robot reset; press SPACE to engage teleoperation")

        if not gm.HEADLESS:
            KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.SPACE, start_or_anchor)
            KeyboardEventHandler.add_keyboard_callback(lazy.carb.input.KeyboardInput.R, reset)
        print("Press SPACE to start or re-anchor; press R to reset; Ctrl+C exits.")
        if args.auto_anchor:
            print("Auto-anchor enabled; the first fresh frame anchors the wrist.")

        steps = 0
        logged_goal_termination = False
        previous_goal_status = None
        while (step_limit <= 0 or steps < step_limit) and not (shutdown is not None and shutdown.requested):
            worker.check_health()
            if not control["engaged"]:
                robot.set_joint_positions(reset_positions)
                robot.keep_still()
                og.sim.step()
                steps += 1
                continue
            snapshot = worker.snapshot(maximum_age=args.maximum_frame_age)
            anchor_after = control["anchor_after"]
            if snapshot is None or (anchor_after is not None and snapshot.frame.timestamp <= anchor_after):
                if anchor_after is not None and time.monotonic() - anchor_after > args.initial_frame_timeout:
                    raise RuntimeError(
                        f"No fresh {args.hand}-hand frame arrived within {args.initial_frame_timeout:.1f}s after anchoring"
                    )
                robot.keep_still()
                og.sim.step()
                steps += 1
                continue
            control["anchor_after"] = None
            frozen_fingers = grasp_supervisor.frozen_fingers if grasp_supervisor is not None else None
            _, _, terminated, truncated, info = env.step(adapter.action(snapshot, frozen_fingers=frozen_fingers))
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

        if evaluator is None:
            return None
        result = evaluator.finalize()
        print(format_item_summary(result))
        if results_dir is not None:
            result_path = write_item_result(result, results_dir)
            print(f"Wrote {result_path}")
        return result
    finally:
        _finalize_recording(recording_env, staging_path, recording_path, evaluation_episodes)


def main(argv: list[str] | None = None) -> None:
    catalog = AratTaskCatalog()
    args = _parser(catalog).parse_args(argv)
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
    ):
        raise SystemExit("Step limits and maximum frame age must be positive (single-task --steps may be 0)")

    tasks = catalog.resolve(args.task, args.subscale)
    if args.assisted_grasp and args.view_only:
        raise SystemExit("--assisted-grasp is only supported during teleoperation, not with --view-only")
    if args.recording_path is not None and args.view_only:
        raise SystemExit("--recording-path is only supported during teleoperation, not with --view-only")
    if args.recording_path is not None and len(tasks) != 1:
        raise SystemExit("--recording-path requires a single --task; record subscale activities to separate files")
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
    # View-only needs OmniGibson's movable viewer camera. Normal teleoperation
    # uses the wrist camera in Kit's main viewport directly, avoiding a fourth,
    # unused viewer-camera render product and hiding the default view at startup.
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
    shutdown = GracefulShutdown()
    try:
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
        og.shutdown()

"""Hand-bench scene: a fixed right Sharpa hand above a table, defined in YAML."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from dex_teleop.arat.camera_rig import build_camera_sensor_configs, load_camera_rig
from dex_teleop.arat.scene import ROBOT_DATASET_NAME, ROBOT_END_EFFECTOR, ROBOT_MODEL
from dex_teleop.hands import SHARPA_ACTION_JOINTS


HAND_BENCH_PATH = Path(__file__).with_name("hand_bench.yaml")
SUPPORTED_SCHEMA_VERSION = 1
DEFAULT_CAMERA_RIG = "hand_bench"
CAMERA_LAYOUT = "teleop"
ROBOT_NAME = "hand_bench_sharpa_right"
ROBOT_COMPOSED_MODEL = f"{ROBOT_MODEL}_{ROBOT_END_EFFECTOR}"
ROBOT_PRIM_PATH = f"/controllable__{ROBOT_COMPOSED_MODEL}__{ROBOT_NAME}"
TABLE_NAME = "table"
FRANKA_ARM_DOFS = 7

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DATASETS_ROOT = _REPOSITORY_ROOT / "datasets"


def _vector(value, length: int, description: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{description} must contain {length} finite numbers")
    return tuple(float(item) for item in value)


def _positive(value, description: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{description} must be a positive finite number")
    return float(value)


@dataclass(frozen=True)
class ExpectedHandPose:
    """Where the Sharpa palm base must land once the hidden arm holds its configuration."""

    link: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    position_tolerance_m: float
    orientation_tolerance_deg: float


@dataclass(frozen=True)
class HandBenchTable:
    dataset_name: str
    category: str
    model: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    tabletop_height: float
    height_tolerance_m: float


@dataclass(frozen=True)
class HandBenchRobot:
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    arm_joint_positions: tuple[float, ...]
    hidden_prim_prefixes: tuple[str, ...]
    expected_hand_pose: ExpectedHandPose


@dataclass(frozen=True)
class HandBenchScene:
    floor_plane_color: tuple[float, float, float]
    table: HandBenchTable
    lights: tuple[dict, ...]
    robot: HandBenchRobot


def load_hand_bench_scene(path: str | Path = HAND_BENCH_PATH) -> HandBenchScene:
    """Load and validate the packaged hand-bench scene definition."""

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"{path} must use schema_version {SUPPORTED_SCHEMA_VERSION}")
    scene = raw.get("scene")
    if not isinstance(scene, dict):
        raise ValueError(f"{path} must define a scene mapping")

    table_raw = scene.get("table")
    if not isinstance(table_raw, dict):
        raise ValueError("Hand bench scene must define a table")
    for field in ("dataset_name", "category", "model"):
        if not isinstance(table_raw.get(field), str) or not table_raw[field]:
            raise ValueError(f"Hand bench table {field} must be a non-empty string")
    table = HandBenchTable(
        dataset_name=table_raw["dataset_name"],
        category=table_raw["category"],
        model=table_raw["model"],
        position=_vector(table_raw.get("position"), 3, "Table position"),
        orientation_xyzw=_vector(table_raw.get("orientation_xyzw"), 4, "Table orientation_xyzw"),
        tabletop_height=_positive(table_raw.get("tabletop_height"), "Table tabletop_height"),
        height_tolerance_m=_positive(table_raw.get("height_tolerance_m", 0.01), "Table height_tolerance_m"),
    )

    lights_raw = scene.get("lights", [])
    if not isinstance(lights_raw, list):
        raise ValueError("Hand bench lights must be a list")
    lights = []
    light_names = set()
    for light in lights_raw:
        if not isinstance(light, dict) or not isinstance(light.get("name"), str) or not light["name"]:
            raise ValueError("Every hand bench light must be a mapping with a name")
        if light["name"] in light_names:
            raise ValueError(f"Hand bench light name {light['name']!r} is repeated")
        light_names.add(light["name"])
        if not isinstance(light.get("light_type"), str):
            raise ValueError(f"Light {light['name']!r} must define light_type")
        lights.append(
            {
                "name": light["name"],
                "light_type": light["light_type"],
                "radius": _positive(light.get("radius", 1.0), f"Light {light['name']!r} radius"),
                "intensity": _positive(light.get("intensity", 50000.0), f"Light {light['name']!r} intensity"),
                "position": list(_vector(light.get("position"), 3, f"Light {light['name']!r} position")),
            }
        )

    robot_raw = scene.get("robot")
    if not isinstance(robot_raw, dict):
        raise ValueError("Hand bench scene must define a robot")
    prefixes = robot_raw.get("hidden_prim_prefixes", [])
    if not isinstance(prefixes, list) or not all(isinstance(prefix, str) and prefix for prefix in prefixes):
        raise ValueError("Robot hidden_prim_prefixes must be a list of non-empty strings")
    if any("right_hand".startswith(prefix) or prefix.startswith("right_hand") for prefix in prefixes):
        raise ValueError("hidden_prim_prefixes may not hide the Sharpa hand itself")
    expected_raw = robot_raw.get("expected_hand_pose")
    if not isinstance(expected_raw, dict) or not isinstance(expected_raw.get("link"), str):
        raise ValueError("Robot expected_hand_pose must name the palm link")
    expected = ExpectedHandPose(
        link=expected_raw["link"],
        position=_vector(expected_raw.get("position"), 3, "Expected hand position"),
        orientation_xyzw=_vector(expected_raw.get("orientation_xyzw"), 4, "Expected hand orientation_xyzw"),
        position_tolerance_m=_positive(expected_raw.get("position_tolerance_m", 0.01), "Hand position tolerance"),
        orientation_tolerance_deg=_positive(
            expected_raw.get("orientation_tolerance_deg", 2.0), "Hand orientation tolerance"
        ),
    )
    robot = HandBenchRobot(
        position=_vector(robot_raw.get("position"), 3, "Robot position"),
        orientation_xyzw=_vector(robot_raw.get("orientation_xyzw"), 4, "Robot orientation_xyzw"),
        arm_joint_positions=_vector(
            robot_raw.get("arm_joint_positions"), FRANKA_ARM_DOFS, "Robot arm_joint_positions"
        ),
        hidden_prim_prefixes=tuple(prefixes),
        expected_hand_pose=expected,
    )
    return HandBenchScene(
        floor_plane_color=_vector(scene.get("floor_plane_color", [0.5, 0.5, 0.5]), 3, "floor_plane_color"),
        table=table,
        lights=tuple(lights),
        robot=robot,
    )


def reset_joint_positions(scene: HandBenchScene) -> list[float]:
    """The held Franka configuration followed by an open Sharpa hand."""

    return list(scene.robot.arm_joint_positions) + [0.0] * len(SHARPA_ACTION_JOINTS)


def hidden_prim_names(names: Iterable[str], prefixes: Sequence[str]) -> tuple[str, ...]:
    """Return the link or child-prim names whose visuals the bench hides, in the given order."""

    return tuple(name for name in names if any(name.startswith(prefix) for prefix in prefixes))


def hand_pose_error(
    position: Sequence[float],
    quaternion_xyzw: Sequence[float],
    expected: ExpectedHandPose,
) -> tuple[float, float]:
    """Return (metres, degrees) between a measured palm pose and the expected one."""

    measured_position = np.asarray(position, dtype=np.float64).reshape(3)
    measured_quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    measured_quaternion = measured_quaternion / np.linalg.norm(measured_quaternion)
    expected_quaternion = np.asarray(expected.orientation_xyzw, dtype=np.float64)
    expected_quaternion = expected_quaternion / np.linalg.norm(expected_quaternion)
    translation = float(np.linalg.norm(measured_position - np.asarray(expected.position)))
    dot = min(1.0, abs(float(np.dot(measured_quaternion, expected_quaternion))))
    return translation, math.degrees(2.0 * math.acos(dot))


def robot_usd_path() -> Path:
    """Resolve the USD that franka.yaml declares for the sharpa_right end effector."""

    definition_path = _DATASETS_ROOT / ROBOT_DATASET_NAME / "models" / ROBOT_MODEL / f"{ROBOT_MODEL}.yaml"
    try:
        with definition_path.open("r", encoding="utf-8") as stream:
            definition = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing robot definition: {definition_path}") from error
    try:
        usd_path = definition["manipulation"]["end_effectors"][ROBOT_END_EFFECTOR]["usd_path"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{definition_path} does not declare the {ROBOT_END_EFFECTOR} end effector") from error
    return _DATASETS_ROOT / ROBOT_DATASET_NAME / usd_path


def validate_runtime_assets(scene: HandBenchScene) -> None:
    """Fail early when the table or the Franka + Sharpa asset is unavailable."""

    table = scene.table
    required = [
        _DATASETS_ROOT
        / table.dataset_name
        / "objects"
        / table.category
        / table.model
        / "usd"
        / f"{table.model}.usd",
        robot_usd_path(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required hand bench assets:\n" + "\n".join(missing))


def build_hand_bench_config(
    scene: HandBenchScene,
    *,
    camera_rig_name: str = DEFAULT_CAMERA_RIG,
    layout_name: str = CAMERA_LAYOUT,
    viewport_name: str | None = "Viewport",
    sensor_modalities: Sequence[str] | None = None,
) -> dict:
    """Build the OmniGibson environment configuration for the hand bench.

    The Franka arm is held by a ``NullJointController`` at the scene's solved
    configuration so ``env.action_space`` is exactly the 22 Sharpa finger
    joints; the wrist therefore cannot be commanded at all.
    """

    import torch as th

    camera_rig = load_camera_rig(camera_rig_name)
    sensors = build_camera_sensor_configs(
        camera_rig,
        layout_name,
        robot_prim_path=ROBOT_PRIM_PATH,
        viewport_name=viewport_name,
    )
    if sensor_modalities is not None:
        for sensor in sensors:
            sensor["modalities"] = list(sensor_modalities)
    table = scene.table
    robot = scene.robot
    objects = [
        {
            "type": "DatasetObject",
            "name": TABLE_NAME,
            "category": table.category,
            "model": table.model,
            "dataset_name": table.dataset_name,
            "fixed_base": True,
            "position": list(table.position),
            "orientation": list(table.orientation_xyzw),
        }
    ]
    for light in scene.lights:
        objects.append(
            {
                "type": "LightObject",
                "name": light["name"],
                "light_type": light["light_type"],
                "radius": light["radius"],
                "intensity": light["intensity"],
                "position": list(light["position"]),
            }
        )
    return {
        "env": {
            "action_frequency": 30.0,
            "automatic_reset": False,
            "external_sensors": sensors,
        },
        "scene": {
            "type": "Scene",
            "use_floor_plane": True,
            "floor_plane_visible": True,
            "floor_plane_color": list(scene.floor_plane_color),
            # The dome light would over-light the floor that serves as backdrop.
            "use_skybox": False,
        },
        "objects": objects,
        "robots": [
            {
                "model": ROBOT_MODEL,
                "dataset_name": ROBOT_DATASET_NAME,
                "end_effector": ROBOT_END_EFFECTOR,
                "name": ROBOT_NAME,
                "position": list(robot.position),
                "orientation": list(robot.orientation_xyzw),
                "grasping_direction": "upper",
                "grasping_mode": "physical",
                "obs_modalities": [],
                "action_normalize": False,
                "fixed_base": True,
                "self_collisions": False,
                "reset_joint_pos": reset_joint_positions(scene),
                "controller_config": {
                    "arm_0": {
                        "name": "NullJointController",
                        "motor_type": "position",
                        "default_goal": th.tensor(robot.arm_joint_positions, dtype=th.float32),
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
        "task": {"type": "DummyTask"},
    }

"""Hand bench: fixed Sharpa hand scene, camera rig, and finger-only CLI (no simulator required)."""

from pathlib import Path

import numpy as np
import pytest
import torch as th
import yaml

from dex_teleop.arat.camera_rig import camera_rig_names, layout_camera_ids, load_camera_rig
from dex_teleop.hand_bench import (
    CAMERA_LAYOUT,
    DEFAULT_CAMERA_RIG,
    HAND_BENCH_PATH,
    ROBOT_NAME,
    ROBOT_PRIM_PATH,
    TABLE_NAME,
    build_hand_bench_config,
    hand_pose_error,
    hidden_prim_names,
    load_hand_bench_scene,
    reset_joint_positions,
)
from dex_teleop.hand_bench import launcher
from dex_teleop.hands import SHARPA_ACTION_JOINTS
from dex_teleop.tracking import FIXED_WRIST_SOURCE_NAME, FixedWristSource
from dex_teleop.types import Handedness


class _IdleSource:
    def start(self):
        pass

    def check_health(self):
        pass

    def close(self):
        pass

    def read_articulation(self, handedness):
        return None


def test_scene_definition_pins_the_arm_and_keeps_the_hand_visible():
    scene = load_hand_bench_scene()

    assert len(scene.robot.arm_joint_positions) == 7
    assert scene.robot.expected_hand_pose.link == "right_hand_C_MC"
    assert "panda_" in scene.robot.hidden_prim_prefixes
    assert not any(prefix.startswith("right_hand") for prefix in scene.robot.hidden_prim_prefixes)
    assert scene.table.tabletop_height > 0.5
    assert len(scene.lights) >= 1
    # The hand pose is above the table centre, palm toward the operator (-X), fingers up.
    position = np.asarray(scene.robot.expected_hand_pose.position)
    assert position[2] > scene.table.tabletop_height + 0.05
    x, y, z, w = scene.robot.expected_hand_pose.orientation_xyzw
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    palm_normal, thumb_axis, finger_axis = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    assert palm_normal[0] < -0.9
    assert thumb_axis[1] < -0.9
    assert finger_axis[2] > 0.9


def test_reset_pose_is_held_arm_plus_open_hand():
    scene = load_hand_bench_scene()
    reset = reset_joint_positions(scene)

    assert len(reset) == 7 + len(SHARPA_ACTION_JOINTS)
    assert tuple(reset[:7]) == scene.robot.arm_joint_positions
    assert reset[7:] == [0.0] * len(SHARPA_ACTION_JOINTS)


def test_environment_config_holds_the_arm_and_exposes_only_fingers():
    scene = load_hand_bench_scene()
    config = build_hand_bench_config(scene, viewport_name=None)

    robot = config["robots"][0]
    arm = robot["controller_config"]["arm_0"]
    assert robot["name"] == ROBOT_NAME
    assert arm["name"] == "NullJointController"
    assert th.allclose(arm["default_goal"], th.tensor(scene.robot.arm_joint_positions, dtype=th.float32))
    assert robot["controller_config"]["gripper_0"]["mode"] == "independent"
    assert robot["reset_joint_pos"] == reset_joint_positions(scene)
    assert robot["fixed_base"] is True
    assert list(robot["position"]) == list(scene.robot.position)
    assert config["task"] == {"type": "DummyTask"}
    assert config["scene"]["use_skybox"] is False
    assert config["scene"]["floor_plane_color"] == list(scene.floor_plane_color)
    objects = {obj["name"]: obj for obj in config["objects"]}
    assert objects[TABLE_NAME]["type"] == "DatasetObject"
    assert objects[TABLE_NAME]["dataset_name"] == scene.table.dataset_name
    assert objects[TABLE_NAME]["fixed_base"] is True
    for light in scene.lights:
        assert objects[light["name"]]["type"] == "LightObject"
    sensors = {sensor["name"]: sensor for sensor in config["env"]["external_sensors"]}
    assert set(sensors) == {"hand_bench_dorsal_camera", "hand_bench_palmar_camera"}
    assert all(sensor["modalities"] == [] for sensor in sensors.values())
    assert all(sensor["sensor_kwargs"]["viewport_name"] is None for sensor in sensors.values())

    with_rgb = build_hand_bench_config(scene, sensor_modalities=("rgb",))
    assert all(sensor["modalities"] == ["rgb"] for sensor in with_rgb["env"]["external_sensors"])
    assert all(sensor["sensor_kwargs"]["viewport_name"] == "Viewport" for sensor in with_rgb["env"]["external_sensors"])


def test_hand_bench_camera_rig_uses_the_joylo_toggle_key():
    assert DEFAULT_CAMERA_RIG in camera_rig_names()
    rig = load_camera_rig(DEFAULT_CAMERA_RIG)
    layout = rig.layout(CAMERA_LAYOUT)

    assert layout["viewports"]["main"]["camera"] == "dorsal"
    assert layout["toggles"] == [{"key": "B", "viewport": "main", "cameras": ["dorsal", "palmar"]}]
    assert layout_camera_ids(rig, CAMERA_LAYOUT) == ("dorsal", "palmar")
    assert layout["workspace"]["viewport_only"] is True
    for camera_id in ("dorsal", "palmar"):
        camera = rig.camera(camera_id)
        assert camera["parent"] == {"frame": "scene"}
        assert rig.calibration(camera_id)["image_width"] == 1080
    # Dorsal looks from behind the back of the hand (+X), palmar from the operator side (-X); both above.
    assert rig.camera("dorsal")["pose"]["position"][0] > 0 > rig.camera("palmar")["pose"]["position"][0]
    assert all(rig.camera(camera_id)["pose"]["position"][2] > 1.2 for camera_id in ("dorsal", "palmar"))
    assert ROBOT_PRIM_PATH.endswith(f"__{ROBOT_NAME}")


def test_hidden_prim_names_and_pose_error():
    names = ("panda_base", "panda_link7", "right_hand_C_MC", "right_thumb_PP", "right_wrist_camera_mount")
    assert hidden_prim_names(names, ("panda_", "right_wrist_camera_")) == (
        "panda_base",
        "panda_link7",
        "right_wrist_camera_mount",
    )
    expected = load_hand_bench_scene().robot.expected_hand_pose
    same = hand_pose_error(expected.position, expected.orientation_xyzw, expected)
    flipped_sign = hand_pose_error(expected.position, -np.asarray(expected.orientation_xyzw), expected)
    moved = hand_pose_error(np.asarray(expected.position) + [0.0, 0.0, 0.02], expected.orientation_xyzw, expected)
    assert same == pytest.approx((0.0, 0.0), abs=1e-6)
    assert flipped_sign == pytest.approx((0.0, 0.0), abs=1e-6)
    assert moved[0] == pytest.approx(0.02)


def test_scene_yaml_rejects_hiding_the_hand(tmp_path: Path):
    raw = yaml.safe_load(HAND_BENCH_PATH.read_text(encoding="utf-8"))
    raw["scene"]["robot"]["hidden_prim_prefixes"] = ["panda_", "right_hand_C"]
    path = tmp_path / "hand_bench.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="may not hide the Sharpa hand"):
        load_hand_bench_scene(path)


@pytest.mark.parametrize(
    "argv",
    [
        ["--wrist-source", "vive"],
        ["--wrist-source=quest"],
        ["--source", "hts"],
        ["--vive-calibration", "vive.json"],
        ["--record-wrist-source", "quest"],
        ["--manus-expected-tracker-id", "LHR-1"],
    ],
)
def test_cli_refuses_wrist_tracking_options(argv):
    with pytest.raises(SystemExit, match="disables wrist tracking"):
        launcher.reject_wrist_options(argv)
    with pytest.raises(SystemExit):
        launcher._parser().parse_args(argv)


def test_cli_accepts_only_finger_sources():
    parser = launcher._parser()
    for source in ("quest", "hts", "manus"):
        args = parser.parse_args(["--hand-source", source])
        assert args.hand_source == source
    with pytest.raises(SystemExit):
        parser.parse_args(["--hand-source", "vive"])
    manus = parser.parse_args(["--hand-source", "manus", "--manus-calibration", "right.mcal"])
    launcher._validate_args(manus)
    with pytest.raises(SystemExit, match="require --hand-source manus"):
        launcher._validate_args(parser.parse_args(["--manus-calibration", "right.mcal"]))
    with pytest.raises(SystemExit, match="--manus-mode remote"):
        launcher._validate_args(parser.parse_args(["--hand-source", "manus", "--manus-core-host", "10.0.0.2"]))
    with pytest.raises(SystemExit, match="drop --view-only"):
        launcher._validate_args(parser.parse_args(["--view-only", "--screenshot-dir", "shots"]))


def test_create_worker_pins_the_wrist_to_the_selected_finger_source():
    args = launcher._parser().parse_args(["--hand-source", "hts"])
    source = _IdleSource()

    worker = launcher.create_worker(args, articulation_source=source)

    assert set(worker.sources) == {"quest", FIXED_WRIST_SOURCE_NAME}
    assert worker.articulation_source == "quest"
    assert worker.wrist_source == FIXED_WRIST_SOURCE_NAME
    assert isinstance(worker.sources["quest"], FixedWristSource)
    assert worker.sources["quest"] is worker.sources[FIXED_WRIST_SOURCE_NAME]
    assert worker.sources["quest"].articulation_source is source
    assert worker.retargeting_stream == "adaptive.quest+fixed_wrist"
    assert worker.handedness == Handedness.RIGHT
    # URDF order differs from the execution order; the finger adapter maps by name.
    assert set(worker.retargeter.joint_names) == set(SHARPA_ACTION_JOINTS)

import h5py
import pytest

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.arat.scene import (
    ROBOT_DATASET_NAME,
    ROBOT_END_EFFECTOR,
    ROBOT_MODEL,
    build_task_metadata,
    get_task_scene_data,
    get_task_scene_object_names,
    get_task_scene_path,
)
from dex_teleop.omnigibson.launcher import (
    GracefulShutdown,
    LEFT_SHOULDER_CAMERA_POSITION,
    RIGHT_SHOULDER_CAMERA_POSITION,
    ROBOT_COMPOSED_MODEL,
    WRIST_CAMERA_LINK,
    WRIST_CAMERA_POSITION,
    _parser,
    _finalize_recording,
    _reset_arat_box,
    _reset_goal_status_labels,
    _show_robot_end_effectors,
    _update_goal_status_labels,
    _validate_loaded_apparatus,
    build_environment_config,
    default_recording_path,
    recording_staging_path,
)


def test_catalog_has_required_task_and_layout_counts():
    catalog = AratTaskCatalog()

    assert catalog.placeholder_goals is False
    assert {name: len(tasks) for name, tasks in catalog.subscales.items()} == {
        "grasp": 6,
        "grip": 4,
        "pinch": 6,
        "gross_movement": 3,
    }
    assert len({task.layout for task in catalog.tasks.values()}) == 13
    assert catalog.tasks["arat_grasp_block_7_5cm"].label.endswith("7.5cm")


def test_subscale_resolution_preserves_declared_order():
    catalog = AratTaskCatalog()

    assert tuple(task.activity for task in catalog.resolve(None, "grasp")) == catalog.subscales["grasp"]


def test_every_task_maps_to_its_saved_version_1_scene():
    catalog = AratTaskCatalog()

    for task in catalog.tasks.values():
        scene_path = get_task_scene_path(task)
        scene = get_task_scene_data(task)
        names = get_task_scene_object_names(task)
        metadata = build_task_metadata(task)["inst_to_name"]

        assert scene_path.is_file()
        assert scene["init_info"]["class_name"] == "Scene"
        assert scene["init_info"]["args"]["use_floor_plane"] is True
        assert scene["init_info"]["args"]["floor_plane_color"] == [0.5, 0.5, 0.5]
        if task.subscale == "gross_movement":
            assert names == {"mannequin"}
            assert "breakfast_table.n.01_1" not in metadata
        else:
            assert "table" in names
            assert "arat_box" in names
            assert metadata["breakfast_table.n.01_1"] == "table"
        if task.activity == "arat_grip_pour_water":
            assert metadata["water.n.06_1"] == "water"
        assert metadata["agent.n.01_1"] == "franka_sharpa_right"
        assert set(metadata.values()).difference({"franka_sharpa_right", "water"}).issubset(names)


def test_planks_and_bolts_have_separate_bddl_instances():
    catalog = AratTaskCatalog()

    for activity in ("arat_grip_alloy_tube_1cm", "arat_grip_alloy_tube_2_5cm"):
        instances = catalog.tasks[activity].instances
        assert {key for key in instances if key.startswith("arat_plank.")} == {
            "arat_plank.n.01_1",
            "arat_plank.n.01_2",
        }
        assert {key for key in instances if key.startswith("arat_bolt.")} == {
            "arat_bolt.n.01_1",
            "arat_bolt.n.01_2",
        }
        assert not any(key.startswith("arat_tube_fixture.") for key in instances)

    washer_instances = catalog.tasks["arat_grip_washer_over_bolt"].instances
    assert washer_instances["arat_plank.n.01_1"] == "plank_target_point"
    assert washer_instances["arat_bolt.n.01_1"] == "bolt_target_point"


def test_every_bddl_definition_compiles_and_matches_saved_scope():
    from bddl.activity import Conditions, get_initial_conditions, get_object_scope

    catalog = AratTaskCatalog()
    for task in catalog.tasks.values():
        conditions = Conditions(task.activity, 0, "behavior-1k")
        scope = get_object_scope(conditions)
        initial_conditions = get_initial_conditions(conditions, scope)
        saved_metadata = get_task_scene_data(task)["metadata"]["task"]["inst_to_name"]

        assert initial_conditions
        assert set(scope) == set(saved_metadata)
        # The visible floor is the simulator's ground-plane primitive, not a
        # DatasetObject that can participate in OmniGibson object states.
        assert "floor.n.01_1" not in scope


def test_launcher_uses_compatible_frequency_and_version_1_scene():
    task = AratTaskCatalog().tasks["arat_grasp_block_10cm"]
    config = build_environment_config(task)

    assert config["env"]["action_frequency"] == 30.0
    assert config["scene"]["type"] == "Scene"
    assert config["scene"]["use_floor_plane"] is True
    assert config["scene"]["floor_plane_color"] == [0.5, 0.5, 0.5]
    assert config["scene"]["scene_file"]["metadata"]["task"]["activity"] == task.activity
    robot = config["robots"][0]
    assert robot["model"] == ROBOT_MODEL == "franka"
    assert robot["dataset_name"] == ROBOT_DATASET_NAME == "omnigibson-robot-assets"
    assert robot["end_effector"] == ROBOT_END_EFFECTOR == "sharpa_right"
    cameras = {camera["name"]: camera for camera in config["env"]["external_sensors"]}
    assert set(cameras) == {
        "arat_left_shoulder_camera",
        "arat_right_shoulder_camera",
        "arat_wrist_camera",
    }
    assert cameras["arat_left_shoulder_camera"]["position"] == LEFT_SHOULDER_CAMERA_POSITION
    assert cameras["arat_right_shoulder_camera"]["position"] == RIGHT_SHOULDER_CAMERA_POSITION
    assert cameras["arat_left_shoulder_camera"]["pose_frame"] == "scene"
    assert cameras["arat_right_shoulder_camera"]["pose_frame"] == "scene"
    wrist_camera = cameras["arat_wrist_camera"]
    assert wrist_camera["relative_prim_path"].startswith(f"/controllable__{ROBOT_COMPOSED_MODEL}__")
    assert f"/{WRIST_CAMERA_LINK}/" in wrist_camera["relative_prim_path"]
    assert wrist_camera["position"] == WRIST_CAMERA_POSITION
    assert wrist_camera["pose_frame"] == "parent"
    assert all(camera["modalities"] == [] and not camera["include_in_obs"] for camera in cameras.values())


def test_launcher_accepts_recording_path():
    catalog = AratTaskCatalog()

    args = _parser(catalog).parse_args(
        ["--task", "arat_grasp_block_10cm", "--recording-path", "outputs/arat_demo.hdf5"]
    )

    assert args.recording_path == "outputs/arat_demo.hdf5"


def test_launcher_default_recording_path_is_task_specific():
    task = AratTaskCatalog().tasks["arat_grasp_block_10cm"]

    assert default_recording_path(task).as_posix().endswith(
        "/dex_teleop/outputs/recordings/arat_grasp_block_10cm.hdf5"
    )


def test_sigint_requests_graceful_shutdown_without_raising():
    shutdown = GracefulShutdown()

    shutdown(None, None)
    shutdown(None, None)

    assert shutdown.requested is True


def test_recording_is_atomically_published_after_close(tmp_path):
    output_path = tmp_path / "demo.hdf5"
    staging_path = recording_staging_path(output_path, process_id=123)
    output_path.write_bytes(b"old valid recording")
    staging_path.write_bytes(b"new valid recording")

    class RecordingEnvironment:
        saved = False

        def save_data(self):
            self.saved = True

    recording_env = RecordingEnvironment()
    _finalize_recording(recording_env, staging_path, output_path)

    assert recording_env.saved is True
    assert output_path.read_bytes() == b"new valid recording"
    assert not staging_path.exists()


def test_failed_recording_close_preserves_previous_output(tmp_path):
    output_path = tmp_path / "demo.hdf5"
    staging_path = recording_staging_path(output_path, process_id=123)
    output_path.write_bytes(b"old valid recording")
    staging_path.write_bytes(b"incomplete recording")

    class RecordingEnvironment:
        def save_data(self):
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        _finalize_recording(RecordingEnvironment(), staging_path, output_path)

    assert output_path.read_bytes() == b"old valid recording"
    assert staging_path.read_bytes() == b"incomplete recording"


def test_failed_evaluation_append_preserves_previous_output(tmp_path):
    output_path = tmp_path / "demo.hdf5"
    staging_path = recording_staging_path(output_path, process_id=123)
    output_path.write_bytes(b"old valid recording")
    with h5py.File(staging_path, "w") as recording:
        data = recording.create_group("data")
        data.create_group("demo_0").attrs["num_samples"] = 1

    class RecordingEnvironment:
        def save_data(self):
            pass

    with pytest.raises(RuntimeError, match="has 0 steps; trajectory has 1"):
        _finalize_recording(RecordingEnvironment(), staging_path, output_path, [[]])

    assert output_path.read_bytes() == b"old valid recording"
    assert staging_path.exists()


def test_goal_status_labels_follow_bddl_satisfaction():
    class Label:
        selected = False

    labels = [Label(), Label(), Label()]
    _update_goal_status_labels(labels, {"satisfied": [0, 2], "unsatisfied": [1]})

    assert [label.selected for label in labels] == [True, False, True]

    _reset_goal_status_labels(labels)
    assert not any(label.selected for label in labels)


def test_view_only_config_omits_robot_and_behavior_task():
    task = AratTaskCatalog().tasks["arat_grasp_block_10cm"]
    config = build_environment_config(task, view_only=True)

    assert config["robots"] == []
    assert config["task"] == {"type": "DummyTask"}
    assert config["scene"]["scene_file"]["metadata"]["task"]["activity"] == task.activity
    assert [camera["name"] for camera in config["env"]["external_sensors"]] == [
        "arat_left_shoulder_camera",
        "arat_right_shoulder_camera",
    ]


def test_box_reset_is_a_noop_for_mannequin_only_scene():
    class ObjectRegistry:
        def __call__(self, key, value):
            assert (key, value) == ("name", "arat_box")
            return None

    class Scene:
        object_registry = ObjectRegistry()

    class Environment:
        scene = Scene()

    _reset_arat_box(Environment())


def test_robot_end_effector_visuals_are_restored():
    class Link:
        visible = False

    class Robot:
        arm_names = ["0"]
        eef_link_names = {"0": "right_hand_C_MC"}
        links = {"right_hand_C_MC": Link()}

    robot = Robot()
    _show_robot_end_effectors(robot)

    assert robot.links["right_hand_C_MC"].visible is True


def test_gross_movement_runtime_validation_allows_intended_robot():
    class Mannequin:
        category = "mannequin"
        model = "nphsfp"

    class Robot:
        name = "franka_sharpa_right"

    class ObjectRegistry:
        def __call__(self, key, value):
            assert (key, value) == ("name", "mannequin")
            return Mannequin()

        def get_dict(self, key):
            assert key == "name"
            return {"mannequin": Mannequin(), "franka_sharpa_right": Robot()}

    class Scene:
        object_registry = ObjectRegistry()

    class Environment:
        scene = Scene()
        robots = [Robot()]

    task = AratTaskCatalog().tasks["arat_gross_movement_hand_behind_head"]
    _validate_loaded_apparatus(Environment(), task)

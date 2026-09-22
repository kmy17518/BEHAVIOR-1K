import h5py
import pytest

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.arat.camera_rig import camera_rig_names, layout_camera_ids, load_camera_rig
from dex_teleop.arat.reset_poses import (
    DEFAULT_RESET_POSE,
    RESET_POSES,
    reset_joint_positions,
)
from dex_teleop.arat.scene import (
    ROBOT_DATASET_NAME,
    ROBOT_END_EFFECTOR,
    ROBOT_MODEL,
    build_task_metadata,
    get_task_scene_data,
    get_task_scene_object_names,
    get_task_scene_path,
    get_task_tro_data,
    get_task_tro_path,
)
from dex_teleop.omnigibson.launcher import (
    GracefulShutdown,
    ROBOT_PRIM_PATH,
    _camera_layout_panel_dock,
    _parser,
    _finalize_recording,
    _reset_and_wait_for_tracking,
    _resolve_tracking_selection,
    _reset_arat_box,
    _reset_goal_status_labels,
    _set_hand_tracking_metadata,
    _set_viewport_resolution,
    _shutdown_omnigibson,
    _show_robot_end_effectors,
    _update_goal_status_labels,
    _validate_loaded_apparatus,
    build_environment_config,
    default_recording_path,
    main,
    recording_staging_path,
)
from dex_teleop.omnigibson.hand_tracking_recording import HandTrackingRecordingSession
from dex_teleop.tracking import SourceUnavailableError


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
    assert {task.camera_rig for task in catalog.tasks.values()} == {"arat_default"}


def test_subscale_resolution_preserves_declared_order():
    catalog = AratTaskCatalog()

    assert tuple(task.activity for task in catalog.resolve(None, "grasp")) == catalog.subscales["grasp"]


def test_every_task_maps_to_its_saved_version_1_scene():
    catalog = AratTaskCatalog()

    for task in catalog.tasks.values():
        scene_path = get_task_scene_path(task)
        tro_path = get_task_tro_path(task)
        scene = get_task_scene_data(task)
        tro = get_task_tro_data(task)
        names = get_task_scene_object_names(task)
        metadata = build_task_metadata(task)["inst_to_name"]

        assert scene_path.is_file()
        assert tro_path.is_file()
        assert len(tro["robot_poses"]["robot"]) == 1
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
    robot_pose = get_task_tro_data(task)["robot_poses"]["robot"][0]
    assert robot["position"] == robot_pose["position"]
    assert robot["orientation"] == robot_pose["orientation"]
    assert config["task"]["use_presampled_robot_pose"] is True
    assert config["scene"]["scene_file"]["metadata"]["task"]["robot_poses"] == {"robot": [robot_pose]}
    cameras = {camera["name"]: camera for camera in config["env"]["external_sensors"]}
    assert set(cameras) == {
        "arat_mobile_chest_camera",
        "arat_mobile_overview_camera",
        "arat_left_shoulder_camera",
        "arat_right_shoulder_camera",
        "arat_wrist_camera_thumb",
        "arat_wrist_camera_pinky",
    }
    camera_rig = load_camera_rig(task.camera_rig)
    assert cameras["arat_left_shoulder_camera"]["position"] == camera_rig.camera("left_shoulder")["pose"]["position"]
    assert cameras["arat_right_shoulder_camera"]["position"] == camera_rig.camera("right_shoulder")["pose"]["position"]
    assert cameras["arat_left_shoulder_camera"]["pose_frame"] == "scene"
    assert cameras["arat_right_shoulder_camera"]["pose_frame"] == "scene"
    for sensor_name, camera_id in (
        ("arat_wrist_camera_thumb", "thumb_wrist"),
        ("arat_wrist_camera_pinky", "pinky_wrist"),
    ):
        wrist_camera = cameras[sensor_name]
        assert wrist_camera["relative_prim_path"].startswith(ROBOT_PRIM_PATH)
        assert "/right_hand_C_MC/" in wrist_camera["relative_prim_path"]
        assert wrist_camera["position"] == camera_rig.camera(camera_id)["pose"]["position"]
        assert wrist_camera["pose_frame"] == "parent"
    assert cameras["arat_mobile_chest_camera"]["pose_frame"] == "scene"
    assert cameras["arat_mobile_overview_camera"]["pose_frame"] == "scene"
    assert all(camera["modalities"] == [] and not camera["include_in_obs"] for camera in cameras.values())
    assert robot["reset_joint_pos"] == reset_joint_positions("extended")


def test_reset_pose_aliases_compact_asset_and_extended_deoxys():
    compact = reset_joint_positions("compact")
    extended = reset_joint_positions("extended")

    assert compact[:7] == [0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]
    assert compact[7:] == [0.0] * 22
    assert extended[:7] == [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]
    assert extended[7:] == [0.0] * 22
    assert reset_joint_positions() == list(RESET_POSES[DEFAULT_RESET_POSE])
    with pytest.raises(ValueError, match="Unknown reset pose"):
        reset_joint_positions("tuck")


def test_launcher_reset_pose_defaults_to_extended_and_selects_compact():
    catalog = AratTaskCatalog()
    task = catalog.tasks["arat_grasp_block_10cm"]

    default_args = _parser(catalog).parse_args(["--task", "arat_grasp_block_10cm"])
    compact_args = _parser(catalog).parse_args(
        ["--task", "arat_grasp_block_10cm", "--reset-pose", "compact"]
    )

    assert default_args.reset_pose == "extended"
    assert compact_args.reset_pose == "compact"
    compact_config = build_environment_config(task, reset_pose="compact")
    assert compact_config["robots"][0]["reset_joint_pos"] == reset_joint_positions("compact")
    assert compact_config["robots"][0]["reset_joint_pos"] != reset_joint_positions("extended")


def test_camera_rig_declares_calibration_docking_and_toggle_cycles():
    camera_rig = load_camera_rig("arat_sharpa_v1")

    assert camera_rig.calibration("left_shoulder") == {
        "image_width": 640,
        "image_height": 480,
        "horizontal_aperture": 20.955,
        "focal_length": 17.0,
    }
    assert layout_camera_ids(camera_rig, "teleop") == (
        "left_shoulder",
        "right_shoulder",
        "thumb_wrist",
        "pinky_wrist",
        "overview",
    )
    teleop = camera_rig.layout("teleop")
    assert teleop["viewports"]["thumb_wrist"]["dock"] == {
        "parent": "right_shoulder",
        "position": "bottom",
        "ratio": 0.5,
    }
    assert teleop["viewports"]["pinky_wrist"]["dock"]["parent"] == "thumb_wrist"
    assert teleop["toggles"] == [
        {"key": "B", "viewport": "main", "cameras": ["left_shoulder", "overview"]}
    ]


def test_arat_default_camera_rig_is_default_and_selectable():
    assert set(camera_rig_names()) == {"arat_sharpa_v1", "arat_default"}
    camera_rig = load_camera_rig("arat_default")

    assert layout_camera_ids(camera_rig, "teleop") == (
        "chest",
        "overview",
        "left_shoulder",
        "right_shoulder",
        "thumb_wrist",
        "pinky_wrist",
    )
    teleop = camera_rig.layout("teleop")
    assert teleop["workspace"] == {"viewport_only": True, "fill_viewports": False}
    assert camera_rig.layout("view_only")["workspace"] == {
        "viewport_only": True,
        "fill_viewports": False,
    }
    assert teleop["viewports"]["main"]["camera"] == "chest"
    assert "left_lower" not in teleop["viewports"]
    assert "left_lower" not in camera_rig.layout("view_only")["viewports"]
    assert teleop["viewports"]["thumb_wrist"]["camera"] == "thumb_wrist"
    assert teleop["viewports"]["pinky_wrist"]["camera"] == "pinky_wrist"
    assert teleop["viewports"]["left_shoulder"]["dock"]["ratio"] == 0.25
    assert teleop["viewports"]["right_shoulder"]["dock"]["ratio"] == 0.20
    assert teleop["viewports"]["thumb_wrist"]["dock"]["ratio"] == 0.5
    assert teleop["viewports"]["pinky_wrist"]["dock"] == {
        "parent": "thumb_wrist",
        "position": "bottom",
        "ratio": 0.5,
    }
    assert teleop["toggles"] == [
        {"key": "B", "viewport": "main", "cameras": ["chest", "overview"]}
    ]
    assert teleop["panels"] == {
        "emg": {"dock": {"parent": "Content", "position": "same", "ratio": 0.5}},
        "decoder": {"dock": {"parent": "Content", "position": "same", "ratio": 0.5}},
    }

    task = AratTaskCatalog().tasks["arat_grasp_block_5cm"]
    cameras = {camera["name"]: camera for camera in build_environment_config(task)["env"]["external_sensors"]}
    assert set(cameras) == {
        "arat_mobile_chest_camera",
        "arat_mobile_overview_camera",
        "arat_left_shoulder_camera",
        "arat_right_shoulder_camera",
        "arat_wrist_camera_thumb",
        "arat_wrist_camera_pinky",
    }
    chest_camera = cameras["arat_mobile_chest_camera"]
    assert chest_camera["sensor_kwargs"]["horizontal_aperture"] == 40.0
    assert chest_camera["sensor_kwargs"]["image_width"] == 1080
    assert chest_camera["sensor_kwargs"]["image_height"] == 1080
    assert chest_camera["position"] == [-0.6685013461, 0.0915177357, 1.45]
    assert chest_camera["orientation"] == [-0.3312225796, 0.4222105567, 0.6639027239, -0.5208291676]
    assert cameras["arat_left_shoulder_camera"]["position"] == [-0.2416171131, 0.5654538916, 1.6015079823]
    assert cameras["arat_left_shoulder_camera"]["orientation"] == [
        -0.1195093352,
        0.4460149108,
        0.8567866720,
        -0.2295752968,
    ]
    assert cameras["arat_right_shoulder_camera"]["position"] == [-0.3042717373, -0.8909394127, 1.3501765125]
    assert cameras["arat_right_shoulder_camera"]["orientation"] == [
        -0.5677569555,
        0.1521300177,
        0.2093890060,
        -0.7814504088,
    ]
    assert cameras["arat_left_shoulder_camera"]["sensor_kwargs"]["image_width"] == 256
    assert cameras["arat_left_shoulder_camera"]["sensor_kwargs"]["image_height"] == 256
    for name, lateral_offset in (("arat_wrist_camera_thumb", 0.072), ("arat_wrist_camera_pinky", -0.072)):
        assert cameras[name]["relative_prim_path"].startswith(f"{ROBOT_PRIM_PATH}/right_hand_C_MC/")
        assert cameras[name]["position"] == [-0.035, lateral_offset, 0.090]
        assert cameras[name]["orientation"] == [0.5, -0.5, -0.5, 0.5]
        assert cameras[name]["sensor_kwargs"]["image_width"] == 640
        assert cameras[name]["sensor_kwargs"]["image_height"] == 480
        assert cameras[name]["sensor_kwargs"]["focal_length"] == 2.807438
    assert cameras["arat_mobile_overview_camera"]["orientation"] == [
        0.2705931676,
        -0.2705931676,
        -0.6532835048,
        0.6532835048,
    ]


def test_fixed_viewport_resolution_updates_widget_and_renderer_aspect_ratio():
    class ViewportAPI:
        fill_frame = True

    class ViewportWidget:
        fill_frame = True
        resolution = (1280, 720)

    class Viewport:
        viewport_api = ViewportAPI()
        viewport_widget = ViewportWidget()

    viewport = Viewport()
    _set_viewport_resolution(viewport, (256, 256), fill_frame=False)

    assert viewport.viewport_api.fill_frame is False
    assert viewport.viewport_widget.fill_frame is False
    assert viewport.viewport_widget.resolution == (256, 256)


def test_mobile_manipulator_cli_and_lower_panel_docks():
    args = _parser(AratTaskCatalog()).parse_args(
        [
            "--task",
            "arat_grasp_block_5cm",
            "--camera-rig",
            "arat_default",
            "--display",
            "decoder",
        ]
    )
    assert args.camera_rig == "arat_default"
    assert args.display == "decoder"

    class Environment:
        _arat_camera_layout = load_camera_rig("arat_default").layout("teleop")
        _arat_camera_windows = {}

    assert _camera_layout_panel_dock(
        Environment(),
        "emg",
        default_parent="Viewport",
        default_position="right",
        default_ratio=0.25,
    ) == ("Content", "same", 0.5)


def test_all_task_tros_use_the_tuned_grasp_block_5cm_robot_pose():
    catalog = AratTaskCatalog()

    expected_pose = {
        "position": [-0.4685013461, -0.4515322578, 0.0156103678],
        "orientation": [0, 0, 0, 1],
    }
    for task in catalog.tasks.values():
        robot_pose = get_task_tro_data(task)["robot_poses"]["robot"][0]
        assert robot_pose == expected_pose
        assert build_environment_config(task)["robots"][0]["position"] == expected_pose["position"]

    # At the extended reset pose, the visual hand reaches +0.054439 m from
    # the robot root toward the ARAT box. Its closest edge therefore clears
    # the box's negative-Y (right) edge by about 1.56 cm.
    hand_edge_nearest_box_y = expected_pose["position"][1] + 0.0544390706
    arat_box_right_edge_y = -0.3815353783
    assert arat_box_right_edge_y - hand_edge_nearest_box_y == pytest.approx(0.0155578089)


def test_launcher_accepts_recording_path():
    catalog = AratTaskCatalog()

    args = _parser(catalog).parse_args(
        ["--task", "arat_grasp_block_10cm", "--recording-path", "outputs/arat_demo.hdf5"]
    )

    assert args.recording_path == "outputs/arat_demo.hdf5"


def test_launcher_hand_pose_recording_is_opt_in():
    catalog = AratTaskCatalog()

    default_args = _parser(catalog).parse_args(["--task", "arat_grasp_block_10cm"])
    enabled_args = _parser(catalog).parse_args(
        ["--task", "arat_grasp_block_10cm", "--record-hand-poses"]
    )

    assert default_args.record_hand_poses is False
    assert enabled_args.record_hand_poses is True


def test_launcher_accepts_assisted_grasp_diagnostic_controls():
    catalog = AratTaskCatalog()

    default_args = _parser(catalog).parse_args(["--task", "arat_grasp_cricket_ball"])
    args = _parser(catalog).parse_args(
        [
            "--task",
            "arat_grasp_cricket_ball",
            "--assisted-grasp",
            "--assisted-grasp-debug",
            "--assisted-grasp-break-force",
            "none",
            "--assisted-grasp-break-torque",
            "45",
            "--assisted-grasp-squeeze-bias-rad",
            "0.02",
        ]
    )

    assert default_args.assisted_grasp_debug is False
    assert default_args.assisted_grasp_break_force == 100.0
    assert default_args.assisted_grasp_break_torque == 30.0
    assert default_args.assisted_grasp_squeeze_bias_rad == 0.05
    assert args.assisted_grasp_debug is True
    assert args.assisted_grasp_break_force is None
    assert args.assisted_grasp_break_torque == 45.0
    assert args.assisted_grasp_squeeze_bias_rad == 0.02


def test_assisted_grasp_debug_requires_assisted_grasp():
    with pytest.raises(SystemExit, match="requires --assisted-grasp"):
        main(["--task", "arat_grasp_cricket_ball", "--assisted-grasp-debug"])


def test_launcher_tracking_defaults_preserve_the_quest_hts_preset():
    args = _parser(AratTaskCatalog()).parse_args(["--task", "arat_grasp_block_10cm"])

    selection = _resolve_tracking_selection(args)

    assert args.source is None
    assert selection.hand_source == "quest"
    assert selection.wrist_source == "quest"
    assert selection.articulation_sources == ("quest",)
    assert selection.wrist_sources == ("quest",)


def test_launcher_accepts_manus_quest_control_and_native_rate_comparison():
    args = _parser(AratTaskCatalog()).parse_args(
        [
            "--task",
            "arat_grasp_block_10cm",
            "--hand-source",
            "manus",
            "--wrist-source",
            "quest",
            "--record-hand-source",
            "quest",
            "--record-hand-source",
            "hts",
            "--retargeter",
            "dexpilot",
        ]
    )

    selection = _resolve_tracking_selection(args)

    assert selection.hand_source == "manus"
    assert selection.wrist_source == "quest"
    assert selection.record_hand_sources == ("quest",)
    assert selection.articulation_sources == ("manus", "quest")
    assert args.retargeter == "dexpilot"

    vibe_args = _parser(AratTaskCatalog()).parse_args(
        ["--task", "arat_grasp_block_10cm", "--wrist-source", "vibe"]
    )
    assert _resolve_tracking_selection(vibe_args).wrist_source == "vive"


def test_launcher_rejects_mixing_legacy_and_component_control_flags():
    args = _parser(AratTaskCatalog()).parse_args(
        [
            "--task",
            "arat_grasp_block_10cm",
            "--source",
            "hts",
            "--hand-source",
            "manus",
        ]
    )

    with pytest.raises(SystemExit, match="cannot be combined"):
        _resolve_tracking_selection(args)


def test_launcher_reset_waits_for_every_configured_tracking_role():
    calls = []
    expected_snapshot = object()

    class Worker:
        def reset(self):
            calls.append("reset")

        def wait_for_first(self, timeout):
            calls.append(("wait", timeout))
            return expected_snapshot

    assert (
        _reset_and_wait_for_tracking(Worker(), 3.5, context="test boundary")
        is expected_snapshot
    )
    assert calls == ["reset", ("wait", 3.5)]


def test_launcher_tracking_readiness_preserves_missing_role_detail():
    class Worker:
        def reset(self):
            pass

        def wait_for_first(self, _timeout):
            raise SourceUnavailableError("missing record-only articulation source 'quest'")

    with pytest.raises(
        SourceUnavailableError,
        match="test boundary: missing record-only articulation source 'quest'",
    ):
        _reset_and_wait_for_tracking(Worker(), 1.0, context="test boundary")


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--maximum-frame-age", "nan"),
        ("--maximum-frame-age", "inf"),
        ("--initial-frame-timeout", "nan"),
        ("--initial-frame-timeout", "inf"),
    ),
)
def test_launcher_rejects_nonfinite_tracking_timeouts(option, value):
    with pytest.raises(SystemExit, match="source/frame timeouts"):
        main(["--task", "arat_grasp_block_10cm", option, value])


def test_launcher_records_source_and_retargeter_configuration_metadata():
    args = _parser(AratTaskCatalog()).parse_args(["--task", "arat_grasp_block_10cm"])
    source = object()

    class _Worker:
        articulation_streams = {"quest": "articulation.quest"}
        wrist_streams = {"quest": "wrist.quest"}
        wrist_source = "quest"
        control_wrist_stream = "wrist.quest.control"
        retargeting_stream = "adaptive.quest+quest"
        sources = {"quest": source}

    session = HandTrackingRecordingSession()
    _set_hand_tracking_metadata(session, _Worker(), args)
    metadata = session.writer_kwargs()["stream_metadata"]

    assert metadata["articulation.quest"]["provider"] == "Quest Hand Tracking Streamer"
    assert metadata["wrist.quest"]["endpoint"] == "udp://0.0.0.0:9000"
    assert metadata["wrist.quest.control"]["maximum_skew_seconds"] == 0.05
    assert metadata["adaptive.quest+quest"]["configuration"]["sha256"] != "unavailable"


def test_launcher_arm_marker_visualization_is_opt_in_with_workspace_alias():
    catalog = AratTaskCatalog()

    default_args = _parser(catalog).parse_args(["--task", "arat_grasp_block_10cm"])
    enabled_args = _parser(catalog).parse_args(
        ["--task", "arat_grasp_block_10cm", "--visualize-arm-markers"]
    )
    alias_args = _parser(catalog).parse_args(
        ["--task", "arat_grasp_block_10cm", "--visualize-arm-workspace"]
    )

    assert default_args.visualize_arm_markers is False
    assert enabled_args.visualize_arm_markers is True
    assert alias_args.visualize_arm_markers is True


def test_launcher_accepts_tracking_yaw_and_stale_hold_policy():
    default_args = _parser(AratTaskCatalog()).parse_args(["--task", "arat_grasp_block_10cm"])
    args = _parser(AratTaskCatalog()).parse_args(
        [
            "--task",
            "arat_grasp_block_10cm",
            "--tracking-yaw-deg",
            "180",
            "--position-sensitivity",
            "1.75",
            "--maximum-frame-age",
            "0.5",
            "--stale-frame-policy",
            "hold",
            "--wrist-joint-rotation-threshold-deg",
            "140",
            "--wrist-joint-rotation-window-s",
            "4",
            "--wrist-joint-limit-margin-deg",
            "3",
        ]
    )

    assert default_args.position_sensitivity == 1.5
    assert args.tracking_yaw_deg == 180.0
    assert args.position_sensitivity == 1.75
    assert args.maximum_frame_age == 0.5
    assert args.stale_frame_policy == "hold"
    assert args.wrist_joint_rotation_threshold_deg == 140.0
    assert args.wrist_joint_rotation_window_s == 4.0
    assert args.wrist_joint_limit_margin_deg == 3.0


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


@pytest.mark.parametrize("exit_code", (None, 0))
def test_omnigibson_successful_prelaunch_shutdown_is_not_an_error(exit_code):
    class OmniGibson:
        shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            raise SystemExit(exit_code)

    omnigibson = OmniGibson()
    _shutdown_omnigibson(omnigibson)

    assert omnigibson.shutdown_calls == 1


@pytest.mark.parametrize("exit_code", (2, "failure"))
def test_omnigibson_failed_prelaunch_shutdown_is_propagated(exit_code):
    class OmniGibson:
        def shutdown(self):
            raise SystemExit(exit_code)

    with pytest.raises(SystemExit) as caught:
        _shutdown_omnigibson(OmniGibson())

    assert caught.value.code == exit_code


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


def test_preflight_failure_can_close_staging_without_publishing(tmp_path):
    output_path = tmp_path / "demo.hdf5"
    staging_path = recording_staging_path(output_path, process_id=123)
    output_path.write_bytes(b"old valid recording")
    staging_path.write_bytes(b"closed diagnostic recording")

    class RecordingEnvironment:
        saved = False

        def save_data(self):
            self.saved = True

    recording_env = RecordingEnvironment()
    _finalize_recording(
        recording_env,
        staging_path,
        output_path,
        publish=False,
    )

    assert recording_env.saved is True
    assert output_path.read_bytes() == b"old valid recording"
    assert staging_path.read_bytes() == b"closed diagnostic recording"


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
        "arat_mobile_chest_camera",
        "arat_mobile_overview_camera",
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

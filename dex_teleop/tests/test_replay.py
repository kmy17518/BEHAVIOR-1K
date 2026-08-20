import json
from pathlib import Path

import h5py
import pytest

from dex_teleop.omnigibson.replay import (
    _condition_is_applicable,
    _parser,
    RecordingInfo,
    build_replay_camera_configs,
    default_output_path,
    get_episode_lengths,
    inspect_recording,
    prompt_for_episode,
    render_evaluation_panel,
    score_condition_sections,
    select_episode,
)
from dex_teleop.omnigibson.evaluation_trace import (
    build_step_evaluation,
    read_evaluation_episode,
    write_evaluation_episodes,
)


def _write_recording(path: Path) -> None:
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.attrs["config"] = json.dumps(
            {"task": {"type": "BehaviorTask", "activity_name": "arat_grasp_block_10cm"}}
        )
        data.create_group("demo_bad").attrs["num_samples"] = 999
        data.create_group("demo_4").attrs["num_samples"] = 12
        data.create_group("demo_1").attrs["num_samples"] = 30


def test_recording_inspection_infers_task_and_sorts_episodes(tmp_path):
    recording_path = tmp_path / "demo.hdf5"
    _write_recording(recording_path)

    info = inspect_recording(recording_path)

    assert info.task_name == "arat_grasp_block_10cm"
    assert info.episodes == ((1, "demo_1", 30), (4, "demo_4", 12))
    assert select_episode(info) == (1, 30)
    assert select_episode(info, 4) == (4, 12)


def test_recording_episode_selection_rejects_unknown_id(tmp_path):
    recording_path = tmp_path / "demo.hdf5"
    _write_recording(recording_path)

    with pytest.raises(ValueError, match=r"available IDs: \[1, 4\]"):
        select_episode(inspect_recording(recording_path), 3)


def test_default_video_path_includes_episode_id(tmp_path):
    assert default_output_path(tmp_path / "demo.hdf5", 4) == tmp_path / "demo_demo_4.mp4"


@pytest.mark.parametrize("camera_rig_name", ["arat_default", "arat_sharpa_v1"])
def test_replay_camera_configs_enable_all_four_rgb_observations(camera_rig_name):
    cameras = build_replay_camera_configs(camera_rig_name)

    assert {camera["name"] for camera in cameras} == {
        "arat_left_shoulder_camera",
        "arat_right_shoulder_camera",
        "arat_wrist_camera_thumb",
        "arat_wrist_camera_pinky",
    }
    assert all(camera["modalities"] == ["rgb"] for camera in cameras)
    assert all(camera["include_in_obs"] is True for camera in cameras)
    assert all(camera["sensor_kwargs"]["viewport_name"] is None for camera in cameras)


def test_get_episode_lengths_ignores_non_numeric_demo_groups(tmp_path):
    recording_path = tmp_path / "demo.hdf5"
    _write_recording(recording_path)

    with h5py.File(recording_path, "r") as recording:
        assert get_episode_lengths(recording["data"]) == ((1, "demo_1", 30), (4, "demo_4", 12))


def _evaluation_step(score=1):
    return build_step_evaluation(
        ["the block is on the shelf", "the hand is not touching the block"],
        {"satisfied": [0], "unsatisfied": [1]},
        {
            "task_time_s": 1.25,
            "provisional_score": score,
            "score_reasons": ["held_and_lifted_only"],
            "decision": f"{score}/3 if the item ended now: held_and_lifted_only",
            "completed_for_scoring": False,
            "finished": False,
            "conditions": [
                {
                    "key": "held_and_lifted",
                    "label": "Object held and lifted",
                    "met": True,
                    "detail": "lift exceeded 0.020 m for 0.200 s",
                },
                {
                    "key": "clean_release",
                    "label": "Clean release and settle (score 3)",
                    "met": False,
                    "detail": "not completed",
                },
            ],
            "evidence": {
                "contact_pads": ["thumb", "index"],
                "contact_dorsals": [],
                "palm_contact": False,
                "at_target": False,
                "reached_target_height": True,
                "palmar_region_contact": False,
                "dorsal_region_contact": False,
                "approach_progress_m": 0.2,
                "apertures_m": {"index": 0.06},
                "tracked_speed_m_s": 0.03,
                "environment_supports": [],
                "dorsum_push_detected": False,
                "braced_grasp_suspected": False,
                "release_in_progress": False,
                "release_fumbled": False,
                "active_hold": True,
                "scoring_hold": None,
                "water": None,
            },
            "events": [{"t": 1.0, "name": "first_hold_and_lift"}],
            "snapshot": {},
        },
    )


def test_evaluation_trace_round_trips_in_separate_hdf5_group(tmp_path):
    recording_path = tmp_path / "demo.hdf5"
    with h5py.File(recording_path, "w") as recording:
        data = recording.create_group("data")
        data.create_group("demo_0").attrs["num_samples"] = 2

    expected = [_evaluation_step(1), _evaluation_step(2)]
    write_evaluation_episodes(recording_path, [expected])

    assert read_evaluation_episode(recording_path, 0, expected_steps=2) == expected
    with h5py.File(recording_path, "r") as recording:
        assert "evaluation" not in recording["data"]["demo_0"]
        assert len(recording["evaluation"]["demo_0"]["steps"]) == 2


def test_missing_evaluation_trace_has_actionable_error(tmp_path):
    recording_path = tmp_path / "demo.hdf5"
    _write_recording(recording_path)

    with pytest.raises(ValueError, match="Record it again"):
        read_evaluation_episode(recording_path, 1)


def test_evaluation_overlay_parser_accepts_descriptive_name_and_requested_alias():
    assert _parser().parse_args(["demo.hdf5", "--evaluation-overlay"]).evaluation_overlay is True
    assert _parser().parse_args(["demo.hdf5", "--log-eval"]).evaluation_overlay is True


def test_interactive_episode_selection_lists_and_returns_requested_episode(capsys):
    info = RecordingInfo(
        task_name="arat_grasp_block_10cm",
        episodes=((1, "demo_1", 30), (4, "demo_4", 12)),
    )

    selected = prompt_for_episode(info, input_fn=lambda _prompt: "4")

    assert selected == 4
    output = capsys.readouterr().out
    assert "demo_1" in output
    assert "demo_4" in output


def test_interactive_episode_selection_reprompts_for_invalid_id(capsys):
    info = RecordingInfo(
        task_name="arat_grasp_block_10cm",
        episodes=((1, "demo_1", 30), (4, "demo_4", 12)),
    )
    answers = iter(("not-an-id", "9", "1"))

    assert prompt_for_episode(info, input_fn=lambda _prompt: next(answers)) == 1
    output = capsys.readouterr().out
    assert "Enter one of these episode IDs" in output
    assert "Episode 9 is unavailable" in output


def test_evaluation_panel_renders_bddl_and_arat_detail():
    panel = render_evaluation_panel(
        _evaluation_step(),
        task_name="arat_grasp_block_10cm",
        step_index=0,
        total_steps=20,
        width=1920,
    )

    assert panel.shape == (600, 1920, 3)
    assert panel.dtype.name == "uint8"
    assert panel.max() > panel.min()


def test_overlay_omits_only_task_level_non_applicable_conditions():
    arat = {"completed_for_scoring": True}

    assert not _condition_is_applicable(
        {
            "key": "correct_pinch_opposition",
            "met": None,
            "detail": "not a pinch item",
        },
        arat,
    )
    assert _condition_is_applicable(
        {
            "key": "movement_started",
            "met": False,
            "detail": "maximum progress 0.076 m; required 0.080 m",
        },
        arat,
    )
    assert _condition_is_applicable(
        {
            "key": "clean_release",
            "met": None,
            "detail": "waiting for completion",
        },
        arat,
    )


def test_score_sections_are_stable_when_attempt_becomes_completed():
    conditions = [
        {"key": "voluntary_opening", "met": True, "detail": "opened", "label": "Opening"},
        {"key": "qualifying_score1_hold", "met": True, "detail": "held", "label": "Hold"},
        {"key": "completion_or_release_flaw", "met": False, "detail": "waiting", "label": "Complete"},
        {"key": "clean_release", "met": None, "detail": "waiting", "label": "Release"},
        {
            "key": "correct_pinch_opposition",
            "met": None,
            "detail": "not a pinch item",
            "label": "Pinch",
        },
    ]

    before = score_condition_sections(conditions, {"completed_for_scoring": False})
    after = score_condition_sections(conditions, {"completed_for_scoring": True})

    assert [[condition["key"] for condition in section] for _, section in before] == [
        ["voluntary_opening", "qualifying_score1_hold"],
        ["completion_or_release_flaw"],
        ["clean_release"],
    ]
    assert before == after

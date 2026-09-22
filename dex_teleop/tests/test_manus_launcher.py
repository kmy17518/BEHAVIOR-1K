from types import SimpleNamespace
import json

import pytest

from dex_teleop.omnigibson.launcher import (
    _effective_manus_hand_motion,
    _parser,
    _resolve_tracking_selection,
    _set_hand_tracking_metadata,
    _validate_manus_selection,
)
from dex_teleop.omnigibson.hand_tracking_recording import (
    HandTrackingRecordingSession,
)


def _arguments(*arguments):
    catalog = SimpleNamespace(tasks={"activity": object()}, subscales={})
    return _parser(catalog).parse_args(arguments)


def _write_calibration(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "core_world_frame": "manus_core_world_y_up_rh_z_to_viewer_m",
                "reference_frame": "robot_world",
                "reference_from_core_world": {
                    "translation": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "wrists": [
                    {
                        "handedness": "right",
                        "core_tracker_offset_applied": True,
                        "tracker_id": "ultimate-right",
                        "skeleton_wrist_to_anatomical_wrist": {
                            "translation": [0.0, 0.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_existing_manus_hand_invocation_remains_integrated_gloves_only():
    args = _arguments("--hand-source", "manus")
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    assert selection.hand_source == "manus"
    assert selection.wrist_source == "quest"
    assert args.manus_mode == "integrated"
    assert _effective_manus_hand_motion(args) == "auto"


def test_launcher_preserves_negotiated_manus_protocol_metadata():
    class _LegacySource:
        def recording_metadata(self):
            return {
                "bridge_protocol_version": 1,
                "supported_protocol_version": 2,
                "bridge_capabilities": ["integrated_wire_v1"],
            }

    source = _LegacySource()
    worker = SimpleNamespace(
        fuser=SimpleNamespace(articulation_to_wrist=None),
        articulation_streams={"manus": "articulation.manus"},
        wrist_streams={"manus": "wrist.manus"},
        retargeting_stream="retargeted",
        sources={"manus": source},
        articulation_source="manus",
        wrist_source="manus",
        control_wrist_stream="wrist.control",
    )
    session = HandTrackingRecordingSession()
    args = _arguments("--hand-source", "manus")

    _set_hand_tracking_metadata(session, worker, args)
    metadata = session.writer_kwargs()["stream_metadata"]

    assert metadata["articulation.manus"]["bridge_protocol_version"] == 1
    assert metadata["wrist.manus"]["bridge_protocol_version"] == 1


def test_remote_manus_can_serve_both_selected_modalities(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--hand-source",
        "manus",
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-host",
        "192.0.2.10",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "ultimate-right",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    assert selection.articulation_sources == ("manus",)
    assert selection.wrist_sources == ("manus",)
    assert _effective_manus_hand_motion(args) == "tracker"


def test_remote_manus_can_be_a_record_only_wrist_source(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--record-wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "ultimate-right",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)

    assert selection.wrist_source == "quest"
    assert selection.record_wrist_sources == ("manus",)


def test_remote_manus_wrist_policy_is_separate_from_optional_calibration_id(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    document = json.loads(calibration.read_text(encoding="utf-8"))
    document["wrists"][0]["tracker_id"] = None
    calibration.write_text(json.dumps(document), encoding="utf-8")
    args = _arguments(
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "ultimate-right",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    _validate_manus_selection(args, selection)


@pytest.mark.parametrize(
    ("policy_arguments", "message"),
    (
        (
            ("--manus-expected-tracker-user-id", "1"),
            "requires --manus-expected-tracker-id",
        ),
        (
            ("--manus-expected-tracker-id", "ultimate-right"),
            "requires --manus-expected-tracker-user-id",
        ),
    ),
)
def test_remote_manus_wrist_requires_complete_runtime_policy(
    tmp_path, policy_arguments, message
):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        *policy_arguments,
    )
    selection = _resolve_tracking_selection(args)

    with pytest.raises(SystemExit, match=message):
        _validate_manus_selection(args, selection)


def test_remote_manus_wrist_rejects_calibration_tracker_id_mismatch(tmp_path):
    calibration = _write_calibration(tmp_path / "calibration.json")
    args = _arguments(
        "--wrist-source",
        "manus",
        "--manus-mode",
        "remote",
        "--manus-core-calibration",
        str(calibration),
        "--manus-expected-tracker-id",
        "different-ultimate",
        "--manus-expected-tracker-user-id",
        "1",
    )
    selection = _resolve_tracking_selection(args)

    with pytest.raises(SystemExit, match="audit metadata must match"):
        _validate_manus_selection(args, selection)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ("--wrist-source", "manus"),
            "requires --manus-mode remote",
        ),
        (
            (
                "--hand-source",
                "manus",
                "--manus-core-calibration",
                "calibration.json",
            ),
            "--manus-core-calibration.*require --manus-mode remote",
        ),
        (
            (
                "--wrist-source",
                "manus",
                "--manus-mode",
                "remote",
            ),
            "requires --manus-core-calibration",
        ),
        (
            (
                "--wrist-source",
                "manus",
                "--manus-mode",
                "remote",
                "--manus-core-calibration",
                "calibration.json",
                "--manus-hand-motion",
                "imu",
            ),
            "requires --manus-hand-motion tracker",
        ),
        (
            (
                "--manus-mode",
                "remote",
                "--manus-core-host",
                "windows-core",
            ),
            "require a selected or record-only MANUS source",
        ),
        (
            (
                "--wrist-source",
                "manus",
                "--manus-mode",
                "remote",
                "--manus-core-calibration",
                "calibration.json",
                "--no-manus-tracker-diagnostics",
            ),
            "requires tracker monitoring",
        ),
    ),
)
def test_manus_launcher_rejects_unsafe_or_unused_combinations(arguments, message):
    args = _arguments(*arguments)
    selection = _resolve_tracking_selection(args)

    with pytest.raises(SystemExit, match=message):
        _validate_manus_selection(args, selection)

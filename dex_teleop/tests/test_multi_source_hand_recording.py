import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import h5py
import numpy as np
import pytest

import dex_teleop.omnigibson.hand_tracking_recording as recording_module
from dex_teleop.omnigibson.hand_tracking_recording import (
    HAND_TRACKING_GROUP,
    HAND_TRACKING_SCHEMA_VERSION,
    ActionHandSelection,
    HandTrackingRecordingSession,
    RetargetedCommandSample,
    write_multi_source_hand_recording,
)
from dex_teleop.types import (
    HandArticulationSample,
    Handedness,
    RetargetedHandCommand,
    WristPoseSample,
)


def _write_trajectory(path, lengths=((0, 2),)):
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        for episode_id, length in lengths:
            episode = data.create_group(f"demo_{episode_id}")
            episode.attrs["num_samples"] = length
            episode.create_dataset(
                "action", data=np.zeros((length, 3), dtype=np.float64)
            )
        legacy = recording.create_group("human_hand_pose")
        legacy.attrs["schema_version"] = 1
        legacy.create_dataset("compatibility_sentinel", data=np.array([4, 5, 6]))


def _articulation(
    timestamp,
    *,
    source,
    schema,
    names,
    offset=0.0,
    orientations=False,
    invalid=(),
    source_timestamp_ns=None,
    source_frame_id=None,
    coordinate_frame="wrist_local",
    provenance=None,
):
    positions = {
        name: np.array([index + offset, index * 0.1, index * 0.01], dtype=np.float64)
        for index, name in enumerate(names)
    }
    orientation_values = None
    if orientations:
        orientation_values = {
            name: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            for name in names[:-1]
        }
    return HandArticulationSample(
        timestamp=timestamp,
        receipt_timestamp=timestamp + 0.01,
        source_timestamp_ns=source_timestamp_ns,
        source_frame_id=source_frame_id,
        handedness=Handedness.RIGHT,
        joint_positions=positions,
        joint_orientations_xyzw=orientation_values,
        joint_validity={name: name not in invalid for name in names},
        source=source,
        schema=schema,
        coordinate_frame=coordinate_frame,
        confidence=0.8,
        provenance=provenance,
    )


def _wrist(
    timestamp,
    *,
    offset=0.0,
    source_frame_id=None,
    anatomical_frame="wrist_local",
    provenance=None,
):
    return WristPoseSample(
        timestamp=timestamp,
        receipt_timestamp=timestamp + 0.02,
        source_timestamp_ns=round(timestamp * 1e9),
        source_frame_id=source_frame_id,
        handedness=Handedness.RIGHT,
        position=np.array([offset, 0.2, 0.3]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        source="quest",
        reference_frame="quest_stage",
        anatomical_frame=anatomical_frame,
        confidence=0.9,
        provenance=provenance,
    )


def _command(timestamp, value):
    return RetargetedHandCommand(
        timestamp=timestamp,
        handedness=Handedness.RIGHT,
        hand_model="sharpa",
        joint_names=("joint_a", "joint_b"),
        joint_positions=np.array([value, value + 0.1]),
    )


def _fixture_streams():
    quest_names = ("wrist", "index_mcp", "index_tip")
    manus_names = ("wrist", "index_metacarpal", "index_proximal", "index_tip")
    articulation_streams = {
        "quest_right": [
            _articulation(
                1.0 + index * 0.25,
                source="quest",
                schema="mediapipe21",
                names=quest_names,
                offset=index,
                source_timestamp_ns=1_000_000_000 + index * 250_000_000,
                source_frame_id=10 + index,
                provenance={"native_index": index, "device": "quest"},
            )
            for index in range(4)
        ],
        "manus_right": [
            _articulation(
                1.0 + index * 0.5,
                source="manus",
                schema="manus_raw_25",
                names=manus_names,
                offset=10 + index,
                orientations=True,
                invalid=("index_tip",) if index == 1 else (),
                source_timestamp_ns=None,
                source_frame_id=20 + index,
                provenance={"native_index": index, "glove_id": "547A010E"},
            )
            for index in range(3)
        ],
    }
    wrist_streams = {
        "quest_wrist_right": [
            _wrist(
                1.02,
                offset=0.0,
                source_frame_id=30,
                provenance={"support_frame": 30},
            ),
            _wrist(
                2.02,
                offset=1.0,
                source_frame_id=31,
                provenance={"support_frame": 31},
            ),
        ]
    }
    retargeting_streams = {
        "manus_adaptive_sharpa": [
            RetargetedCommandSample(
                command=_command(1.02, 0.1),
                retargeter="adaptive",
                articulation_stream="manus_right",
                articulation_row=0,
                wrist_stream="quest_wrist_right",
                wrist_row=0,
            ),
            RetargetedCommandSample(
                command=_command(2.02, 0.2),
                retargeter="adaptive",
                articulation_stream="manus_right",
                articulation_row=2,
                wrist_stream="quest_wrist_right",
                wrist_row=1,
            ),
        ],
        "quest_shadow_adaptive_sharpa": [
            RetargetedCommandSample(
                command=_command(1.03, 0.3),
                retargeter="adaptive",
                articulation_stream="quest_right",
                articulation_row=0,
                wrist_stream="quest_wrist_right",
                wrist_row=0,
            ),
            RetargetedCommandSample(
                command=_command(2.03, 0.4),
                retargeter="adaptive",
                articulation_stream="quest_right",
                articulation_row=3,
                wrist_stream="quest_wrist_right",
                wrist_row=1,
            ),
        ],
    }
    selections = [
        ActionHandSelection(
            articulation_stream="manus_right",
            articulation_row=0,
            wrist_stream="quest_wrist_right",
            wrist_row=0,
            retargeting_stream="manus_adaptive_sharpa",
            retargeting_row=0,
            synchronization_skew_seconds=0.02,
        ),
        ActionHandSelection(
            articulation_stream="manus_right",
            articulation_row=2,
            wrist_stream="quest_wrist_right",
            wrist_row=1,
            retargeting_stream="manus_adaptive_sharpa",
            retargeting_row=1,
            synchronization_skew_seconds=0.02,
        ),
    ]
    return articulation_streams, wrist_streams, retargeting_streams, selections


def _decode_strings(dataset):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset
    ]


def test_multi_source_recording_keeps_native_rates_shadow_outputs_and_legacy_group(
    tmp_path,
):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation, wrists, retargeting, selections = _fixture_streams()

    write_multi_source_hand_recording(
        path,
        articulation_streams=articulation,
        wrist_streams=wrists,
        retargeting_streams=retargeting,
        action_alignment_episodes=[selections],
        stream_metadata={
            "manus_right": {
                "representation": "raw",
                "glove_id": "547A010E",
                "calibration_id": "user_right_547A010E_SDK3.1.1_2026-08-19",
            },
            "quest_right": {"representation": "canonical"},
        },
    )

    with h5py.File(path, "r") as recording:
        np.testing.assert_array_equal(
            recording["human_hand_pose/compatibility_sentinel"][:], [4, 5, 6]
        )
        assert recording["human_hand_pose"].attrs["schema_version"] == 1

        root = recording[HAND_TRACKING_GROUP]
        assert root.attrs["schema_version"] == HAND_TRACKING_SCHEMA_VERSION
        quest = root["articulation_streams/quest_right"]
        manus = root["articulation_streams/manus_right"]
        assert quest["joint_positions"].shape == (4, 3, 3)
        assert manus["joint_positions"].shape == (3, 4, 3)
        assert tuple(json.loads(manus.attrs["joint_names"])) == (
            "wrist",
            "index_metacarpal",
            "index_proximal",
            "index_tip",
        )
        assert json.loads(manus.attrs["metadata"])["glove_id"] == "547A010E"
        assert manus.attrs["coordinate_frame"] == "wrist_local"
        assert json.loads(_decode_strings(manus["provenance_json"][:])[1]) == {
            "glove_id": "547A010E",
            "native_index": 1,
        }
        np.testing.assert_array_equal(
            quest["capture_monotonic_ns"][:],
            [1_000_000_000, 1_250_000_000, 1_500_000_000, 1_750_000_000],
        )
        np.testing.assert_array_equal(manus["source_timestamp_ns"][:], [-1, -1, -1])
        assert not quest["joint_orientation_available"][:].any()
        assert np.isnan(quest["joint_orientations_xyzw"][:]).all()
        np.testing.assert_array_equal(
            manus["joint_orientation_available"][0], [True, True, True, False]
        )
        assert not bool(manus["joint_validity"][1, 3])

        wrist = root["wrist_streams/quest_wrist_right"]
        assert wrist.attrs["reference_frame"] == "quest_stage"
        assert wrist.attrs["anatomical_frame"] == "wrist_local"
        assert json.loads(_decode_strings(wrist["provenance_json"][:])[0]) == {
            "support_frame": 30
        }
        assert wrist["position"].shape == (2, 3)

        selected_output = root["retargeting_streams/manus_adaptive_sharpa"]
        shadow_output = root["retargeting_streams/quest_shadow_adaptive_sharpa"]
        np.testing.assert_allclose(
            selected_output["joint_positions"][:], [[0.1, 0.2], [0.2, 0.3]]
        )
        np.testing.assert_allclose(
            shadow_output["joint_positions"][:], [[0.3, 0.4], [0.4, 0.5]]
        )
        assert _decode_strings(selected_output["articulation_stream"][:]) == [
            "manus_right",
            "manus_right",
        ]

        alignment = root["action_alignment/demo_0"]
        assert alignment.attrs["actual_action_dataset"] == "/data/demo_0/action"
        assert _decode_strings(alignment["retargeting_stream"][:]) == [
            "manus_adaptive_sharpa",
            "manus_adaptive_sharpa",
        ]
        np.testing.assert_array_equal(alignment["retargeting_row"][:], [0, 1])
        np.testing.assert_allclose(
            alignment["synchronization_skew_seconds"][:], [0.02, 0.02]
        )


def test_action_alignment_may_reuse_one_selected_command_for_multiple_actions(tmp_path):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation, wrists, retargeting, selections = _fixture_streams()
    selections[1] = ActionHandSelection(
        articulation_stream="manus_right",
        articulation_row=0,
        wrist_stream="quest_wrist_right",
        wrist_row=0,
        retargeting_stream="manus_adaptive_sharpa",
        retargeting_row=0,
        synchronization_skew_seconds=0.02,
    )

    write_multi_source_hand_recording(
        path,
        articulation_streams=articulation,
        wrist_streams=wrists,
        retargeting_streams=retargeting,
        action_alignment_episodes=[selections],
    )

    with h5py.File(path, "r") as recording:
        alignment = recording[f"{HAND_TRACKING_GROUP}/action_alignment/demo_0"]
        np.testing.assert_array_equal(alignment["articulation_row"][:], [0, 0])
        np.testing.assert_array_equal(alignment["retargeting_row"][:], [0, 0])


def test_session_collector_deduplicates_concurrent_sources_and_returns_stable_rows(
    tmp_path,
):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation = _articulation(
        1.0,
        source="manus",
        schema="manus_raw_25",
        names=("wrist", "index_tip"),
        orientations=True,
        source_frame_id=40,
        source_timestamp_ns=1_000_000_000,
    )
    next_articulation = _articulation(
        2.0,
        source="manus",
        schema="manus_raw_25",
        names=("wrist", "index_tip"),
        orientations=True,
        offset=1.0,
        source_frame_id=41,
        source_timestamp_ns=2_000_000_000,
    )
    wrist = _wrist(1.02, source_frame_id=50)
    output = RetargetedCommandSample(
        command=_command(1.02, 0.5),
        retargeter="adaptive",
        articulation_stream="manus_right",
        articulation_row=0,
        wrist_stream="quest_wrist_right",
        wrist_row=0,
    )
    selection = ActionHandSelection(
        articulation_stream="manus_right",
        articulation_row=0,
        wrist_stream="quest_wrist_right",
        wrist_row=0,
        retargeting_stream="manus_adaptive_sharpa",
        retargeting_row=0,
        synchronization_skew_seconds=0.02,
    )
    session = HandTrackingRecordingSession()

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = tuple(
            executor.map(
                lambda _index: session.append_articulation("manus_right", articulation),
                range(64),
            )
        )
    assert rows == (0,) * 64
    assert session.append_articulation("manus_right", next_articulation) == 1
    assert session.append_wrist("quest_wrist_right", wrist) == 0
    assert session.append_wrist("quest_wrist_right", wrist) == 0
    assert session.append_retargeted("manus_adaptive_sharpa", output) == 0
    assert session.append_retargeted("manus_adaptive_sharpa", output) == 0
    session.set_stream_metadata("manus_right", {"calibration_id": "calibration-1"})
    assert session.begin_episode() == 0
    assert session.append_action_selection(selection) == 0
    assert session.append_action_selection(selection) == 1

    snapshot = session.writer_kwargs()
    assert len(snapshot["articulation_streams"]["manus_right"]) == 2
    assert len(snapshot["wrist_streams"]["quest_wrist_right"]) == 1
    assert len(snapshot["retargeting_streams"]["manus_adaptive_sharpa"]) == 1
    assert len(snapshot["action_alignment_episodes"][0]) == 2
    session.close()
    session.write(path)

    with pytest.raises(RuntimeError, match="session is closed"):
        session.append_wrist("quest_wrist_right", wrist)
    with h5py.File(path, "r") as recording:
        root = recording[HAND_TRACKING_GROUP]
        assert root["articulation_streams/manus_right/joint_positions"].shape == (
            2,
            2,
            3,
        )
        np.testing.assert_array_equal(
            root["action_alignment/demo_0/retargeting_row"][:], [0, 0]
        )


def test_invalid_alignment_is_rejected_before_replacing_existing_recording(tmp_path):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation, wrists, retargeting, selections = _fixture_streams()
    with h5py.File(path, "r+") as recording:
        existing = recording.create_group(HAND_TRACKING_GROUP)
        existing.attrs["sentinel"] = "preserve-on-validation-error"
    selections[1] = ActionHandSelection(
        articulation_stream="manus_right",
        articulation_row=99,
        wrist_stream="quest_wrist_right",
        wrist_row=1,
        retargeting_stream="manus_adaptive_sharpa",
        retargeting_row=1,
        synchronization_skew_seconds=0.02,
    )

    with pytest.raises(ValueError, match="outside stream 'manus_right'"):
        write_multi_source_hand_recording(
            path,
            articulation_streams=articulation,
            wrist_streams=wrists,
            retargeting_streams=retargeting,
            action_alignment_episodes=[selections],
        )

    with h5py.File(path, "r") as recording:
        assert (
            recording[HAND_TRACKING_GROUP].attrs["sentinel"]
            == "preserve-on-validation-error"
        )


def test_writer_rejects_episode_length_mismatch(tmp_path):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation, wrists, retargeting, selections = _fixture_streams()

    with pytest.raises(RuntimeError, match="has 1 steps; trajectory has 2"):
        write_multi_source_hand_recording(
            path,
            articulation_streams=articulation,
            wrist_streams=wrists,
            retargeting_streams=retargeting,
            action_alignment_episodes=[selections[:1]],
        )


def test_session_metadata_is_deeply_detached_from_callers_and_snapshots():
    session = HandTrackingRecordingSession()
    metadata = {"calibration": {"translation": [1.0, 2.0, 3.0]}}
    session.set_stream_metadata("manus_right", metadata)
    metadata["calibration"]["translation"][0] = 99.0

    first = session.writer_kwargs()
    assert first["stream_metadata"]["manus_right"]["calibration"]["translation"] == [
        1.0,
        2.0,
        3.0,
    ]
    first["stream_metadata"]["manus_right"]["calibration"]["translation"][1] = 88.0
    second = session.writer_kwargs()
    assert second["stream_metadata"]["manus_right"]["calibration"]["translation"] == [
        1.0,
        2.0,
        3.0,
    ]


def test_source_identity_collision_compares_receipt_frames_and_provenance():
    session = HandTrackingRecordingSession()
    articulation = _articulation(
        1.0,
        source="manus",
        schema="manus_raw_25",
        names=("wrist", "index_tip"),
        source_timestamp_ns=1_000_000_000,
        source_frame_id=1,
        provenance={"glove_id": "right"},
    )
    wrist = _wrist(
        1.0,
        source_frame_id=1,
        provenance={"tracker": "quest"},
    )
    assert session.append_articulation("manus_right", articulation) == 0
    assert session.append_wrist("quest_right", wrist) == 0

    with pytest.raises(ValueError, match="reused a source identity"):
        session.append_articulation(
            "manus_right",
            replace(
                articulation,
                receipt_timestamp=articulation.receipt_timestamp + 0.01,
            ),
        )
    with pytest.raises(ValueError, match="reused a source identity"):
        session.append_articulation(
            "manus_right",
            replace(articulation, coordinate_frame="different_wrist_basis"),
        )
    with pytest.raises(ValueError, match="reused a source identity"):
        session.append_wrist(
            "quest_right",
            replace(wrist, provenance={"tracker": "different"}),
        )


def test_write_failure_keeps_existing_group_and_cleans_pending_transaction(
    tmp_path, monkeypatch
):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation, wrists, retargeting, selections = _fixture_streams()
    with h5py.File(path, "r+") as recording:
        existing = recording.create_group(HAND_TRACKING_GROUP)
        existing.attrs["sentinel"] = "preserve-on-write-error"

    def fail_wrist_write(*_args, **_kwargs):
        raise OSError("simulated HDF5 wrist write failure")

    monkeypatch.setattr(recording_module, "_write_wrist_stream", fail_wrist_write)
    with pytest.raises(OSError, match="simulated HDF5 wrist write failure"):
        write_multi_source_hand_recording(
            path,
            articulation_streams=articulation,
            wrist_streams=wrists,
            retargeting_streams=retargeting,
            action_alignment_episodes=[selections],
        )

    with h5py.File(path, "r") as recording:
        assert (
            recording[HAND_TRACKING_GROUP].attrs["sentinel"]
            == "preserve-on-write-error"
        )
        assert not any(
            name.startswith(f"__{HAND_TRACKING_GROUP}_") for name in recording
        )


def test_invalid_metadata_is_rejected_before_replacing_existing_group(tmp_path):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation, wrists, retargeting, selections = _fixture_streams()
    with h5py.File(path, "r+") as recording:
        existing = recording.create_group(HAND_TRACKING_GROUP)
        existing.attrs["sentinel"] = "preserve-on-metadata-error"

    with pytest.raises(ValueError, match="Metadata for stream 'manus_right'"):
        write_multi_source_hand_recording(
            path,
            articulation_streams=articulation,
            wrist_streams=wrists,
            retargeting_streams=retargeting,
            action_alignment_episodes=[selections],
            stream_metadata={"manus_right": {"invalid": object()}},
        )

    with h5py.File(path, "r") as recording:
        assert (
            recording[HAND_TRACKING_GROUP].attrs["sentinel"]
            == "preserve-on-metadata-error"
        )


def test_writer_verifies_num_samples_against_the_actual_action_rows(tmp_path):
    path = tmp_path / "recording.hdf5"
    _write_trajectory(path)
    articulation, wrists, retargeting, selections = _fixture_streams()
    with h5py.File(path, "r+") as recording:
        recording["data/demo_0"].attrs["num_samples"] = 3
        existing = recording.create_group(HAND_TRACKING_GROUP)
        existing.attrs["sentinel"] = "preserve-on-trajectory-error"

    with pytest.raises(
        RuntimeError, match="action dataset has 2 rows; num_samples is 3"
    ):
        write_multi_source_hand_recording(
            path,
            articulation_streams=articulation,
            wrist_streams=wrists,
            retargeting_streams=retargeting,
            action_alignment_episodes=[selections],
        )

    with h5py.File(path, "r") as recording:
        assert (
            recording[HAND_TRACKING_GROUP].attrs["sentinel"]
            == "preserve-on-trajectory-error"
        )

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dex_teleop.omnigibson.hand_pose_recording import (
    HAND_POSE_GROUP,
    HAND_POSE_SCHEMA_VERSION,
    HumanHandPoseSample,
    write_hand_pose_episodes,
)
from dex_teleop.types import HandFrame, Handedness, MEDIAPIPE_JOINT_NAMES


def _frame(index: int) -> HandFrame:
    landmarks = np.arange(63, dtype=np.float64).reshape(21, 3) + index
    return HandFrame(
        timestamp=10.0 + index,
        receipt_timestamp=10.1 + index,
        source_timestamp_ns=1_000_000_000 + index,
        source_frame_id=100 + index,
        handedness=Handedness.RIGHT,
        joints=dict(zip(MEDIAPIPE_JOINT_NAMES, landmarks, strict=True)),
        wrist_position=np.array([1.0, 2.0, 3.0]) + index,
        wrist_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        source="hts",
        confidence=0.9,
    )


def _source_diagnostics(index: int):
    return SimpleNamespace(
        wrist_receipt_timestamp=20.0 + index,
        landmarks_receipt_timestamp=20.01 + index,
        raw_wrist_position_unity=np.array([4.0, 5.0, 6.0]) + index,
        raw_wrist_quaternion_xyzw=np.array([0.1, 0.2, 0.3, 0.9]),
        raw_landmarks_unity=np.arange(63, dtype=np.float64).reshape(21, 3) - index,
    )


def _write_trajectory_groups(path, lengths):
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        for episode_id, length in lengths:
            data.create_group(f"demo_{episode_id}").attrs["num_samples"] = length


def test_hand_poses_are_action_aligned_and_preserve_raw_hts_values(tmp_path):
    recording_path = tmp_path / "demo.hdf5"
    _write_trajectory_groups(recording_path, [(0, 2), (3, 0)])
    frames = [_frame(0), _frame(1)]
    samples = [
        HumanHandPoseSample.capture(frames[0], _source_diagnostics(0)),
        HumanHandPoseSample.capture(frames[1]),
    ]

    write_hand_pose_episodes(recording_path, [samples, []])

    with h5py.File(recording_path, "r") as recording:
        hand_poses = recording[HAND_POSE_GROUP]
        assert hand_poses.attrs["schema_version"] == HAND_POSE_SCHEMA_VERSION
        assert (
            tuple(json.loads(hand_poses.attrs["landmark_names"]))
            == MEDIAPIPE_JOINT_NAMES
        )
        episode = hand_poses["demo_0"]
        assert episode.attrs["source"] == "hts"
        assert episode.attrs["handedness"] == "right"
        np.testing.assert_array_equal(episode["source_frame_id"][:], [100, 101])
        np.testing.assert_array_equal(episode["timestamp_monotonic_ns"][:], [10_000_000_000, 11_000_000_000])
        np.testing.assert_array_equal(episode["receipt_monotonic_ns"][:], [10_100_000_000, 11_100_000_000])
        np.testing.assert_allclose(
            episode["wrist_position"][:], [frame.wrist_position for frame in frames]
        )
        np.testing.assert_allclose(
            episode["wrist_quaternion_xyzw"][:],
            [frame.wrist_quaternion_xyzw for frame in frames],
        )
        np.testing.assert_allclose(
            episode["landmarks"][:], [frame.mediapipe_landmarks() for frame in frames]
        )
        np.testing.assert_array_equal(episode["raw_available"][:], [True, False])
        np.testing.assert_allclose(
            episode["raw_wrist_position_unity"][0],
            _source_diagnostics(0).raw_wrist_position_unity,
        )
        np.testing.assert_allclose(
            episode["raw_wrist_quaternion_xyzw"][0],
            _source_diagnostics(0).raw_wrist_quaternion_xyzw,
        )
        np.testing.assert_allclose(
            episode["raw_landmarks_unity"][0],
            _source_diagnostics(0).raw_landmarks_unity,
        )
        assert np.isnan(episode["raw_wrist_position_unity"][1]).all()
        assert hand_poses["demo_3"]["landmarks"].shape == (0, 21, 3)


def test_hand_pose_writer_rejects_trajectory_length_mismatch(tmp_path):
    recording_path = tmp_path / "demo.hdf5"
    _write_trajectory_groups(recording_path, [(0, 2)])

    with pytest.raises(RuntimeError, match="has 1 steps; trajectory has 2"):
        write_hand_pose_episodes(
            recording_path, [[HumanHandPoseSample.capture(_frame(0))]]
        )

"""Action-aligned human hand poses stored alongside OmniGibson trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dex_teleop.types import HandFrame, MEDIAPIPE_JOINT_NAMES


HAND_POSE_GROUP = "human_hand_pose"
HAND_POSE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HumanHandPoseSample:
    """One pre-retargeting hand frame and any available source-native values."""

    frame: HandFrame
    wrist_receipt_timestamp: float | None = None
    landmarks_receipt_timestamp: float | None = None
    raw_wrist_position_unity: np.ndarray | None = None
    raw_wrist_quaternion_xyzw: np.ndarray | None = None
    raw_landmarks_unity: np.ndarray | None = None

    @classmethod
    def capture(
        cls, frame: HandFrame, source_diagnostics: Any | None = None
    ) -> "HumanHandPoseSample":
        """Copy a frame and optional HTS source-native diagnostics for durable recording."""

        if source_diagnostics is None:
            return cls(frame=frame)
        return cls(
            frame=frame,
            wrist_receipt_timestamp=float(source_diagnostics.wrist_receipt_timestamp),
            landmarks_receipt_timestamp=float(
                source_diagnostics.landmarks_receipt_timestamp
            ),
            raw_wrist_position_unity=np.asarray(
                source_diagnostics.raw_wrist_position_unity, dtype=np.float64
            )
            .reshape(3)
            .copy(),
            raw_wrist_quaternion_xyzw=np.asarray(
                source_diagnostics.raw_wrist_quaternion_xyzw, dtype=np.float64
            )
            .reshape(4)
            .copy(),
            raw_landmarks_unity=np.asarray(
                source_diagnostics.raw_landmarks_unity, dtype=np.float64
            )
            .reshape(len(MEDIAPIPE_JOINT_NAMES), 3)
            .copy(),
        )

    @property
    def raw_available(self) -> bool:
        return self.raw_wrist_position_unity is not None


def _episode_groups(group: h5py.Group) -> list[tuple[int, h5py.Group]]:
    episodes = []
    for name, episode in group.items():
        if not name.startswith("demo_") or not isinstance(episode, h5py.Group):
            continue
        try:
            episode_id = int(name.removeprefix("demo_"))
        except ValueError:
            continue
        episodes.append((episode_id, episode))
    return sorted(episodes)


def _stack_or_empty(values: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    return np.stack(values) if values else np.empty((0, *shape), dtype=np.float64)


def _raw_array(
    sample: HumanHandPoseSample, field: str, shape: tuple[int, ...]
) -> np.ndarray:
    value = getattr(sample, field)
    return np.full(shape, np.nan, dtype=np.float64) if value is None else value


def _single_value(samples: list[HumanHandPoseSample], field: str) -> str:
    values = {
        str(
            getattr(sample.frame, field).value
            if field == "handedness"
            else getattr(sample.frame, field)
        )
        for sample in samples
    }
    if len(values) > 1:
        raise ValueError(f"Hand-pose episode mixes {field} values: {sorted(values)}")
    return next(iter(values), "")


def write_hand_pose_episodes(
    input_path: str | Path, episodes: list[list[HumanHandPoseSample]]
) -> None:
    """Append action-aligned human wrist and landmark samples to a completed recording."""

    input_path = Path(input_path)
    with h5py.File(input_path, "r+") as recording:
        trajectory_episodes = _episode_groups(recording["data"])
        if len(episodes) != len(trajectory_episodes):
            raise RuntimeError(
                f"Hand-pose episode count {len(episodes)} does not match trajectory count {len(trajectory_episodes)}"
            )

        if HAND_POSE_GROUP in recording:
            del recording[HAND_POSE_GROUP]
        hand_poses = recording.create_group(HAND_POSE_GROUP)
        hand_poses.attrs["schema_version"] = HAND_POSE_SCHEMA_VERSION
        hand_poses.attrs["landmark_names"] = json.dumps(MEDIAPIPE_JOINT_NAMES)
        hand_poses.attrs["coordinate_frame"] = (
            "source-independent right-handed world frame"
        )
        hand_poses.attrs["raw_coordinate_frame"] = (
            "HTS Unity source frame; landmarks are wrist-local"
        )
        hand_poses.attrs["wrist_quaternion_order"] = "xyzw"
        hand_poses.attrs["missing_integer_sentinel"] = -1
        hand_poses.attrs["desktop_clock"] = "time.monotonic_ns"

        for samples, (episode_id, trajectory) in zip(episodes, trajectory_episodes):
            expected_steps = int(trajectory.attrs["num_samples"])
            if len(samples) != expected_steps:
                raise RuntimeError(
                    f"Hand-pose demo_{episode_id} has {len(samples)} steps; trajectory has {expected_steps}"
                )

            episode = hand_poses.create_group(f"demo_{episode_id}")
            episode.attrs["source"] = _single_value(samples, "source")
            episode.attrs["handedness"] = _single_value(samples, "handedness")
            episode.create_dataset(
                "timestamp",
                data=np.asarray([sample.frame.timestamp for sample in samples]),
            )
            episode.create_dataset(
                "timestamp_monotonic_ns",
                data=np.asarray([round(sample.frame.timestamp * 1e9) for sample in samples], dtype=np.int64),
            )
            episode.create_dataset(
                "receipt_timestamp",
                data=np.asarray([sample.frame.receipt_timestamp for sample in samples]),
            )
            episode.create_dataset(
                "receipt_monotonic_ns",
                data=np.asarray([round(sample.frame.receipt_timestamp * 1e9) for sample in samples], dtype=np.int64),
            )
            episode.create_dataset(
                "source_timestamp_ns",
                data=np.asarray(
                    [
                        -1
                        if sample.frame.source_timestamp_ns is None
                        else sample.frame.source_timestamp_ns
                        for sample in samples
                    ],
                    dtype=np.int64,
                ),
            )
            episode.create_dataset(
                "source_frame_id",
                data=np.asarray(
                    [
                        -1
                        if sample.frame.source_frame_id is None
                        else sample.frame.source_frame_id
                        for sample in samples
                    ],
                    dtype=np.int64,
                ),
            )
            episode.create_dataset(
                "confidence",
                data=np.asarray(
                    [
                        np.nan
                        if sample.frame.confidence is None
                        else sample.frame.confidence
                        for sample in samples
                    ]
                ),
            )
            episode.create_dataset(
                "wrist_position",
                data=_stack_or_empty(
                    [sample.frame.wrist_position for sample in samples], (3,)
                ),
            )
            episode.create_dataset(
                "wrist_quaternion_xyzw",
                data=_stack_or_empty(
                    [sample.frame.wrist_quaternion_xyzw for sample in samples], (4,)
                ),
            )
            episode.create_dataset(
                "landmarks",
                data=_stack_or_empty(
                    [sample.frame.mediapipe_landmarks() for sample in samples], (21, 3)
                ),
            )
            episode.create_dataset(
                "raw_available",
                data=np.asarray(
                    [sample.raw_available for sample in samples], dtype=np.bool_
                ),
            )
            episode.create_dataset(
                "wrist_receipt_timestamp",
                data=np.asarray(
                    [
                        np.nan
                        if sample.wrist_receipt_timestamp is None
                        else sample.wrist_receipt_timestamp
                        for sample in samples
                    ]
                ),
            )
            episode.create_dataset(
                "landmarks_receipt_timestamp",
                data=np.asarray(
                    [
                        np.nan
                        if sample.landmarks_receipt_timestamp is None
                        else sample.landmarks_receipt_timestamp
                        for sample in samples
                    ]
                ),
            )
            episode.create_dataset(
                "raw_wrist_position_unity",
                data=_stack_or_empty(
                    [
                        _raw_array(sample, "raw_wrist_position_unity", (3,))
                        for sample in samples
                    ],
                    (3,),
                ),
            )
            episode.create_dataset(
                "raw_wrist_quaternion_xyzw",
                data=_stack_or_empty(
                    [
                        _raw_array(sample, "raw_wrist_quaternion_xyzw", (4,))
                        for sample in samples
                    ],
                    (4,),
                ),
            )
            episode.create_dataset(
                "raw_landmarks_unity",
                data=_stack_or_empty(
                    [
                        _raw_array(sample, "raw_landmarks_unity", (21, 3))
                        for sample in samples
                    ],
                    (21, 3),
                ),
            )
        recording.flush()

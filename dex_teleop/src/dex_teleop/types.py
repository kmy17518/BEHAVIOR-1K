"""Source-independent hand-tracking and retargeting value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np


MEDIAPIPE_JOINT_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class HandFrame:
    """One source-independent hand sample in a right-handed world frame.

    ``timestamp`` is the sample time in the desktop monotonic clock domain.
    Network sources may additionally retain their raw source timestamp and the
    desktop time at which the complete sample became available.
    """

    timestamp: float
    handedness: Handedness
    joints: Mapping[str, np.ndarray]
    wrist_position: np.ndarray
    wrist_quaternion_xyzw: np.ndarray
    source: str
    confidence: float | None = None
    receipt_timestamp: float | None = None
    source_timestamp_ns: int | None = None
    source_frame_id: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("HandFrame timestamp must be finite")
        receipt_timestamp = self.timestamp if self.receipt_timestamp is None else self.receipt_timestamp
        if not np.isfinite(receipt_timestamp):
            raise ValueError("HandFrame receipt timestamp must be finite")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns < 0:
            raise ValueError("HandFrame source timestamp must be non-negative")
        if self.source_frame_id is not None and self.source_frame_id < 0:
            raise ValueError("HandFrame source frame ID must be non-negative")
        joint_copy = {
            name: np.asarray(position, dtype=np.float64).reshape(3).copy()
            for name, position in self.joints.items()
        }
        missing = set(MEDIAPIPE_JOINT_NAMES).difference(joint_copy)
        if missing:
            raise ValueError(f"HandFrame is missing required joints: {sorted(missing)}")
        if not all(np.isfinite(position).all() for position in joint_copy.values()):
            raise ValueError("HandFrame joints contain a non-finite value")
        wrist_position = np.asarray(self.wrist_position, dtype=np.float64).reshape(3).copy()
        wrist_quaternion = np.asarray(self.wrist_quaternion_xyzw, dtype=np.float64).reshape(4).copy()
        if not np.isfinite(wrist_position).all() or not np.isfinite(wrist_quaternion).all():
            raise ValueError("HandFrame wrist pose contains a non-finite value")
        norm = float(np.linalg.norm(wrist_quaternion))
        if norm <= 0.0:
            raise ValueError("wrist_quaternion_xyzw must have non-zero norm")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("HandFrame confidence must be in [0, 1]")
        object.__setattr__(self, "joints", MappingProxyType(joint_copy))
        object.__setattr__(self, "wrist_position", wrist_position)
        object.__setattr__(self, "wrist_quaternion_xyzw", wrist_quaternion / norm)
        object.__setattr__(self, "receipt_timestamp", float(receipt_timestamp))

    def mediapipe_landmarks(self) -> np.ndarray:
        """Return the required landmarks in the standard MediaPipe order."""

        return np.stack([self.joints[name] for name in MEDIAPIPE_JOINT_NAMES])


@dataclass(frozen=True)
class RetargetedHandCommand:
    """A robot-hand command whose joint names make ordering explicit."""

    timestamp: float
    handedness: Handedness
    hand_model: str
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.joint_positions, dtype=np.float64).reshape(-1).copy()
        if len(self.joint_names) != len(positions):
            raise ValueError(
                f"Expected {len(self.joint_names)} positions for {self.hand_model}, got {len(positions)}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("RetargetedHandCommand contains a non-finite joint position")
        object.__setattr__(self, "joint_positions", positions)

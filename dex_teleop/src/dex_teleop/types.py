"""Source-independent hand-tracking and retargeting value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

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

# Joint order from ``XR_EXT_hand_tracking``.  This is a data-schema constant;
# using it does not require an OpenXR runtime.
OPENXR_HAND_JOINT_NAMES = (
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)

# Both joint vectors and wrist-pose orientations must use the same anatomical
# wrist basis before they can be composed.  Sources that follow the OpenXR
# wrist-joint convention use this identifier; a fuser rejects different frame
# identifiers unless it is given an explicit rigid transform.
OPENXR_ANATOMICAL_WRIST_FRAME = "openxr_anatomical_wrist"


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"


def _validate_sample_metadata(
    *,
    timestamp: float,
    receipt_timestamp: float | None,
    source_timestamp_ns: int | None,
    source_frame_id: int | None,
    confidence: float | None,
    source: str,
    sample_name: str,
) -> float:
    if not np.isfinite(timestamp):
        raise ValueError(f"{sample_name} timestamp must be finite")
    receipt = timestamp if receipt_timestamp is None else receipt_timestamp
    if not np.isfinite(receipt):
        raise ValueError(f"{sample_name} receipt timestamp must be finite")
    if source_timestamp_ns is not None and source_timestamp_ns < 0:
        raise ValueError(f"{sample_name} source timestamp must be non-negative")
    if source_frame_id is not None and source_frame_id < 0:
        raise ValueError(f"{sample_name} source frame ID must be non-negative")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{sample_name} confidence must be in [0, 1]")
    if not source or not source.strip():
        raise ValueError(f"{sample_name} source must be non-empty")
    return float(receipt)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _frozen_provenance(value: Mapping[str, Any] | None, sample_name: str) -> Mapping[str, Any]:
    document = {} if value is None else dict(value)
    for key, item in document.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{sample_name} provenance keys must be non-empty strings")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError(f"{sample_name} provenance values must be JSON scalars")
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError(f"{sample_name} provenance floats must be finite")
    # Serialize once as an additional guard against exotic int subclasses.
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{sample_name} provenance must be finite JSON data") from error
    return MappingProxyType(document)


@dataclass(frozen=True)
class HandArticulationSample:
    """One named, wrist-local hand-articulation sample.

    The schema may be OpenXR-26, MediaPipe-21, a vendor-native skeleton, or a
    future named schema.  Source adapters retain the richest representation
    they have; a retargeter-specific conversion happens downstream.

    Missing entries in ``joint_validity`` default to valid.  An invalid joint
    may carry non-finite placeholder values, while valid positions and
    orientations must be finite.  Orientations are optional and may be sparse.
    """

    timestamp: float
    handedness: Handedness
    joint_positions: Mapping[str, np.ndarray]
    source: str
    schema: str = "named"
    coordinate_frame: str = "wrist_local"
    joint_orientations_xyzw: Mapping[str, np.ndarray] | None = None
    joint_validity: Mapping[str, bool] | None = None
    confidence: float | None = None
    receipt_timestamp: float | None = None
    source_timestamp_ns: int | None = None
    source_frame_id: int | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        receipt = _validate_sample_metadata(
            timestamp=self.timestamp,
            receipt_timestamp=self.receipt_timestamp,
            source_timestamp_ns=self.source_timestamp_ns,
            source_frame_id=self.source_frame_id,
            confidence=self.confidence,
            source=self.source,
            sample_name="HandArticulationSample",
        )
        if not self.schema or not self.schema.strip():
            raise ValueError("HandArticulationSample schema must be non-empty")
        if not self.coordinate_frame or not self.coordinate_frame.strip():
            raise ValueError("HandArticulationSample coordinate frame must be non-empty")
        if not self.joint_positions:
            raise ValueError("HandArticulationSample must contain at least one joint")

        validity_input = {} if self.joint_validity is None else dict(self.joint_validity)
        unknown_validity = set(validity_input).difference(self.joint_positions)
        if unknown_validity:
            raise ValueError(
                "HandArticulationSample validity contains unknown joints: "
                f"{sorted(unknown_validity)}"
            )
        validity = {
            name: bool(validity_input.get(name, True))
            for name in self.joint_positions
        }

        positions = {}
        for name, value in self.joint_positions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("HandArticulationSample joint names must be non-empty strings")
            position = np.asarray(value, dtype=np.float64).reshape(3).copy()
            if validity[name] and not np.isfinite(position).all():
                raise ValueError(f"HandArticulationSample joint {name!r} has a non-finite valid position")
            positions[name] = _readonly(position)

        orientation_input = (
            {} if self.joint_orientations_xyzw is None else dict(self.joint_orientations_xyzw)
        )
        unknown_orientations = set(orientation_input).difference(positions)
        if unknown_orientations:
            raise ValueError(
                "HandArticulationSample orientations contain unknown joints: "
                f"{sorted(unknown_orientations)}"
            )
        orientations = {}
        for name, value in orientation_input.items():
            quaternion = np.asarray(value, dtype=np.float64).reshape(4).copy()
            if validity[name]:
                if not np.isfinite(quaternion).all():
                    raise ValueError(
                        f"HandArticulationSample joint {name!r} has a non-finite valid orientation"
                    )
                norm = float(np.linalg.norm(quaternion))
                if norm <= 0.0:
                    raise ValueError(
                        f"HandArticulationSample joint {name!r} has a zero-norm orientation"
                    )
                quaternion /= norm
            orientations[name] = _readonly(quaternion)

        object.__setattr__(self, "handedness", Handedness(self.handedness))
        object.__setattr__(self, "joint_positions", MappingProxyType(positions))
        object.__setattr__(self, "joint_orientations_xyzw", MappingProxyType(orientations))
        object.__setattr__(self, "joint_validity", MappingProxyType(validity))
        object.__setattr__(
            self,
            "provenance",
            _frozen_provenance(self.provenance, "HandArticulationSample"),
        )
        object.__setattr__(self, "receipt_timestamp", receipt)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.joint_positions)

    def positions(self, joint_names: Iterable[str] | None = None) -> np.ndarray:
        """Return positions in the requested order, or the stored named order."""

        names = self.joint_names if joint_names is None else tuple(joint_names)
        return np.stack([self.joint_positions[name] for name in names])

    def orientations_xyzw(self, joint_names: Iterable[str] | None = None) -> np.ndarray:
        """Return orientations in order, using NaNs when a joint has no orientation."""

        names = self.joint_names if joint_names is None else tuple(joint_names)
        missing = np.full(4, np.nan, dtype=np.float64)
        return np.stack([self.joint_orientations_xyzw.get(name, missing) for name in names])

    def validity(self, joint_names: Iterable[str] | None = None) -> np.ndarray:
        """Return per-joint validity in the requested order."""

        names = self.joint_names if joint_names is None else tuple(joint_names)
        return np.asarray([self.joint_validity[name] for name in names], dtype=np.bool_)


@dataclass(frozen=True)
class WristPoseSample:
    """One tracked anatomical-wrist pose in a named reference frame."""

    timestamp: float
    handedness: Handedness
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    source: str
    reference_frame: str = "tracking"
    anatomical_frame: str = "wrist_local"
    confidence: float | None = None
    receipt_timestamp: float | None = None
    source_timestamp_ns: int | None = None
    source_frame_id: int | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        receipt = _validate_sample_metadata(
            timestamp=self.timestamp,
            receipt_timestamp=self.receipt_timestamp,
            source_timestamp_ns=self.source_timestamp_ns,
            source_frame_id=self.source_frame_id,
            confidence=self.confidence,
            source=self.source,
            sample_name="WristPoseSample",
        )
        if not self.reference_frame or not self.reference_frame.strip():
            raise ValueError("WristPoseSample reference frame must be non-empty")
        if not self.anatomical_frame or not self.anatomical_frame.strip():
            raise ValueError("WristPoseSample anatomical frame must be non-empty")
        position = np.asarray(self.position, dtype=np.float64).reshape(3).copy()
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64).reshape(4).copy()
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError("WristPoseSample pose contains a non-finite value")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError("WristPoseSample quaternion must have non-zero norm")
        quaternion /= norm
        object.__setattr__(self, "handedness", Handedness(self.handedness))
        object.__setattr__(self, "position", _readonly(position))
        object.__setattr__(self, "quaternion_xyzw", _readonly(quaternion))
        object.__setattr__(
            self,
            "provenance",
            _frozen_provenance(self.provenance, "WristPoseSample"),
        )
        object.__setattr__(self, "receipt_timestamp", receipt)


@dataclass(frozen=True)
class FusedHandObservation:
    """A synchronized articulation and wrist pose with both provenances intact."""

    articulation: HandArticulationSample
    wrist: WristPoseSample
    synchronization_skew_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.articulation.handedness != self.wrist.handedness:
            raise ValueError(
                "Cannot fuse articulation and wrist samples with different handedness"
            )
        if self.articulation.coordinate_frame != self.wrist.anatomical_frame:
            raise ValueError(
                "Cannot fuse articulation and wrist samples with different anatomical frames"
            )
        if (
            not np.isfinite(self.synchronization_skew_seconds)
            or self.synchronization_skew_seconds < 0.0
        ):
            raise ValueError("Synchronization skew must be finite and non-negative")

    @property
    def timestamp(self) -> float:
        """Control timestamp; articulation is the trigger sample."""

        return self.articulation.timestamp

    @property
    def receipt_timestamp(self) -> float:
        """Time at which both components were available on the host."""

        return max(self.articulation.receipt_timestamp, self.wrist.receipt_timestamp)

    @property
    def handedness(self) -> Handedness:
        return self.articulation.handedness


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

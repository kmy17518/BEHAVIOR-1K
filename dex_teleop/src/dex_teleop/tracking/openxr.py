"""Named OpenXR-26 / MediaPipe-21 schema conversion utilities.

OpenXR is used here only as a canonical joint vocabulary.  None of these
functions imports or starts an OpenXR runtime.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dex_teleop.types import (
    FusedHandObservation,
    HandArticulationSample,
    HandFrame,
    MEDIAPIPE_JOINT_NAMES,
    OPENXR_HAND_JOINT_NAMES,
    WristPoseSample,
)


MEDIAPIPE_TO_OPENXR_JOINTS: Mapping[str, str] = {
    "wrist": "wrist",
    "thumb_cmc": "thumb_metacarpal",
    "thumb_mcp": "thumb_proximal",
    "thumb_ip": "thumb_distal",
    "thumb_tip": "thumb_tip",
    "index_mcp": "index_proximal",
    "index_pip": "index_intermediate",
    "index_dip": "index_distal",
    "index_tip": "index_tip",
    "middle_mcp": "middle_proximal",
    "middle_pip": "middle_intermediate",
    "middle_dip": "middle_distal",
    "middle_tip": "middle_tip",
    "ring_mcp": "ring_proximal",
    "ring_pip": "ring_intermediate",
    "ring_dip": "ring_distal",
    "ring_tip": "ring_tip",
    "pinky_mcp": "little_proximal",
    "pinky_pip": "little_intermediate",
    "pinky_dip": "little_distal",
    "pinky_tip": "little_tip",
}
OPENXR_TO_MEDIAPIPE_JOINTS: Mapping[str, str] = {
    openxr: mediapipe for mediapipe, openxr in MEDIAPIPE_TO_OPENXR_JOINTS.items()
}


def _converted_sample(
    sample: HandArticulationSample,
    *,
    positions: Mapping[str, np.ndarray],
    schema: str,
    orientations: Mapping[str, np.ndarray] | None = None,
    validity: Mapping[str, bool] | None = None,
) -> HandArticulationSample:
    return HandArticulationSample(
        timestamp=sample.timestamp,
        receipt_timestamp=sample.receipt_timestamp,
        source_timestamp_ns=sample.source_timestamp_ns,
        source_frame_id=sample.source_frame_id,
        handedness=sample.handedness,
        joint_positions=positions,
        joint_orientations_xyzw=orientations,
        joint_validity=validity,
        source=sample.source,
        schema=schema,
        coordinate_frame=sample.coordinate_frame,
        confidence=sample.confidence,
        provenance=sample.provenance,
    )


def openxr26_to_mediapipe21(sample: HandArticulationSample) -> HandArticulationSample:
    """Select the MediaPipe-21 equivalents from a named OpenXR-26 sample."""

    required = tuple(MEDIAPIPE_TO_OPENXR_JOINTS.values())
    missing = set(required).difference(sample.joint_positions)
    if missing:
        raise ValueError(f"OpenXR articulation is missing required joints: {sorted(missing)}")

    positions = {
        mediapipe: sample.joint_positions[openxr]
        for mediapipe, openxr in MEDIAPIPE_TO_OPENXR_JOINTS.items()
    }
    orientations = {
        mediapipe: sample.joint_orientations_xyzw[openxr]
        for mediapipe, openxr in MEDIAPIPE_TO_OPENXR_JOINTS.items()
        if openxr in sample.joint_orientations_xyzw
    }
    validity = {
        mediapipe: sample.joint_validity[openxr]
        for mediapipe, openxr in MEDIAPIPE_TO_OPENXR_JOINTS.items()
    }
    return _converted_sample(
        sample,
        positions=positions,
        orientations=orientations,
        validity=validity,
        schema="mediapipe21",
    )


def mediapipe21_to_openxr26(sample: HandArticulationSample) -> HandArticulationSample:
    """Expand a MediaPipe-21 sample into the fixed OpenXR-26 vocabulary.

    MediaPipe has no palm joint or non-thumb metacarpal joints.  Their positions
    are interpolated so fixed-shape consumers receive 26 rows, but their
    validity flags are false and no orientations are invented.
    """

    missing = set(MEDIAPIPE_JOINT_NAMES).difference(sample.joint_positions)
    if missing:
        raise ValueError(f"MediaPipe articulation is missing required joints: {sorted(missing)}")

    reverse = OPENXR_TO_MEDIAPIPE_JOINTS
    positions: dict[str, np.ndarray] = {
        openxr: sample.joint_positions[mediapipe]
        for openxr, mediapipe in reverse.items()
    }
    validity: dict[str, bool] = {
        openxr: sample.joint_validity[mediapipe]
        for openxr, mediapipe in reverse.items()
    }
    orientations = {
        openxr: sample.joint_orientations_xyzw[mediapipe]
        for openxr, mediapipe in reverse.items()
        if mediapipe in sample.joint_orientations_xyzw
    }

    wrist = sample.joint_positions["wrist"]
    proximal_names = ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")
    positions["palm"] = np.mean(
        np.stack([wrist, *(sample.joint_positions[name] for name in proximal_names)]),
        axis=0,
    )
    validity["palm"] = False
    for finger, mediapipe_name in zip(
        ("index", "middle", "ring", "little"), proximal_names, strict=True
    ):
        positions[f"{finger}_metacarpal"] = 0.5 * (
            wrist + sample.joint_positions[mediapipe_name]
        )
        validity[f"{finger}_metacarpal"] = False

    ordered_positions = {name: positions[name] for name in OPENXR_HAND_JOINT_NAMES}
    ordered_orientations = {
        name: orientations[name] for name in OPENXR_HAND_JOINT_NAMES if name in orientations
    }
    ordered_validity = {name: validity[name] for name in OPENXR_HAND_JOINT_NAMES}
    return _converted_sample(
        sample,
        positions=ordered_positions,
        orientations=ordered_orientations,
        validity=ordered_validity,
        schema="openxr26",
    )


def articulation_to_mediapipe21(sample: HandArticulationSample) -> HandArticulationSample:
    """Normalize any supported named articulation to MediaPipe-21."""

    if set(MEDIAPIPE_JOINT_NAMES).issubset(sample.joint_positions):
        positions = {name: sample.joint_positions[name] for name in MEDIAPIPE_JOINT_NAMES}
        orientations = {
            name: sample.joint_orientations_xyzw[name]
            for name in MEDIAPIPE_JOINT_NAMES
            if name in sample.joint_orientations_xyzw
        }
        validity = {name: sample.joint_validity[name] for name in MEDIAPIPE_JOINT_NAMES}
        return _converted_sample(
            sample,
            positions=positions,
            orientations=orientations,
            validity=validity,
            schema="mediapipe21",
        )
    return openxr26_to_mediapipe21(sample)


def _rotate(points: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    xyz = quaternion_xyzw[:3]
    w = quaternion_xyzw[3]
    cross = 2.0 * np.cross(xyz, points)
    return points + w * cross + np.cross(xyz, cross)


def hand_frame_to_articulation(frame: HandFrame) -> HandArticulationSample:
    """Recover a wrist-local MediaPipe-21 sample from a legacy world-frame value."""

    world_positions = frame.mediapipe_landmarks() - frame.wrist_position
    inverse_quaternion = frame.wrist_quaternion_xyzw * np.array([-1.0, -1.0, -1.0, 1.0])
    local_positions = _rotate(world_positions, inverse_quaternion)
    return HandArticulationSample(
        timestamp=frame.timestamp,
        receipt_timestamp=frame.receipt_timestamp,
        source_timestamp_ns=frame.source_timestamp_ns,
        source_frame_id=frame.source_frame_id,
        handedness=frame.handedness,
        joint_positions=dict(zip(MEDIAPIPE_JOINT_NAMES, local_positions, strict=True)),
        source=frame.source,
        schema="mediapipe21",
        coordinate_frame="wrist_local",
        confidence=frame.confidence,
    )


def hand_frame_to_wrist(frame: HandFrame, reference_frame: str = "tracking") -> WristPoseSample:
    """Extract the wrist component from a legacy frame."""

    return WristPoseSample(
        timestamp=frame.timestamp,
        receipt_timestamp=frame.receipt_timestamp,
        source_timestamp_ns=frame.source_timestamp_ns,
        source_frame_id=frame.source_frame_id,
        handedness=frame.handedness,
        position=frame.wrist_position,
        quaternion_xyzw=frame.wrist_quaternion_xyzw,
        source=frame.source,
        reference_frame=reference_frame,
        anatomical_frame="wrist_local",
        confidence=frame.confidence,
    )


def observation_to_hand_frame(observation: FusedHandObservation) -> HandFrame:
    """Build the unchanged legacy world-frame API from one fused observation."""

    articulation = articulation_to_mediapipe21(observation.articulation)
    invalid = [name for name in MEDIAPIPE_JOINT_NAMES if not articulation.joint_validity[name]]
    if invalid:
        raise ValueError(f"Cannot build HandFrame from invalid MediaPipe joints: {invalid}")
    local_positions = articulation.positions(MEDIAPIPE_JOINT_NAMES)
    world_positions = (
        _rotate(local_positions, observation.wrist.quaternion_xyzw)
        + observation.wrist.position
    )
    confidences = tuple(
        value
        for value in (articulation.confidence, observation.wrist.confidence)
        if value is not None
    )
    source = articulation.source
    if observation.wrist.source != source:
        source = f"{source}+{observation.wrist.source}"
    return HandFrame(
        timestamp=articulation.timestamp,
        receipt_timestamp=observation.receipt_timestamp,
        source_timestamp_ns=articulation.source_timestamp_ns,
        source_frame_id=articulation.source_frame_id,
        handedness=articulation.handedness,
        joints=dict(zip(MEDIAPIPE_JOINT_NAMES, world_positions, strict=True)),
        wrist_position=observation.wrist.position,
        wrist_quaternion_xyzw=observation.wrist.quaternion_xyzw,
        source=source,
        confidence=min(confidences) if confidences else None,
    )


# Explicit alias for discoverability at call sites that traffic in fused values.
fused_observation_to_hand_frame = observation_to_hand_frame

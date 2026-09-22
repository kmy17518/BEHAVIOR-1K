"""Hand-tracking source interfaces and implementations."""

from dex_teleop.tracking.base import HandTrackingSource, SourceUnavailableError
from dex_teleop.tracking.fusion import ArticulationFrameTransform, HandObservationFuser
from dex_teleop.tracking.hts import HTSSource
from dex_teleop.tracking.manus import (
    ManusIntegratedSource,
    ManusProtocolError,
    ManusSource,
    build_manus_bridge,
    discover_manus_sdk,
)
from dex_teleop.tracking.manus_calibration import (
    MANUS_CORE_WORLD_FRAME,
    ManusCoreCalibration,
    ManusWristCalibration,
)
from dex_teleop.tracking.multimodal import (
    DrainingHandArticulationSource,
    DrainingMultimodalHandTrackingSource,
    DrainingWristPoseSource,
    HandArticulationSource,
    HandTrackingSampleBatch,
    MultimodalHandTrackingSource,
    WristPoseSource,
)
from dex_teleop.tracking.openxr import (
    articulation_to_mediapipe21,
    hand_frame_to_articulation,
    hand_frame_to_wrist,
    mediapipe21_to_openxr26,
    observation_to_hand_frame,
    openxr26_to_mediapipe21,
)
from dex_teleop.tracking.vive import RigidTransform, ViveCalibration, ViveWristSource

__all__ = [
    "HTSSource",
    "ArticulationFrameTransform",
    "DrainingHandArticulationSource",
    "DrainingMultimodalHandTrackingSource",
    "DrainingWristPoseSource",
    "HandArticulationSource",
    "HandObservationFuser",
    "HandTrackingSampleBatch",
    "HandTrackingSource",
    "ManusIntegratedSource",
    "ManusSource",
    "ManusCoreCalibration",
    "ManusWristCalibration",
    "MANUS_CORE_WORLD_FRAME",
    "ManusProtocolError",
    "MultimodalHandTrackingSource",
    "SourceUnavailableError",
    "RigidTransform",
    "ViveCalibration",
    "ViveWristSource",
    "WristPoseSource",
    "articulation_to_mediapipe21",
    "build_manus_bridge",
    "discover_manus_sdk",
    "hand_frame_to_articulation",
    "hand_frame_to_wrist",
    "mediapipe21_to_openxr26",
    "observation_to_hand_frame",
    "openxr26_to_mediapipe21",
]

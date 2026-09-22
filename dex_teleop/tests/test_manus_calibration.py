import numpy as np
import pytest

from dex_teleop.tracking.manus_calibration import (
    MANUS_CORE_WORLD_FRAME,
    ManusCoreCalibration,
)
from dex_teleop.tracking.transforms import RigidTransform


def _document():
    return {
        "schema_version": 1,
        "core_world_frame": MANUS_CORE_WORLD_FRAME,
        "reference_frame": "robot_world",
        "reference_from_core_world": {
            "translation": [1.0, 2.0, 3.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "wrists": [
            {
                "handedness": "right",
                "source_pose": "core_baked_skeleton_wrist",
                "core_tracker_offset_applied": True,
                "tracker_id": "ultimate-right",
                "tracker_offset_preset": "right-wrist",
                "tracker_to_anatomical_wrist": None,
                "skeleton_wrist_to_anatomical_wrist": {
                    "translation": [0.0, 0.1, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }


def test_manus_core_calibration_round_trips_and_applies_transform_chain():
    calibration = ManusCoreCalibration.from_mapping(_document())
    result = calibration.apply(
        "right",
        RigidTransform(
            translation=np.array([0.5, 0.0, 0.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        ),
    )

    np.testing.assert_allclose(result.translation, [1.5, 2.1, 3.0])
    assert calibration.as_mapping() == _document()


def test_manus_core_calibration_requires_boolean_core_offset_declaration():
    document = _document()
    document["wrists"][0]["core_tracker_offset_applied"] = "true"

    with pytest.raises(ValueError, match="must be boolean"):
        ManusCoreCalibration.from_mapping(document)


def test_manus_core_calibration_refuses_unapplied_tracker_mount_offset():
    document = _document()
    document["wrists"][0]["core_tracker_offset_applied"] = False

    with pytest.raises(ValueError, match="configure the Ultimate preset"):
        ManusCoreCalibration.from_mapping(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tracker_type", "controller"),
        ("tracker_system", "openxr"),
        ("tracker_user_id", None),
    ),
)
def test_manus_calibration_cannot_define_or_weaken_tracker_policy(field, value):
    document = _document()
    document["wrists"][0][field] = value

    with pytest.raises(
        ValueError, match="identity policy does not belong in geometric calibration"
    ):
        ManusCoreCalibration.from_mapping(document)

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dex_teleop.tracking.vive import RigidTransform, ViveCalibration, compose_transforms
from dex_teleop.tracking.vive import VIVE_TRACKER_POSE_FRAME
from dex_teleop.tracking.vive_calibration import (
    ViveCalibrationCapture,
    ViveCalibrationPair,
    capture_as_mapping,
    estimate_vive_calibration,
)
from dex_teleop.types import Handedness


def _transform(translation, rotation_vector) -> RigidTransform:
    return RigidTransform(translation, Rotation.from_rotvec(rotation_vector).as_quat())


TRUE_REFERENCE_FROM_LIGHTHOUSE = _transform([0.72, -0.34, 0.41], [0.35, -0.18, 0.11])
TRUE_TRACKER_FROM_WRIST = _transform([0.035, -0.026, 0.092], [-0.24, 0.17, 0.31])


def _capture(
    *, sample_count=12, seed=7, noisy=False, outlier=False
) -> ViveCalibrationCapture:
    random = np.random.default_rng(seed)
    pairs = []
    for index in range(sample_count):
        lighthouse_from_tracker = RigidTransform(
            random.uniform(-0.6, 0.6, size=3),
            Rotation.random(random_state=random).as_quat(),
        )
        reference_from_wrist = compose_transforms(
            compose_transforms(TRUE_REFERENCE_FROM_LIGHTHOUSE, lighthouse_from_tracker),
            TRUE_TRACKER_FROM_WRIST,
        )
        if noisy:
            tracker_noise = _transform(
                random.normal(0.0, 0.001, size=3),
                random.normal(0.0, np.deg2rad(0.15), size=3),
            )
            wrist_noise = _transform(
                random.normal(0.0, 0.0015, size=3),
                random.normal(0.0, np.deg2rad(0.2), size=3),
            )
            lighthouse_from_tracker = compose_transforms(
                lighthouse_from_tracker, tracker_noise
            )
            reference_from_wrist = compose_transforms(reference_from_wrist, wrist_noise)
        if outlier and index == sample_count - 1:
            reference_from_wrist = compose_transforms(
                reference_from_wrist,
                _transform([0.035, -0.025, 0.02], [0.08, -0.06, 0.05]),
            )
        pairs.append(
            ViveCalibrationPair(
                reference_timestamp=index * 0.1,
                tracker_timestamp=index * 0.1 + 0.003,
                reference_from_wrist=reference_from_wrist,
                lighthouse_from_tracker=lighthouse_from_tracker,
            )
        )
    return ViveCalibrationCapture(
        reference_frame="quest_tracking",
        tracker_serial="LHR-test-right",
        handedness=Handedness.RIGHT,
        pairs=tuple(pairs),
    )


def _rotation_error(first: RigidTransform, second: RigidTransform) -> float:
    first_rotation = Rotation.from_quat(first.quaternion_xyzw)
    second_rotation = Rotation.from_quat(second.quaternion_xyzw)
    return float((first_rotation.inv() * second_rotation).magnitude())


def test_exact_offline_calibration_recovers_both_transforms(tmp_path):
    capture = _capture()
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture_as_mapping(capture)), encoding="utf-8")

    result = estimate_vive_calibration(ViveCalibrationCapture.load(capture_path))

    np.testing.assert_allclose(
        result.calibration.reference_from_lighthouse.translation,
        TRUE_REFERENCE_FROM_LIGHTHOUSE.translation,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.calibration.mounts[0].tracker_to_wrist.translation,
        TRUE_TRACKER_FROM_WRIST.translation,
        atol=1e-9,
    )
    assert (
        _rotation_error(
            result.calibration.reference_from_lighthouse, TRUE_REFERENCE_FROM_LIGHTHOUSE
        )
        < 1e-9
    )
    assert (
        _rotation_error(
            result.calibration.mounts[0].tracker_to_wrist, TRUE_TRACKER_FROM_WRIST
        )
        < 1e-9
    )
    assert result.report.translation_rmse_m < 1e-10
    assert result.report.rotation_rmse_deg < 1e-8
    assert result.report.timestamp_skew_max_ms == pytest.approx(3.0)

    output_path = tmp_path / "vive.json"
    output_path.write_text(
        json.dumps(result.calibration.as_mapping()), encoding="utf-8"
    )
    assert ViveCalibration.load(output_path).mounts[0].serial == "LHR-test-right"


def test_robust_calibration_handles_sensor_noise_and_one_outlier():
    result = estimate_vive_calibration(
        _capture(sample_count=18, noisy=True, outlier=True)
    )

    assert (
        np.linalg.norm(
            result.calibration.reference_from_lighthouse.translation
            - TRUE_REFERENCE_FROM_LIGHTHOUSE.translation
        )
        < 0.01
    )
    assert (
        np.linalg.norm(
            result.calibration.mounts[0].tracker_to_wrist.translation
            - TRUE_TRACKER_FROM_WRIST.translation
        )
        < 0.01
    )
    assert _rotation_error(
        result.calibration.reference_from_lighthouse, TRUE_REFERENCE_FROM_LIGHTHOUSE
    ) < np.deg2rad(1.0)
    assert _rotation_error(
        result.calibration.mounts[0].tracker_to_wrist, TRUE_TRACKER_FROM_WRIST
    ) < np.deg2rad(1.0)
    assert result.report.translation_rmse_m < 0.015
    assert result.report.rotation_rmse_deg < 2.0


def test_degenerate_single_axis_motion_is_rejected():
    pairs = []
    for index in range(8):
        lighthouse_from_tracker = _transform(
            [index * 0.05, 0.0, 0.0], [0.0, 0.0, index * 0.15]
        )
        reference_from_wrist = compose_transforms(
            compose_transforms(TRUE_REFERENCE_FROM_LIGHTHOUSE, lighthouse_from_tracker),
            TRUE_TRACKER_FROM_WRIST,
        )
        pairs.append(
            ViveCalibrationPair(
                index * 0.1,
                index * 0.1 + 0.001,
                reference_from_wrist,
                lighthouse_from_tracker,
            )
        )
    capture = ViveCalibrationCapture(
        "quest_tracking", "LHR-degenerate", Handedness.RIGHT, tuple(pairs)
    )

    with pytest.raises(ValueError, match="non-parallel axes"):
        estimate_vive_calibration(capture)


def test_excessive_pair_time_skew_is_rejected():
    capture = _capture()

    with pytest.raises(ValueError, match="time skew"):
        estimate_vive_calibration(capture, max_time_skew_seconds=0.001)


def test_capture_rejects_an_ambiguous_libsurvive_pose_basis(tmp_path):
    document = capture_as_mapping(_capture())
    document["tracker_pose_frame"] = "raw_libsurvive"
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="convert raw libsurvive"):
        ViveCalibrationCapture.load(path)

    assert (
        capture_as_mapping(_capture())["tracker_pose_frame"] == VIVE_TRACKER_POSE_FRAME
    )


def test_calibration_script_help_documents_schema():
    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_vive.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "reference_T_wrist" in completed.stdout
    assert "lighthouse_T_tracker" in completed.stdout
    assert "dex_teleop_lighthouse_rh_z_up" in completed.stdout
    assert "position [x, -z, y]" in completed.stdout
    assert "At least six" in completed.stdout

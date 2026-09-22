import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.tracking.vive import (
    RigidTransform,
    ViveCalibration,
    ViveTrackerMount,
    ViveWristSource,
    VIVE_TRACKER_POSE_FRAME,
    _survive_pose_to_lighthouse_transform,
    compose_transforms,
)
from dex_teleop.types import Handedness, OPENXR_ANATOMICAL_WRIST_FRAME


def _calibration() -> ViveCalibration:
    return ViveCalibration(
        reference_frame="quest_tracking",
        reference_from_lighthouse=RigidTransform([1, 0, 0], [0, 0, 0, 1]),
        mounts=(
            ViveTrackerMount(
                serial="LHR-test-right",
                handedness=Handedness.RIGHT,
                tracker_to_wrist=RigidTransform([0, 0, 0.1], [0, 0, 0, 1]),
            ),
        ),
    )


def test_transform_composition_rotates_child_translation():
    quarter_turn = RigidTransform(
        [1, 2, 3],
        [0, 0, np.sqrt(0.5), np.sqrt(0.5)],
    )
    result = compose_transforms(quarter_turn, RigidTransform([1, 0, 0], [0, 0, 0, 1]))

    np.testing.assert_allclose(result.translation, [1, 3, 3], atol=1e-12)


def test_libsurvive_pose_conversion_is_a_proper_change_of_basis():
    raw_rotation = Rotation.from_euler("xyz", [0.31, -0.47, 0.83]).as_matrix()
    raw_quaternion_xyzw = Rotation.from_matrix(raw_rotation).as_quat()
    pose = SimpleNamespace(
        Pos=(2.0, 3.0, 4.0),
        Rot=(
            raw_quaternion_xyzw[3],
            raw_quaternion_xyzw[0],
            raw_quaternion_xyzw[1],
            raw_quaternion_xyzw[2],
        ),
    )

    converted = _survive_pose_to_lighthouse_transform(pose)
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

    assert np.linalg.det(basis) == pytest.approx(1.0)
    np.testing.assert_allclose(converted.translation, [2.0, -4.0, 3.0])
    np.testing.assert_allclose(
        Rotation.from_quat(converted.quaternion_xyzw).as_matrix(),
        basis @ raw_rotation @ basis.T,
        atol=1e-12,
    )


def test_vive_calibration_round_trip_and_rejects_duplicate_roles(tmp_path):
    path = tmp_path / "vive.json"
    path.write_text(json.dumps(_calibration().as_mapping()), encoding="utf-8")

    loaded = ViveCalibration.load(path)

    assert loaded.reference_frame == "quest_tracking"
    assert loaded.mounts[0].serial == "LHR-test-right"
    assert loaded.as_mapping()["lighthouse_frame"] == VIVE_TRACKER_POSE_FRAME
    with pytest.raises(ValueError, match="same hand"):
        ViveCalibration(
            reference_frame="tracking",
            reference_from_lighthouse=RigidTransform([0, 0, 0], [0, 0, 0, 1]),
            mounts=(
                loaded.mounts[0],
                loaded.mounts[0].__class__(
                    "WM-other", Handedness.RIGHT, loaded.mounts[0].tracker_to_wrist
                ),
            ),
        )


def test_vive_calibration_rejects_ambiguous_raw_lighthouse_frame(tmp_path):
    document = _calibration().as_mapping()
    del document["lighthouse_frame"]
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="raw libsurvive"):
        ViveCalibration.load(path)


def test_vive_calibration_rejects_non_object_and_missing_tracker_fields(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        ViveCalibration.load(path)

    document = _calibration().as_mapping()
    del document["trackers"][0]["serial"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid VIVE calibration"):
        ViveCalibration.load(path)


class _Update:
    def __init__(self, pose_timestamp=None):
        self.pose_timestamp = (
            time.time() - 0.005 if pose_timestamp is None else pose_timestamp
        )

    def Name(self):
        return b"TR0"

    def SerialNumber(self):
        return b"LHR-test-right"

    def Pose(self):
        pose = SimpleNamespace(Pos=(2.0, 3.0, 4.0), Rot=(1.0, 0.0, 0.0, 0.0))
        return pose, self.pose_timestamp


class _Context:
    def __init__(self):
        self.sent = False

    def NextUpdated(self):
        if not self.sent:
            self.sent = True
            return _Update()
        return None


def test_vive_source_uses_hardware_serial_native_time_and_both_calibrations():
    closed = []
    source = ViveWristSource(
        _calibration(),
        context_factory=_Context,
        context_close=lambda context: closed.append(context),
        serial_reader=lambda update: update.SerialNumber(),
    )
    source.start()
    try:
        deadline = time.monotonic() + 1.0
        sample = None
        while sample is None and time.monotonic() < deadline:
            sample = source.read_wrist(Handedness.RIGHT)
            time.sleep(0.001)
        assert sample is not None
        # libsurvive [x, y, z] -> [x, -z, y], then reference +1 x and mount +0.1 z.
        np.testing.assert_allclose(sample.position, [3.0, -4.0, 3.1])
        assert sample.source == "vive:LHR-test-right"
        assert sample.reference_frame == "quest_tracking"
        assert sample.anatomical_frame == OPENXR_ANATOMICAL_WRIST_FRAME
        assert sample.source_timestamp_ns is None
        assert sample.provenance["hardware_serial"] == "LHR-test-right"
        assert sample.provenance["libsurvive_codename"] == "TR0"
        assert (
            sample.provenance["raw_pose_timestamp_units"] == "seconds_since_unix_epoch"
        )
        assert isinstance(sample.provenance["raw_pose_timestamp"], float)
        assert 0.0 <= sample.receipt_timestamp - sample.timestamp < 0.1
        assert source.drain_wrists(Handedness.RIGHT) == (sample,)
        assert source.drain_wrists(Handedness.RIGHT) == ()
        assert source.read_wrist(Handedness.LEFT) is None
    finally:
        source.close()
    assert closed


def test_vive_source_fails_explicitly_when_pending_queue_overflows():
    class OverflowContext:
        def __init__(self):
            now = time.time()
            self.updates = [_Update(now - 0.003), _Update(now - 0.002)]

        def NextUpdated(self):
            return self.updates.pop(0) if self.updates else None

    source = ViveWristSource(
        _calibration(),
        max_pending_samples=1,
        context_factory=OverflowContext,
        serial_reader=lambda update: update.SerialNumber(),
    )
    try:
        try:
            source.start()
        except SourceUnavailableError as error:
            assert "overflowed" in str(error)
            return
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                source.check_health()
            except SourceUnavailableError as error:
                assert "overflowed" in str(error)
                break
            time.sleep(0.001)
        else:
            pytest.fail("VIVE queue overflow was not reported")
    finally:
        source.close()


def test_vive_close_interrupts_a_blocking_custom_context():
    released = []

    class BlockingContext:
        def __init__(self):
            self.stop = threading.Event()

        def NextUpdated(self):
            self.stop.wait()
            return None

    context = BlockingContext()
    source = ViveWristSource(
        _calibration(),
        context_factory=lambda: context,
        context_close=lambda value: (released.append(value), value.stop.set()),
        serial_reader=lambda update: update.SerialNumber(),
    )
    source.start()
    source.close()

    assert released == [context]

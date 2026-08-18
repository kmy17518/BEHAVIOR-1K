import time

import numpy as np
import pytest

from dex_teleop.runtime import SafetyConfig, SafetyFilter, TrackingRetargetingWorker
from dex_teleop.tracking.base import HandTrackingSource, SourceUnavailableError
from dex_teleop.tracking.hts import _HandState
from dex_teleop.types import HandFrame, Handedness, MEDIAPIPE_JOINT_NAMES, RetargetedHandCommand


def _frame(timestamp=None):
    landmarks = np.arange(63, dtype=np.float64).reshape(21, 3) / 1000.0
    return HandFrame(
        timestamp=time.monotonic() if timestamp is None else timestamp,
        handedness=Handedness.RIGHT,
        joints=dict(zip(MEDIAPIPE_JOINT_NAMES, landmarks, strict=True)),
        wrist_position=np.zeros(3),
        wrist_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        source="test",
    )


def test_hts_state_requires_complete_wrist_and_landmarks():
    state = _HandState()
    state.update_wrist([0, 0, 0, 0, 0, 0, 1])
    assert state.frame(Handedness.RIGHT) is None

    state.update_landmarks(np.arange(63) / 1000.0)
    state.wrist_timestamp = 10.0
    state.landmarks_timestamp = 11.0
    frame = state.frame(Handedness.RIGHT)
    assert frame is not None
    assert frame.timestamp == 10.0
    assert frame.mediapipe_landmarks().shape == (21, 3)


def test_hts_state_does_not_publish_untracked_zero_landmarks():
    state = _HandState()
    state.update_wrist([0, 0, 0, 0, 0, 0, 1])
    state.update_landmarks(np.zeros(63))

    assert state.frame(Handedness.RIGHT) is None


def test_safety_filter_rate_limits_instead_of_changing_dimensions():
    safety = SafetyFilter(
        n_arm=1,
        n_hand=1,
        dt=0.1,
        config=SafetyConfig(
            smoothing_alpha=1.0,
            max_arm_velocity=1.0,
            max_hand_velocity=2.0,
            max_arm_delta_per_tick=None,
            max_hand_delta_per_tick=None,
        ),
    )
    safety.reset(np.zeros(2))

    assert np.allclose(safety.apply(np.ones(2)), [0.1, 0.2])


class _Source(HandTrackingSource):
    def __init__(self, frame):
        self.frame = frame

    def start(self):
        pass

    def read(self, handedness):
        return self.frame

    def check_health(self):
        pass

    def close(self):
        pass


class _Retargeter:
    def retarget(self, frame):
        return RetargetedHandCommand(
            timestamp=frame.timestamp,
            handedness=frame.handedness,
            hand_model="sharpa",
            joint_names=tuple(f"joint_{index}" for index in range(22)),
            joint_positions=np.zeros(22),
        )


def test_worker_reports_stale_selected_source_without_substitution():
    worker = TrackingRetargetingWorker(_Source(_frame()), _Retargeter(), Handedness.RIGHT)
    worker.start()
    try:
        snapshot = worker.wait_for_first(timeout=1.0)
        object.__setattr__(snapshot.frame, "timestamp", time.monotonic() - 1.0)
        with pytest.raises(SourceUnavailableError, match="stale"):
            worker.snapshot(maximum_age=0.01)
    finally:
        worker.close()

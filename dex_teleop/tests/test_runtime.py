import time

import numpy as np
import pytest

from dex_teleop.runtime import SafetyConfig, SafetyFilter, TrackingRetargetingWorker
from dex_teleop.tracking.base import HandTrackingSource, SourceUnavailableError
from dex_teleop.tracking.hts import HTSProtocolError, HTSSource, _HandState, _parse_line
from dex_teleop.types import HandFrame, Handedness, MEDIAPIPE_JOINT_NAMES, RetargetedHandCommand


def _landmark_values(offset=0.0):
    return np.arange(63, dtype=np.float64) / 1000.0 + offset


def _hts_packet(frame_id, source_timestamp_ns, wrist_x=0.0, landmark_offset=0.0):
    landmarks = ", ".join(str(value) for value in _landmark_values(landmark_offset))
    return (
        f"Right wrist | f = {frame_id} | t = {source_timestamp_ns}:, "
        f"{wrist_x}, 0, 0, 0, 0, 0, 1\n"
        f"Right landmarks | f = {frame_id} | t = {source_timestamp_ns}:, {landmarks}"
    )


class _FakeTCPConnection:
    def __init__(self, payloads):
        self._payloads = iter(payloads)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self, size):
        return next(self._payloads)


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
    pairing_key = ("source", 3, 10_000_000_000)
    state.update_wrist(
        [0, 0, 0, 0, 0, 0, 1],
        receipt_timestamp=10.0,
        source_frame_id=3,
        source_timestamp_ns=10_000_000_000,
        pairing_key=pairing_key,
    )
    assert state.frame(Handedness.RIGHT) is None

    state.update_landmarks(
        _landmark_values(),
        receipt_timestamp=11.0,
        source_frame_id=3,
        source_timestamp_ns=10_000_000_000,
        pairing_key=pairing_key,
    )
    frame = state.frame(Handedness.RIGHT)
    assert frame is not None
    assert frame.timestamp == 11.0
    assert frame.receipt_timestamp == 11.0
    assert frame.source_timestamp_ns == 10_000_000_000
    assert frame.source_frame_id == 3
    assert frame.mediapipe_landmarks().shape == (21, 3)


def test_hts_state_does_not_publish_untracked_zero_landmarks():
    state = _HandState()
    pairing_key = ("source", 4, 11_000_000_000)
    state.update_wrist(
        [0, 0, 0, 0, 0, 0, 1],
        source_frame_id=4,
        source_timestamp_ns=11_000_000_000,
        pairing_key=pairing_key,
    )
    state.update_landmarks(
        np.zeros(63),
        source_frame_id=4,
        source_timestamp_ns=11_000_000_000,
        pairing_key=pairing_key,
    )

    assert state.frame(Handedness.RIGHT) is None


def test_hts_parser_preserves_source_pairing_metadata():
    record = _parse_line("Right wrist | f = 123 | t = 987654321:, 0, 0, 0, 0, 0, 0, 1")

    assert record is not None
    assert record.handedness == Handedness.RIGHT
    assert record.kind == "wrist"
    assert record.source_frame_id == 123
    assert record.source_timestamp_ns == 987654321


def test_hts_source_pairs_records_and_aligns_source_time_to_desktop_time():
    source = HTSSource()
    source._handle_payload(_hts_packet(1, 5_000_000_000), receipt_timestamp=100.0)

    first = source.read(Handedness.RIGHT)
    assert first is not None
    assert first.timestamp == 100.0
    assert first.receipt_timestamp == 100.0

    source._handle_payload(_hts_packet(2, 5_020_000_000), receipt_timestamp=100.125)
    second = source.read(Handedness.RIGHT)
    assert second is not None
    assert second.timestamp == pytest.approx(100.02)
    assert second.receipt_timestamp == 100.125
    assert second.source_timestamp_ns == 5_020_000_000
    assert second.source_frame_id == 2


def test_hts_source_errors_on_records_from_different_source_frames():
    source = HTSSource()
    wrist = "Right wrist | f = 1 | t = 1000:, 0, 0, 0, 0, 0, 0, 1"
    landmarks = ", ".join(str(value) for value in _landmark_values())
    source._handle_line(wrist, receipt_timestamp=10.0)
    with pytest.raises(HTSProtocolError, match="does not match incomplete pair"):
        source._handle_line(
            f"Right landmarks | f = 2 | t = 2000:, {landmarks}",
            receipt_timestamp=10.1,
        )

    assert source.read(Handedness.RIGHT) is None


def test_hts_source_does_not_publish_a_late_older_pair():
    source = HTSSource()
    source._handle_payload(_hts_packet(2, 2000, wrist_x=2), receipt_timestamp=10.1)
    current = source.read(Handedness.RIGHT)
    assert current is not None
    assert current.source_frame_id == 2

    source._handle_payload(_hts_packet(1, 1000, wrist_x=1), receipt_timestamp=10.2)

    assert source.read(Handedness.RIGHT) is current


def test_hts_source_errors_without_source_metadata():
    source = HTSSource()
    landmarks = ", ".join(str(value) for value in _landmark_values())
    with pytest.raises(HTSProtocolError, match="requires both source frame ID 'f' and source timestamp 't'"):
        source._handle_payload(
            f"Right wrist:, 0, 0, 0, 0, 0, 0, 1\nRight landmarks:, {landmarks}",
            receipt_timestamp=12.0,
        )


@pytest.mark.parametrize(
    "header",
    (
        "Right wrist | f = 1",
        "Right wrist | t = 1000",
    ),
)
def test_hts_source_errors_on_partial_source_metadata(header):
    with pytest.raises(HTSProtocolError, match="requires both source frame ID 'f' and source timestamp 't'"):
        _parse_line(f"{header}:, 0, 0, 0, 0, 0, 0, 1")


def test_hts_udp_datagram_requires_both_pair_components():
    source = HTSSource(protocol="udp")

    with pytest.raises(HTSProtocolError, match="must contain exactly one wrist and one landmarks record"):
        source._handle_payload(
            "Right wrist | f = 1 | t = 1000:, 0, 0, 0, 0, 0, 0, 1",
            receipt_timestamp=12.0,
            require_complete_pairs=True,
        )


def test_hts_tcp_pairs_records_split_across_receives():
    source = HTSSource(protocol="tcp")
    wrist, landmarks = _hts_packet(1, 1000).splitlines()
    connection = _FakeTCPConnection(
        [
            f"{wrist}\n".encode(),
            f"{landmarks}\n".encode(),
            b"",
        ]
    )

    source._handle_connection(connection)

    source.check_health()
    frame = source.read(Handedness.RIGHT)
    assert frame is not None
    assert frame.source_frame_id == 1


def test_hts_tcp_connection_close_with_incomplete_pair_fails_source():
    source = HTSSource(protocol="tcp")
    connection = _FakeTCPConnection(
        [
            b"Right wrist | f = 1 | t = 1000:, 0, 0, 0, 0, 0, 0, 1\n",
            b"",
        ]
    )

    source._handle_connection(connection)

    with pytest.raises(SourceUnavailableError, match="closed with incomplete pair") as error:
        source.check_health()
    assert isinstance(error.value.__cause__, HTSProtocolError)


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
        object.__setattr__(snapshot.frame, "receipt_timestamp", time.monotonic() - 1.0)
        with pytest.raises(SourceUnavailableError, match="stale"):
            worker.snapshot(maximum_age=0.01)
    finally:
        worker.close()

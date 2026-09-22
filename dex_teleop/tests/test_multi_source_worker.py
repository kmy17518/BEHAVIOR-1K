from collections import deque
import threading
import time

import numpy as np
import pytest

from dex_teleop.omnigibson.hand_tracking_recording import HandTrackingRecordingSession
from dex_teleop.runtime import MultiSourceTrackingWorker
from dex_teleop.tracking import SourceUnavailableError
from dex_teleop.tracking.multimodal import HandTrackingSampleBatch
from dex_teleop.types import (
    HandArticulationSample,
    Handedness,
    MEDIAPIPE_JOINT_NAMES,
    RetargetedHandCommand,
    WristPoseSample,
)


def _articulation(
    timestamp: float,
    *,
    source: str,
    frame_id: int,
) -> HandArticulationSample:
    positions = np.arange(63, dtype=np.float64).reshape(21, 3) / 1000.0
    return HandArticulationSample(
        timestamp=timestamp,
        receipt_timestamp=timestamp,
        source_timestamp_ns=round(timestamp * 1e9),
        source_frame_id=frame_id,
        handedness=Handedness.RIGHT,
        joint_positions=dict(zip(MEDIAPIPE_JOINT_NAMES, positions, strict=True)),
        source=source,
        schema="mediapipe21",
    )


def _wrist(
    timestamp: float,
    *,
    source: str,
    frame_id: int,
    x: float = 0.0,
) -> WristPoseSample:
    return WristPoseSample(
        timestamp=timestamp,
        receipt_timestamp=timestamp,
        source_timestamp_ns=round(timestamp * 1e9),
        source_frame_id=frame_id,
        handedness=Handedness.RIGHT,
        position=np.array([x, 0.0, 0.0]),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        source=source,
        reference_frame="world",
    )


class _FakeSource:
    def __init__(self, name, lifecycle_log=None, *, start_error=None):
        self.name = name
        self.lifecycle_log = [] if lifecycle_log is None else lifecycle_log
        self.start_error = start_error
        self.read_articulation_error = None
        self.read_wrist_error = None
        self.health_error = None
        self.articulations = deque()
        self.wrists = deque()
        self._lock = threading.Lock()
        self.wrist_reads = 0

    def start(self):
        self.lifecycle_log.append(("start", self.name))
        if self.start_error is not None:
            raise self.start_error

    def read_articulation(self, handedness):
        if self.read_articulation_error is not None:
            raise self.read_articulation_error
        with self._lock:
            return self.articulations.popleft() if self.articulations else None

    def read_wrist(self, handedness):
        if self.read_wrist_error is not None:
            raise self.read_wrist_error
        with self._lock:
            self.wrist_reads += 1
            return self.wrists.popleft() if self.wrists else None

    def push_articulation(self, sample):
        with self._lock:
            self.articulations.append(sample)

    def push_wrist(self, sample):
        with self._lock:
            self.wrists.append(sample)

    def check_health(self):
        if self.health_error is not None:
            raise self.health_error

    def close(self):
        self.lifecycle_log.append(("close", self.name))


class _LatestDrainingSource(_FakeSource):
    """Real-source shape: latest reads overwrite, drains retain every sample."""

    def __init__(self, name):
        super().__init__(name)
        self.latest_articulation = None
        self.latest_wrist = None
        self.articulation_drains = 0
        self.wrist_drains = 0

    def push_articulation(self, sample):
        with self._lock:
            self.latest_articulation = sample
            self.articulations.append(sample)

    def push_wrist(self, sample):
        with self._lock:
            self.latest_wrist = sample
            self.wrists.append(sample)

    def read_articulation(self, handedness):
        with self._lock:
            return self.latest_articulation

    def read_wrist(self, handedness):
        with self._lock:
            self.wrist_reads += 1
            return self.latest_wrist

    def drain_articulations(self, handedness):
        with self._lock:
            self.articulation_drains += 1
            samples = tuple(self.articulations)
            self.articulations.clear()
            return samples

    def drain_wrists(self, handedness):
        with self._lock:
            self.wrist_drains += 1
            samples = tuple(self.wrists)
            self.wrists.clear()
            return samples


class _CombinedSource(_FakeSource):
    def __init__(self, name):
        super().__init__(name)
        self.combined_drains = 0
        self.latest_articulation = None
        self.latest_wrist = None

    def push_pair(self, articulation, wrist):
        with self._lock:
            self.latest_articulation = articulation
            self.latest_wrist = wrist
            self.articulations.append(articulation)
            self.wrists.append(wrist)

    def read_articulation(self, handedness):
        with self._lock:
            return self.latest_articulation

    def read_wrist(self, handedness):
        with self._lock:
            return self.latest_wrist

    def drain_hand_tracking(self, handedness):
        with self._lock:
            self.combined_drains += 1
            result = HandTrackingSampleBatch(
                articulations=tuple(self.articulations),
                wrists=tuple(self.wrists),
            )
            self.articulations.clear()
            self.wrists.clear()
            return result


class _FakeRetargeter:
    hand_model = "sharpa"
    hand_side = "right"
    joint_names = ("joint",)

    def __init__(self):
        self.retarget_thread_ids = []
        self.reset_thread_ids = []

    def retarget(self, frame):
        self.retarget_thread_ids.append(threading.get_ident())
        return RetargetedHandCommand(
            timestamp=frame.timestamp,
            handedness=frame.handedness,
            hand_model=self.hand_model,
            joint_names=self.joint_names,
            joint_positions=np.array([frame.wrist_position[0]]),
        )

    def reset(self):
        self.reset_thread_ids.append(threading.get_ident())


class _BlockingRetargeter(_FakeRetargeter):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.frame_ids = []

    def retarget(self, frame):
        self.frame_ids.append(frame.source_frame_id)
        if not self.entered.is_set():
            self.entered.set()
            if not self.release.wait(timeout=2.0):
                raise TimeoutError("test retargeter was not released")
        return super().retarget(frame)


class _RichRetargeter:
    hand_model = "sharpa"
    hand_side = "right"
    joint_names = ("joint",)

    def __init__(self):
        self.observations = []

    def retarget_observation(self, observation):
        self.observations.append(observation)
        return RetargetedHandCommand(
            timestamp=observation.timestamp,
            handedness=observation.handedness,
            hand_model=self.hand_model,
            joint_names=self.joint_names,
            joint_positions=np.array([observation.wrist.position[0]]),
        )

    def reset(self):
        pass


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("Condition was not satisfied before timeout")


def test_multi_source_worker_requires_explicit_registered_modalities():
    quest = _FakeSource("quest")
    with pytest.raises(
        ValueError, match="Unknown selected articulation source 'manus'"
    ):
        MultiSourceTrackingWorker(
            {"quest": quest},
            articulation_source="manus",
            wrist_source="quest",
            retargeter=_FakeRetargeter(),
            handedness=Handedness.RIGHT,
        )

    class _ArticulationOnly:
        def start(self):
            pass

        def read_articulation(self, handedness):
            return None

        def check_health(self):
            pass

        def close(self):
            pass

    with pytest.raises(TypeError, match="Wrist source 'quest'"):
        MultiSourceTrackingWorker(
            {"quest": _ArticulationOnly()},
            articulation_source="quest",
            wrist_source="quest",
            retargeter=_FakeRetargeter(),
            handedness=Handedness.RIGHT,
        )


def test_multi_source_worker_preserves_rich_observation_for_capable_retargeter():
    base = time.monotonic()
    quest = _FakeSource("quest")
    articulation = _articulation(base, source="hts", frame_id=1)
    quest.push_articulation(articulation)
    quest.push_wrist(_wrist(base, source="hts", frame_id=1, x=0.25))
    retargeter = _RichRetargeter()
    worker = MultiSourceTrackingWorker(
        {"quest": quest},
        articulation_source="quest",
        wrist_source="quest",
        retargeter=retargeter,
        handedness=Handedness.RIGHT,
    )

    worker.start()
    try:
        snapshot = worker.wait_for_first(1.0)
    finally:
        worker.close()

    assert retargeter.observations[0].articulation is articulation
    assert snapshot.observation is retargeter.observations[0]
    assert snapshot.command.joint_positions[0] == pytest.approx(0.25)


def test_worker_bypasses_skew_search_for_same_callback_multimodal_pair():
    base = time.monotonic()
    source = _CombinedSource("manus")
    articulation = _articulation(
        base,
        source="manus:remote:1417281806",
        frame_id=7,
    )
    wrist = _wrist(
        base,
        source="manus:remote:1417281806",
        frame_id=7,
        x=0.75,
    )
    source.push_pair(articulation, wrist)
    retargeter = _RichRetargeter()
    session = HandTrackingRecordingSession()
    worker = MultiSourceTrackingWorker(
        {"manus": source},
        articulation_source="manus",
        wrist_source="manus",
        retargeter=retargeter,
        handedness=Handedness.RIGHT,
        maximum_skew_seconds=0.0,
        recording_session=session,
    )

    def forbid_timestamp_search(*_args, **_kwargs):
        raise AssertionError("same-callback MANUS pair entered timestamp search")

    worker.fuser.fuse = forbid_timestamp_search
    worker.start()
    try:
        snapshot = worker.wait_for_first(1.0)
    finally:
        worker.close()

    assert snapshot.observation.articulation is articulation
    assert snapshot.observation.wrist is wrist
    assert snapshot.observation.synchronization_skew_seconds == 0.0
    assert snapshot.wrist_stream == "wrist.manus"
    assert source.combined_drains > 0
    recorded = session.writer_kwargs()
    assert recorded["articulation_streams"]["articulation.manus"] == (articulation,)
    assert recorded["wrist_streams"]["wrist.manus"] == (wrist,)


def test_multi_source_worker_deduplicates_lifecycle_and_closes_in_reverse_registry_order():
    lifecycle = []
    shadow = _FakeSource("shadow", lifecycle)
    quest = _FakeSource("quest", lifecycle)
    worker = MultiSourceTrackingWorker(
        {"shadow": shadow, "quest": quest},
        articulation_source="quest",
        wrist_source="quest",
        record_articulation_sources=("shadow",),
        record_wrist_sources=("quest",),
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
    )

    worker.start()
    worker.close()

    assert lifecycle == [
        ("start", "shadow"),
        ("start", "quest"),
        ("close", "quest"),
        ("close", "shadow"),
    ]


def test_multi_source_worker_rolls_back_a_partial_start_in_reverse_order():
    lifecycle = []
    quest = _FakeSource("quest", lifecycle)
    shadow = _FakeSource("shadow", lifecycle, start_error=OSError("not present"))
    worker = MultiSourceTrackingWorker(
        {"quest": quest, "shadow": shadow},
        articulation_source="quest",
        wrist_source="quest",
        record_articulation_sources=("shadow",),
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
    )

    with pytest.raises(
        SourceUnavailableError, match="record-only articulation source 'shadow'"
    ):
        worker.start()

    assert lifecycle == [
        ("start", "quest"),
        ("start", "shadow"),
        ("close", "shadow"),
        ("close", "quest"),
    ]


def test_record_only_wrist_is_recorded_but_never_substituted_for_selected_wrist():
    base = time.monotonic()
    manus = _FakeSource("manus")
    quest = _FakeSource("quest")
    vive = _FakeSource("vive")
    manus.push_articulation(_articulation(base + 0.2, source="manus", frame_id=1))
    quest.push_wrist(_wrist(base, source="hts", frame_id=1))
    vive.push_wrist(_wrist(base + 0.2, source="vive:tracker", frame_id=1, x=9.0))
    session = HandTrackingRecordingSession()
    worker = MultiSourceTrackingWorker(
        {"manus": manus, "quest": quest, "vive": vive},
        articulation_source="manus",
        wrist_source="quest",
        record_wrist_sources=("vive",),
        maximum_skew_seconds=0.01,
        recording_session=session,
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
    )
    worker.start()
    try:
        with pytest.raises(
            SourceUnavailableError,
            match="articulation source 'manus'.*wrist source 'quest'",
        ):
            worker.wait_for_first(0.05)
        assert len(session.writer_kwargs()["wrist_streams"]["wrist.vive"]) == 1

        quest.push_wrist(_wrist(base + 0.2, source="hts", frame_id=2, x=1.0))
        snapshot = worker.wait_for_first(1.0)
        assert snapshot.frame.wrist_position[0] == pytest.approx(1.0)
        assert snapshot.frame.wrist_position[0] != 9.0
    finally:
        worker.close()


def test_recording_attachment_uses_registry_stream_ids_and_resets_rows_per_task():
    base = time.monotonic()
    manus = _FakeSource("manus")
    quest = _FakeSource("quest")
    vive = _FakeSource("vive")
    worker = MultiSourceTrackingWorker(
        {"manus": manus, "quest": quest, "vive": vive},
        articulation_source="manus",
        wrist_source="quest",
        record_articulation_sources=("quest",),
        record_wrist_sources=("vive",),
        retargeter=_FakeRetargeter(),
        retargeter_name="adaptive",
        handedness=Handedness.RIGHT,
    )
    worker.start()
    first_session = HandTrackingRecordingSession()
    try:
        worker.set_recording_session(first_session)
        manus.push_articulation(_articulation(base, source="manus-sdk", frame_id=1))
        quest.push_articulation(_articulation(base, source="hts", frame_id=1))
        quest.push_wrist(_wrist(base, source="hts", frame_id=1, x=0.1))
        vive.push_wrist(_wrist(base, source="vive:serial", frame_id=1, x=0.2))
        first = worker.wait_for_first(1.0)

        assert first.articulation_stream == "articulation.manus"
        assert first.articulation_row == 0
        assert first.wrist_stream == "wrist.quest"
        assert first.wrist_row == 0
        assert first.native_wrist_stream == "wrist.quest"
        assert first.retargeting_stream == "adaptive.manus+quest"
        assert first.retargeting_row == 0
        selection = first.action_selection()
        assert selection is not None
        episode = first_session.begin_episode()
        assert (
            first_session.append_action_selection(selection, episode_index=episode) == 0
        )

        recorded = first_session.writer_kwargs()
        assert set(recorded["articulation_streams"]) == {
            "articulation.manus",
            "articulation.quest",
        }
        assert set(recorded["wrist_streams"]) == {"wrist.quest", "wrist.vive"}
        assert set(recorded["retargeting_streams"]) == {"adaptive.manus+quest"}
        assert recorded["articulation_streams"]["articulation.quest"][0].source == "hts"

        worker.set_recording_session(None)
        first_session.close()
        second_session = HandTrackingRecordingSession()
        worker.set_recording_session(second_session)
        manus.push_articulation(
            _articulation(base + 1.0, source="manus-sdk", frame_id=2)
        )
        quest.push_articulation(_articulation(base + 1.0, source="hts", frame_id=2))
        quest.push_wrist(_wrist(base + 1.0, source="hts", frame_id=2, x=0.3))
        vive.push_wrist(_wrist(base + 1.0, source="vive:serial", frame_id=2, x=0.4))
        second = worker.wait_for_first(1.0)
        assert second.articulation_row == 0
        assert second.wrist_row == 0
        assert second.retargeting_row == 0
        worker.set_recording_session(None)
        second_session.close()
    finally:
        worker.close()


def test_interpolated_control_wrist_gets_an_exact_derived_recording_row():
    base = time.monotonic()
    manus = _FakeSource("manus")
    quest = _FakeSource("quest")
    session = HandTrackingRecordingSession()
    worker = MultiSourceTrackingWorker(
        {"manus": manus, "quest": quest},
        articulation_source="manus",
        wrist_source="quest",
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
        maximum_skew_seconds=1.1,
        recording_session=session,
    )
    worker.start()
    try:
        quest.push_wrist(_wrist(base, source="hts", frame_id=1, x=0.0))
        quest.push_wrist(_wrist(base + 2.0, source="hts", frame_id=2, x=2.0))
        _wait_until(lambda: quest.wrist_reads >= 3)
        manus.push_articulation(
            _articulation(base + 1.0, source="manus-sdk", frame_id=1)
        )
        snapshot = worker.wait_for_first(1.0)

        assert snapshot.frame.wrist_position[0] == pytest.approx(1.0)
        assert snapshot.wrist_stream == "wrist.quest.control"
        assert snapshot.wrist_row == 0
        assert snapshot.native_wrist_stream is None
        control = session.writer_kwargs()["wrist_streams"]["wrist.quest.control"][0]
        assert control.timestamp == pytest.approx(base + 1.0)
        assert control.position[0] == pytest.approx(1.0)
    finally:
        worker.close()


def test_reset_runs_on_worker_thread_and_flushes_latest_and_fusion_state():
    base = time.monotonic()
    quest = _FakeSource("quest")
    retargeter = _FakeRetargeter()
    quest.push_wrist(_wrist(base, source="hts", frame_id=1))
    quest.push_articulation(_articulation(base, source="hts", frame_id=1))
    worker = MultiSourceTrackingWorker(
        {"quest": quest},
        articulation_source="quest",
        wrist_source="quest",
        retargeter=retargeter,
        handedness=Handedness.RIGHT,
    )
    worker.start()
    try:
        worker.wait_for_first(1.0)
        caller_thread = threading.get_ident()
        worker.reset()
        assert worker.snapshot() is None
        assert retargeter.reset_thread_ids == retargeter.retarget_thread_ids
        assert retargeter.reset_thread_ids[0] != caller_thread
        assert worker.fuser.buffered_wrist_count() == 0

        quest.push_wrist(_wrist(base + 1.0, source="hts", frame_id=2))
        quest.push_articulation(_articulation(base + 1.0, source="hts", frame_id=2))
        worker.wait_for_first(1.0)
        assert len(retargeter.retarget_thread_ids) == 2
    finally:
        worker.close()


def test_worker_error_names_a_failing_record_only_source_role():
    quest = _FakeSource("quest")
    shadow = _FakeSource("shadow")
    shadow.read_articulation_error = OSError("camera disconnected")
    worker = MultiSourceTrackingWorker(
        {"quest": quest, "shadow": shadow},
        articulation_source="quest",
        wrist_source="quest",
        record_articulation_sources=("shadow",),
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
    )
    worker.start()
    try:
        with pytest.raises(
            RuntimeError,
            match="Record-only articulation source 'shadow'.*camera disconnected",
        ):
            worker.wait_for_first(1.0)
    finally:
        worker.close()


def test_wait_for_first_requires_every_explicit_record_only_stream():
    base = time.monotonic()
    quest = _FakeSource("quest")
    shadow = _FakeSource("shadow")
    vive = _FakeSource("vive")
    quest.push_wrist(_wrist(base, source="hts", frame_id=1))
    quest.push_articulation(_articulation(base, source="hts", frame_id=1))
    worker = MultiSourceTrackingWorker(
        {"quest": quest, "shadow": shadow, "vive": vive},
        articulation_source="quest",
        wrist_source="quest",
        record_articulation_sources=("shadow",),
        record_wrist_sources=("vive",),
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
    )
    worker.start()
    try:
        with pytest.raises(
            SourceUnavailableError,
            match=(
                "record-only articulation source 'shadow'.*"
                "record-only wrist source 'vive'"
            ),
        ):
            worker.wait_for_first(0.05)
        assert worker.snapshot() is not None

        shadow.push_articulation(_articulation(base, source="shadow", frame_id=1))
        vive.push_wrist(_wrist(base, source="vive:tracker", frame_id=1))
        assert worker.wait_for_first(1.0) is worker.snapshot()
    finally:
        worker.close()


def test_worker_waits_for_future_wrist_bracket_before_using_nearest():
    base = time.monotonic()
    quest = _FakeSource("quest")
    quest.push_wrist(_wrist(base, source="hts", frame_id=1, x=0.0))
    quest.push_articulation(_articulation(base + 0.01, source="hts", frame_id=1))
    worker = MultiSourceTrackingWorker(
        {"quest": quest},
        articulation_source="quest",
        wrist_source="quest",
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
        maximum_skew_seconds=0.1,
        interpolation_wait_seconds=0.05,
    )
    worker.start()
    try:
        time.sleep(0.01)
        assert worker.snapshot() is None
        quest.push_wrist(_wrist(base + 0.02, source="hts", frame_id=2, x=2.0))
        snapshot = worker.wait_for_first(1.0)
        assert snapshot.observation.wrist.timestamp == pytest.approx(base + 0.01)
        assert snapshot.frame.wrist_position[0] == pytest.approx(1.0)
        assert snapshot.observation.wrist.provenance["interpolated"] is True
    finally:
        worker.close()


def test_worker_interpolation_wait_has_a_bounded_nearest_fallback():
    base = time.monotonic()
    quest = _FakeSource("quest")
    quest.push_wrist(_wrist(base, source="hts", frame_id=1, x=0.5))
    quest.push_articulation(_articulation(base + 0.01, source="hts", frame_id=1))
    worker = MultiSourceTrackingWorker(
        {"quest": quest},
        articulation_source="quest",
        wrist_source="quest",
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
        maximum_skew_seconds=0.1,
        interpolation_wait_seconds=0.03,
    )
    started = time.monotonic()
    worker.start()
    try:
        time.sleep(0.01)
        assert worker.snapshot() is None
        snapshot = worker.wait_for_first(1.0)
        assert time.monotonic() - started >= 0.025
        assert snapshot.observation.wrist.source_frame_id == 1
        assert snapshot.frame.wrist_position[0] == pytest.approx(0.5)
    finally:
        worker.close()


def test_high_rate_articulation_cannot_reset_interpolation_deadline_forever():
    base = time.monotonic()
    quest = _FakeSource("quest")
    quest.push_wrist(_wrist(base - 0.2, source="hts", frame_id=1, x=0.5))
    quest.push_articulation(_articulation(base, source="hts", frame_id=1))
    worker = MultiSourceTrackingWorker(
        {"quest": quest},
        articulation_source="quest",
        wrist_source="quest",
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
        maximum_skew_seconds=0.25,
        interpolation_wait_seconds=0.02,
    )

    def produce_lagging_streams():
        for frame_id in range(2, 102):
            timestamp = base + frame_id * 0.001
            quest.push_wrist(
                _wrist(
                    timestamp - 0.2,
                    source="hts",
                    frame_id=frame_id,
                    x=0.5,
                )
            )
            quest.push_articulation(
                _articulation(timestamp, source="hts", frame_id=frame_id)
            )
            time.sleep(0.001)

    producer = threading.Thread(target=produce_lagging_streams)
    worker.start()
    producer.start()
    try:
        snapshot = worker.wait_for_first(0.07)
        assert producer.is_alive()
        assert snapshot.frame.wrist_position[0] == pytest.approx(0.5)
    finally:
        producer.join(timeout=1.0)
        worker.close()


def test_stream_freshness_names_stale_record_only_comparison_roles():
    base = time.monotonic()
    quest = _FakeSource("quest")
    shadow = _FakeSource("shadow")
    vive = _FakeSource("vive")
    quest.push_wrist(_wrist(base, source="hts", frame_id=1))
    quest.push_articulation(_articulation(base, source="hts", frame_id=1))
    shadow.push_articulation(_articulation(base, source="shadow", frame_id=1))
    vive.push_wrist(_wrist(base, source="vive:tracker", frame_id=1))
    worker = MultiSourceTrackingWorker(
        {"quest": quest, "shadow": shadow, "vive": vive},
        articulation_source="quest",
        wrist_source="quest",
        record_articulation_sources=("shadow",),
        record_wrist_sources=("vive",),
        retargeter=_FakeRetargeter(),
        handedness=Handedness.RIGHT,
    )
    worker.start()
    try:
        worker.wait_for_first(1.0)
        worker.check_stream_freshness(1.0)
        time.sleep(0.03)
        fresh_timestamp = time.monotonic()
        quest.push_wrist(_wrist(fresh_timestamp, source="hts", frame_id=2))
        quest.push_articulation(
            _articulation(fresh_timestamp, source="hts", frame_id=2)
        )
        _wait_until(
            lambda: (
                worker.snapshot() is not None
                and worker.snapshot().frame.source_frame_id == 2
            )
        )
        with pytest.raises(
            SourceUnavailableError,
            match=(
                "record-only articulation source 'shadow'.*"
                "record-only wrist source 'vive'"
            ),
        ):
            worker.check_stream_freshness(0.02)
    finally:
        worker.close()


def test_native_drains_preserve_samples_acquired_while_retargeting_is_slow():
    base = time.monotonic()
    quest = _LatestDrainingSource("quest")
    quest.push_wrist(_wrist(base, source="hts", frame_id=1, x=1.0))
    quest.push_articulation(_articulation(base, source="hts", frame_id=1))
    retargeter = _BlockingRetargeter()
    session = HandTrackingRecordingSession()
    worker = MultiSourceTrackingWorker(
        {"quest": quest},
        articulation_source="quest",
        wrist_source="quest",
        retargeter=retargeter,
        handedness=Handedness.RIGHT,
        recording_session=session,
    )
    worker.start()
    try:
        assert retargeter.entered.wait(timeout=1.0)
        for frame_id in range(2, 6):
            timestamp = base + frame_id * 0.01
            quest.push_wrist(
                _wrist(timestamp, source="hts", frame_id=frame_id, x=frame_id)
            )
            quest.push_articulation(
                _articulation(timestamp, source="hts", frame_id=frame_id)
            )
        retargeter.release.set()
        _wait_until(
            lambda: (
                len(
                    session.writer_kwargs()["articulation_streams"].get(
                        "articulation.quest", ()
                    )
                )
                == 5
                and worker.snapshot() is not None
                and worker.snapshot().frame.source_frame_id == 5
            ),
            timeout=1.0,
        )

        recorded = session.writer_kwargs()
        assert [
            sample.source_frame_id
            for sample in recorded["articulation_streams"]["articulation.quest"]
        ] == [1, 2, 3, 4, 5]
        assert [
            sample.source_frame_id
            for sample in recorded["wrist_streams"]["wrist.quest"]
        ] == [1, 2, 3, 4, 5]
        assert retargeter.frame_ids == [1, 5]
        assert quest.articulation_drains > 0
        assert quest.wrist_drains > 0
    finally:
        retargeter.release.set()
        worker.close()

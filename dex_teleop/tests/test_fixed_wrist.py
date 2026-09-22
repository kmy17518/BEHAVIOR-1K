"""FixedWristSource: articulation-only tracking with a simulator-held wrist."""

from collections import deque
import threading
import time

import numpy as np
import pytest

from dex_teleop.runtime import MultiSourceTrackingWorker
from dex_teleop.tracking import (
    FIXED_WRIST_REFERENCE_FRAME,
    FIXED_WRIST_SOURCE_NAME,
    FixedWristSource,
    HandObservationFuser,
    HandTrackingSampleBatch,
)
from dex_teleop.types import (
    MEDIAPIPE_JOINT_NAMES,
    OPENXR_ANATOMICAL_WRIST_FRAME,
    HandArticulationSample,
    Handedness,
    RetargetedHandCommand,
    WristPoseSample,
)


def _articulation(timestamp, *, frame_id, source="hts", offset=0.0):
    positions = np.arange(63, dtype=np.float64).reshape(21, 3) / 1000.0 + offset
    return HandArticulationSample(
        timestamp=timestamp,
        receipt_timestamp=timestamp + 0.001,
        source_timestamp_ns=round(timestamp * 1e9),
        source_frame_id=frame_id,
        handedness=Handedness.RIGHT,
        joint_positions=dict(zip(MEDIAPIPE_JOINT_NAMES, positions, strict=True)),
        source=source,
        schema="mediapipe21",
        coordinate_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
    )


def _device_wrist(articulation, x=0.4):
    return WristPoseSample(
        timestamp=articulation.timestamp,
        receipt_timestamp=articulation.receipt_timestamp,
        source_timestamp_ns=articulation.source_timestamp_ns,
        source_frame_id=articulation.source_frame_id,
        handedness=Handedness.RIGHT,
        position=np.array([x, 0.0, 0.0]),
        quaternion_xyzw=np.array([0.0, 0.0, 1.0, 0.0]),
        source=articulation.source,
        reference_frame="world",
        anatomical_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
    )


class _LatestOnlySource:
    """Minimal articulation source: read_articulation() peeks at the latest sample."""

    def __init__(self):
        self.latest = None
        self.events = []

    def start(self):
        self.events.append("start")

    def check_health(self):
        self.events.append("health")

    def close(self):
        self.events.append("close")

    def read_articulation(self, handedness):
        return self.latest


class _CombinedSource(_LatestOnlySource):
    """Receiver that also carries its own wrist, which the wrapper must discard."""

    def __init__(self):
        super().__init__()
        self.pairs = deque()
        self._lock = threading.Lock()

    def push(self, articulation):
        with self._lock:
            self.latest = articulation
            self.pairs.append((articulation, _device_wrist(articulation)))

    def read_wrist(self, handedness):
        return None if self.latest is None else _device_wrist(self.latest)

    def drain_hand_tracking(self, handedness):
        with self._lock:
            pairs = tuple(self.pairs)
            self.pairs.clear()
        return HandTrackingSampleBatch(
            articulations=tuple(pair[0] for pair in pairs),
            wrists=tuple(pair[1] for pair in pairs),
        )


class _FingerRetargeter:
    hand_model = "sharpa"
    hand_side = "right"
    joint_names = ("joint",)

    def retarget(self, frame):
        return RetargetedHandCommand(
            timestamp=frame.timestamp,
            handedness=frame.handedness,
            hand_model=self.hand_model,
            joint_names=self.joint_names,
            joint_positions=np.array([frame.joints["index_tip"][0]]),
        )

    def reset(self):
        pass


def test_fixed_wrist_mirrors_articulation_identity_and_anatomical_frame():
    inner = _LatestOnlySource()
    fixed = FixedWristSource(inner, position=(0.1, 0.2, 0.3), quaternion_xyzw=(0.0, 0.0, 2.0, 0.0))
    articulation = _articulation(10.0, frame_id=7)

    wrist = fixed.wrist_for(articulation)

    assert wrist.timestamp == articulation.timestamp
    assert wrist.receipt_timestamp == articulation.receipt_timestamp
    assert wrist.source_frame_id == 7
    assert wrist.source_timestamp_ns == articulation.source_timestamp_ns
    assert wrist.source == FIXED_WRIST_SOURCE_NAME
    assert wrist.reference_frame == FIXED_WRIST_REFERENCE_FRAME
    assert wrist.anatomical_frame == OPENXR_ANATOMICAL_WRIST_FRAME
    assert np.allclose(wrist.position, [0.1, 0.2, 0.3])
    assert np.allclose(wrist.quaternion_xyzw, [0.0, 0.0, 1.0, 0.0])
    assert wrist.provenance["articulation_source"] == "hts"
    assert wrist.provenance["fixed_wrist"] is True


def test_fixed_wrist_discards_the_receivers_own_wrist_and_pairs_one_to_one():
    inner = _CombinedSource()
    fixed = FixedWristSource(inner)
    for index in range(3):
        inner.push(_articulation(1.0 + index * 0.02, frame_id=index))

    batch = fixed.drain_hand_tracking(Handedness.RIGHT)

    assert len(batch.articulations) == len(batch.wrists) == 3
    for articulation, wrist in zip(batch.articulations, batch.wrists, strict=True):
        assert wrist.source == FIXED_WRIST_SOURCE_NAME
        assert wrist.source_frame_id == articulation.source_frame_id
        assert np.allclose(wrist.position, 0.0)
        assert np.allclose(wrist.quaternion_xyzw, [0.0, 0.0, 0.0, 1.0])
    # The Quest-style wrist at x=0.4 never leaves the wrapper.
    assert fixed.read_wrist(Handedness.RIGHT).position[0] == 0.0
    assert fixed.drain_hand_tracking(Handedness.RIGHT) == HandTrackingSampleBatch()


def test_fixed_wrist_dedupes_latest_only_sources_and_delegates_lifecycle():
    inner = _LatestOnlySource()
    fixed = FixedWristSource(inner)
    fixed.start()
    fixed.check_health()

    assert fixed.drain_hand_tracking(Handedness.RIGHT) == HandTrackingSampleBatch()
    assert fixed.read_wrist(Handedness.RIGHT) is None
    inner.latest = _articulation(2.0, frame_id=1)
    first = fixed.drain_hand_tracking(Handedness.RIGHT)
    repeated = fixed.drain_hand_tracking(Handedness.RIGHT)
    inner.latest = _articulation(2.1, frame_id=2)
    second = fixed.drain_hand_tracking(Handedness.RIGHT)
    fixed.close()

    assert [sample.source_frame_id for sample in first.articulations] == [1]
    assert repeated == HandTrackingSampleBatch()
    assert [sample.source_frame_id for sample in second.articulations] == [2]
    assert fixed.read_wrist(Handedness.RIGHT).source_frame_id == 2
    assert inner.events == ["start", "health", "close"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"quaternion_xyzw": (0.0, 0.0, 0.0, 0.0)}, "non-zero norm"),
        ({"position": (float("nan"), 0.0, 0.0)}, "finite"),
        ({"source_name": "a/b"}, "may not contain"),
    ],
)
def test_fixed_wrist_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FixedWristSource(_LatestOnlySource(), **kwargs)


def test_fixed_wrist_requires_an_articulation_source():
    class _WristOnly:
        def start(self):
            pass

        def check_health(self):
            pass

        def close(self):
            pass

    with pytest.raises(TypeError, match="read_articulation"):
        FixedWristSource(_WristOnly())


def test_worker_fuses_fixed_wrist_as_co_emitted_pairs():
    inner = _CombinedSource()
    fixed = FixedWristSource(inner)
    worker = MultiSourceTrackingWorker(
        sources={"quest": fixed, FIXED_WRIST_SOURCE_NAME: fixed},
        articulation_source="quest",
        wrist_source=FIXED_WRIST_SOURCE_NAME,
        retargeter=_FingerRetargeter(),
        handedness=Handedness.RIGHT,
        fuser=HandObservationFuser(interpolate_wrist=False),
        retargeter_name="adaptive",
    )
    now = time.monotonic()
    inner.push(_articulation(now, frame_id=1, offset=0.5))
    with worker:
        snapshot = worker.wait_for_first(2.0)
        # A second sample is fused without any timestamp search or skew.
        inner.push(_articulation(now + 0.02, frame_id=2, offset=0.25))
        deadline = time.monotonic() + 2.0
        while worker.snapshot().observation.articulation.source_frame_id != 2:
            assert time.monotonic() < deadline, "second fixed-wrist observation was not published"
            time.sleep(0.002)
        latest = worker.snapshot()

    assert inner.events[0] == "start" and inner.events[-1] == "close"
    assert worker.retargeting_stream == "adaptive.quest+fixed_wrist"
    assert snapshot.observation.wrist.source == FIXED_WRIST_SOURCE_NAME
    assert snapshot.observation.synchronization_skew_seconds == 0.0
    assert np.allclose(snapshot.frame.wrist_position, 0.0)
    # World-frame landmarks equal the wrist-local ones because the wrist is the identity.
    assert np.allclose(snapshot.frame.joints["index_tip"], snapshot.observation.articulation.joint_positions["index_tip"])
    assert snapshot.command.joint_positions[0] == pytest.approx(snapshot.frame.joints["index_tip"][0])
    assert latest.observation.articulation.source_frame_id == 2
    assert latest.observation.wrist.source_frame_id == 2

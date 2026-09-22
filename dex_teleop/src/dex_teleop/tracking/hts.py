"""Meta Quest Hand Tracking Streamer (HTS) source.

Adapted from AnyDexRetarget's ``example/input/quest3.py`` (MIT License,
Copyright (c) 2025 Shiquan Qiu). See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from dex_teleop.tracking.base import HandTrackingSource, SourceUnavailableError
from dex_teleop.tracking.multimodal import HandTrackingSampleBatch
from dex_teleop.types import (
    HandArticulationSample,
    HandFrame,
    Handedness,
    MEDIAPIPE_JOINT_NAMES,
    OPENXR_ANATOMICAL_WRIST_FRAME,
    WristPoseSample,
)

LOGGER = logging.getLogger(__name__)


class HTSProtocolError(ValueError):
    """Raised when HTS hand data does not satisfy the expected wire contract."""


class HTSBufferOverflowError(RuntimeError):
    """Raised instead of silently dropping an unread native-rate HTS sample."""


# Unity LH (x right, y up, z forward) -> RH (x forward, y left, z up).
_UNITY_TO_RH = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

# Legacy HTS output uses x-forward, y-left, z-up.  Rich multimodal values use
# the OpenXR anatomical wrist basis: x-right, y-up, z-back.  The legacy
# ``HandFrame`` below deliberately remains in its historical basis.
_D_TO_OPENXR = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("HTS supplied a zero-norm wrist quaternion")
    return quaternion / norm


def _rotate(points: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = _normalize_quaternion(quaternion_xyzw)
    xyz = quaternion[:3]
    w = quaternion[3]
    cross = 2.0 * np.cross(xyz, points)
    return points + w * cross + np.cross(xyz, cross)


def _quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) * scale,
                (matrix[0, 2] - matrix[2, 0]) * scale,
                (matrix[1, 0] - matrix[0, 1]) * scale,
                0.25 / scale,
            ]
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ]
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quaternion = np.array(
            [
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quaternion = np.array(
            [
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    return _normalize_quaternion(quaternion)


def _convert_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    unity_rotation = _quaternion_to_matrix(quaternion_xyzw)
    return _matrix_to_quaternion(_UNITY_TO_RH @ unity_rotation @ _UNITY_TO_RH.T)


@dataclass(frozen=True)
class _HTSRecord:
    handedness: Handedness
    kind: str
    values: tuple[float, ...]
    source_frame_id: int | None = None
    source_timestamp_ns: int | None = None


def _parse_line(line: str) -> _HTSRecord | None:
    stripped = line.strip()
    header, separator, payload = stripped.partition(":")
    if not separator:
        if "wrist" in stripped.lower() or "landmarks" in stripped.lower():
            raise HTSProtocolError("HTS hand record is missing the ':' header separator")
        return None

    header_parts = [part.strip() for part in header.split("|")]
    label_parts = header_parts[0].lower().split()
    if not any(kind in label_parts for kind in ("wrist", "landmarks")):
        return None
    if len(label_parts) != 2 or label_parts[0] not in {"left", "right"}:
        raise HTSProtocolError(f"Invalid HTS hand record label: {header_parts[0]!r}")
    handedness = Handedness(label_parts[0])
    kind = label_parts[1]

    source_frame_id = None
    source_timestamp_ns = None
    for metadata in header_parts[1:]:
        key, equals, raw_value = metadata.partition("=")
        if not equals:
            continue
        key = key.strip().lower()
        if key not in {"f", "frame", "frame_id", "t", "ts", "timestamp"}:
            continue
        try:
            value = int(raw_value.strip())
        except ValueError as error:
            raise HTSProtocolError(f"Invalid HTS metadata value: {metadata!r}") from error
        if value < 0:
            raise HTSProtocolError(f"HTS metadata must be non-negative: {metadata!r}")
        if key in {"f", "frame", "frame_id"}:
            source_frame_id = value
        else:
            source_timestamp_ns = value
    if source_frame_id is None or source_timestamp_ns is None:
        raise HTSProtocolError(
            f"HTS {handedness.value} {kind} record requires both source frame ID 'f' "
            "and source timestamp 't'; enable HTS header/debug metadata"
        )

    values = []
    for part in payload.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as error:
            raise HTSProtocolError("HTS hand record contains a non-numeric payload value") from error
    return _HTSRecord(
        handedness=handedness,
        kind=kind,
        values=tuple(values),
        source_frame_id=source_frame_id,
        source_timestamp_ns=source_timestamp_ns,
    )


@dataclass
class _ClockAlignment:
    """Robustly map the HTS monotonic clock onto desktop monotonic time.

    Receipt time is capture time plus non-negative transport/scheduling delay.
    A bounded rolling affine fit tracks headset-clock drift, while a low
    intercept quantile rejects positive arrival jitter and occasional stalls.
    """

    source_origin_ns: int | None = None
    desktop_offset: float | None = None
    rate: float = 1.0
    maximum_drift_ppm: float = 5_000.0
    maximum_offset_step_seconds: float = 0.001
    rate_smoothing: float = 0.2
    refit_interval: int = 8
    observation_count: int = 0
    last_source_timestamp_ns: int | None = None
    last_desktop_timestamp: float | None = None
    observations: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=512)
    )

    def _refit(self) -> None:
        if len(self.observations) < 8:
            return
        values = np.asarray(self.observations, dtype=np.float64)
        source_seconds = values[:, 0]
        receipts = values[:, 1]
        if source_seconds[-1] - source_seconds[0] < 0.5:
            return

        centered_source = source_seconds - np.mean(source_seconds)
        centered_receipts = receipts - np.mean(receipts)
        denominator = float(np.dot(centered_source, centered_source))
        if denominator <= np.finfo(np.float64).eps:
            return
        rate = float(np.dot(centered_source, centered_receipts) / denominator)

        # Remove large scheduling/network stalls, then fit again. Ordinary
        # jitter remains zero-slope noise and does not bias the clock rate.
        intercepts = receipts - rate * source_seconds
        residuals = intercepts - np.median(intercepts)
        mad = float(np.median(np.abs(residuals)))
        if mad > np.finfo(np.float64).eps:
            keep = np.abs(residuals) <= 4.0 * 1.4826 * mad
            if np.count_nonzero(keep) >= 8:
                kept_source = source_seconds[keep]
                kept_receipts = receipts[keep]
                centered_source = kept_source - np.mean(kept_source)
                centered_receipts = kept_receipts - np.mean(kept_receipts)
                denominator = float(np.dot(centered_source, centered_source))
                if denominator > np.finfo(np.float64).eps:
                    rate = float(
                        np.dot(centered_source, centered_receipts) / denominator
                    )
                    source_seconds = kept_source
                    receipts = kept_receipts

        drift_bound = self.maximum_drift_ppm * 1e-6
        target_rate = float(np.clip(rate, 1.0 - drift_bound, 1.0 + drift_bound))
        self.rate += self.rate_smoothing * (target_rate - self.rate)
        # The lower envelope estimates the fixed clock offset plus the smallest
        # observed network delay without chasing a single extreme observation.
        target_offset = float(
            np.quantile(receipts - self.rate * source_seconds, 0.05)
        )
        assert self.desktop_offset is not None
        offset_step = float(
            np.clip(
                target_offset - self.desktop_offset,
                -self.maximum_offset_step_seconds,
                self.maximum_offset_step_seconds,
            )
        )
        self.desktop_offset += offset_step

    def to_desktop(self, source_timestamp_ns: int, receipt_timestamp: float) -> float:
        if (
            self.last_source_timestamp_ns is not None
            and source_timestamp_ns <= self.last_source_timestamp_ns
        ):
            raise HTSProtocolError(
                "HTS source clock must be strictly increasing within a hand stream"
            )
        if self.source_origin_ns is None:
            self.source_origin_ns = source_timestamp_ns
            self.desktop_offset = receipt_timestamp
        assert self.desktop_offset is not None
        source_seconds = (source_timestamp_ns - self.source_origin_ns) / 1e9
        self.observations.append((source_seconds, receipt_timestamp))
        self.observation_count += 1
        if self.observation_count % self.refit_interval == 0:
            self._refit()
        # A capture estimate may equal but must never exceed when it arrived.
        mapped = min(
            self.desktop_offset + self.rate * source_seconds,
            receipt_timestamp,
        )
        if self.last_desktop_timestamp is not None:
            lower_bound = float(np.nextafter(self.last_desktop_timestamp, np.inf))
            if mapped < lower_bound:
                if lower_bound > receipt_timestamp:
                    raise HTSProtocolError(
                        "HTS receipt timestamps leave no room for a strictly "
                        "increasing clock-alignment estimate"
                    )
                mapped = lower_bound
        self.last_source_timestamp_ns = source_timestamp_ns
        self.last_desktop_timestamp = mapped
        return mapped


@dataclass(frozen=True)
class HTSFrameDiagnostics:
    """Raw records and receive timing used to assemble one HTS ``HandFrame``."""

    frame_timestamp: float
    wrist_receipt_timestamp: float
    landmarks_receipt_timestamp: float
    source_frame_id: int | None
    source_timestamp_ns: int | None
    raw_wrist_position_unity: np.ndarray
    raw_wrist_quaternion_xyzw: np.ndarray
    raw_landmarks_unity: np.ndarray

    @property
    def pair_skew_seconds(self) -> float:
        return abs(self.wrist_receipt_timestamp - self.landmarks_receipt_timestamp)


@dataclass
class _PendingHandFrame:
    source_frame_id: int | None
    source_timestamp_ns: int | None
    receipt_timestamp: float = float("-inf")
    wrist_receipt_timestamp: float = float("-inf")
    landmarks_receipt_timestamp: float = float("-inf")
    raw_wrist_position_unity: np.ndarray | None = None
    raw_wrist_quaternion_xyzw: np.ndarray | None = None
    raw_landmarks_unity: np.ndarray | None = None
    wrist_position: np.ndarray | None = None
    wrist_quaternion_xyzw: np.ndarray | None = None
    landmarks_local: np.ndarray | None = None


_PairingKey = tuple[str, int | None, int | None]


@dataclass
class _HandState:
    pending: dict[_PairingKey, _PendingHandFrame] = field(default_factory=dict)
    latest: HandFrame | None = None
    latest_articulation: HandArticulationSample | None = None
    latest_wrist: WristPoseSample | None = None
    diagnostic_history: OrderedDict[float, HTSFrameDiagnostics] = field(default_factory=OrderedDict)
    unread_samples: deque[tuple[HandArticulationSample, WristPoseSample]] = field(
        default_factory=deque
    )
    maximum_buffered_samples: int = 256
    lossless_buffering: bool = False
    last_source_timestamp_ns: int | None = None
    clock_alignment: _ClockAlignment = field(default_factory=_ClockAlignment)

    def update_wrist(
        self,
        values: Iterable[float],
        *,
        handedness: Handedness = Handedness.RIGHT,
        receipt_timestamp: float | None = None,
        source_frame_id: int,
        source_timestamp_ns: int,
        pairing_key: _PairingKey,
    ) -> None:
        self.update(
            _HTSRecord(handedness, "wrist", tuple(values), source_frame_id, source_timestamp_ns),
            receipt_timestamp=time.monotonic() if receipt_timestamp is None else receipt_timestamp,
            pairing_key=pairing_key,
        )

    def update_landmarks(
        self,
        values: Iterable[float],
        *,
        handedness: Handedness = Handedness.RIGHT,
        receipt_timestamp: float | None = None,
        source_frame_id: int,
        source_timestamp_ns: int,
        pairing_key: _PairingKey,
    ) -> None:
        self.update(
            _HTSRecord(handedness, "landmarks", tuple(values), source_frame_id, source_timestamp_ns),
            receipt_timestamp=time.monotonic() if receipt_timestamp is None else receipt_timestamp,
            pairing_key=pairing_key,
        )

    def update(self, record: _HTSRecord, *, receipt_timestamp: float, pairing_key: _PairingKey) -> None:
        pending = self.pending.get(pairing_key)
        if pending is None:
            if self.pending:
                expected_key = next(iter(self.pending))
                raise HTSProtocolError(
                    f"HTS {record.handedness.value} {record.kind} record {pairing_key[1:]} does not match "
                    f"incomplete pair {expected_key[1:]}"
                )
            pending = _PendingHandFrame(record.source_frame_id, record.source_timestamp_ns)
            self.pending[pairing_key] = pending
        pending.receipt_timestamp = max(pending.receipt_timestamp, receipt_timestamp)

        array = np.asarray(record.values, dtype=np.float64)
        if record.kind == "wrist":
            if array.shape != (7,):
                raise HTSProtocolError(f"HTS wrist record must contain 7 values, got {array.size}")
            if pending.wrist_position is not None:
                raise HTSProtocolError(f"HTS pair {pairing_key[1:]} contains duplicate wrist records")
            pending.wrist_receipt_timestamp = receipt_timestamp
            pending.raw_wrist_position_unity = array[:3].copy()
            pending.raw_wrist_quaternion_xyzw = _normalize_quaternion(array[3:])
            pending.wrist_position = _UNITY_TO_RH @ array[:3]
            pending.wrist_quaternion_xyzw = _convert_quaternion(array[3:])
        else:
            if array.shape != (63,):
                raise HTSProtocolError(f"HTS landmark record must contain 63 values, got {array.size}")
            if pending.landmarks_local is not None:
                raise HTSProtocolError(f"HTS pair {pairing_key[1:]} contains duplicate landmark records")
            landmarks = array.reshape(21, 3)
            pending.landmarks_receipt_timestamp = receipt_timestamp
            pending.raw_landmarks_unity = landmarks.copy()
            pending.landmarks_local = (_UNITY_TO_RH @ landmarks.T).T

        self._publish_if_complete(pairing_key, record.handedness)

    def _publish_if_complete(self, pairing_key: _PairingKey, handedness: Handedness) -> None:
        pending = self.pending[pairing_key]
        if (
            pending.wrist_position is None
            or pending.wrist_quaternion_xyzw is None
            or pending.landmarks_local is None
            or pending.raw_wrist_position_unity is None
            or pending.raw_wrist_quaternion_xyzw is None
            or pending.raw_landmarks_unity is None
        ):
            return

        del self.pending[pairing_key]
        if np.max(np.linalg.norm(pending.landmarks_local - pending.landmarks_local[0], axis=1)) < 1e-6:
            return
        if (
            pending.source_timestamp_ns is not None
            and self.last_source_timestamp_ns is not None
            and pending.source_timestamp_ns <= self.last_source_timestamp_ns
        ):
            return

        timestamp = pending.receipt_timestamp
        if pending.source_timestamp_ns is not None:
            timestamp = self.clock_alignment.to_desktop(
                pending.source_timestamp_ns,
                pending.receipt_timestamp,
            )
        if self.latest is not None and timestamp <= self.latest.timestamp:
            # An affine refit may move the estimated offset backwards.  Keep
            # each hand stream strictly ordered while retaining the invariant
            # that capture cannot occur after receipt.
            timestamp = float(np.nextafter(self.latest.timestamp, np.inf))
            if timestamp > pending.receipt_timestamp:
                raise HTSProtocolError(
                    "HTS receipt timestamps leave no room for a strictly "
                    "increasing capture-time estimate"
                )

        landmarks_openxr = (_D_TO_OPENXR @ pending.landmarks_local.T).T
        local_joints = dict(
            zip(MEDIAPIPE_JOINT_NAMES, landmarks_openxr, strict=True)
        )
        articulation = HandArticulationSample(
            timestamp=timestamp,
            receipt_timestamp=pending.landmarks_receipt_timestamp,
            source_timestamp_ns=pending.source_timestamp_ns,
            source_frame_id=pending.source_frame_id,
            handedness=handedness,
            joint_positions=local_joints,
            source="hts",
            schema="mediapipe21",
            coordinate_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
        )
        wrist_openxr_quaternion = _matrix_to_quaternion(
            _quaternion_to_matrix(pending.wrist_quaternion_xyzw)
            @ _D_TO_OPENXR.T
        )
        wrist = WristPoseSample(
            timestamp=timestamp,
            receipt_timestamp=pending.wrist_receipt_timestamp,
            source_timestamp_ns=pending.source_timestamp_ns,
            source_frame_id=pending.source_frame_id,
            handedness=handedness,
            position=pending.wrist_position,
            quaternion_xyzw=wrist_openxr_quaternion,
            source="hts",
            reference_frame="tracking",
            anatomical_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
        )
        landmarks_world = (
            _rotate(pending.landmarks_local, pending.wrist_quaternion_xyzw) + pending.wrist_position
        )
        joints = dict(zip(MEDIAPIPE_JOINT_NAMES, landmarks_world, strict=True))
        frame = HandFrame(
            timestamp=timestamp,
            receipt_timestamp=pending.receipt_timestamp,
            source_timestamp_ns=pending.source_timestamp_ns,
            source_frame_id=pending.source_frame_id,
            handedness=handedness,
            joints=joints,
            wrist_position=pending.wrist_position,
            wrist_quaternion_xyzw=pending.wrist_quaternion_xyzw,
            source="hts",
        )
        diagnostics = HTSFrameDiagnostics(
            frame_timestamp=timestamp,
            wrist_receipt_timestamp=pending.wrist_receipt_timestamp,
            landmarks_receipt_timestamp=pending.landmarks_receipt_timestamp,
            source_frame_id=pending.source_frame_id,
            source_timestamp_ns=pending.source_timestamp_ns,
            raw_wrist_position_unity=pending.raw_wrist_position_unity.copy(),
            raw_wrist_quaternion_xyzw=pending.raw_wrist_quaternion_xyzw.copy(),
            raw_landmarks_unity=pending.raw_landmarks_unity.copy(),
        )
        if self.lossless_buffering:
            if len(self.unread_samples) >= self.maximum_buffered_samples:
                raise HTSBufferOverflowError(
                    f"HTS {handedness.value}-hand native sample buffer exceeded "
                    f"{self.maximum_buffered_samples} unread frames"
                )
            self.unread_samples.append((articulation, wrist))
        self.latest_articulation = articulation
        self.latest_wrist = wrist
        self.latest = frame
        self.diagnostic_history[timestamp] = diagnostics
        while len(self.diagnostic_history) > 512:
            self.diagnostic_history.popitem(last=False)
        if pending.source_timestamp_ns is not None:
            self.last_source_timestamp_ns = pending.source_timestamp_ns
            for key, older in tuple(self.pending.items()):
                if older.source_timestamp_ns is not None and older.source_timestamp_ns <= pending.source_timestamp_ns:
                    del self.pending[key]

    def frame(self, handedness: Handedness) -> HandFrame | None:
        if self.latest is not None and self.latest.handedness != handedness:
            raise ValueError(f"HTS state contains {self.latest.handedness.value} data, not {handedness.value}")
        return self.latest

    def articulation(self, handedness: Handedness) -> HandArticulationSample | None:
        if self.latest_articulation is not None and self.latest_articulation.handedness != handedness:
            raise ValueError(
                f"HTS state contains {self.latest_articulation.handedness.value} data, "
                f"not {handedness.value}"
            )
        return self.latest_articulation

    def wrist(self, handedness: Handedness) -> WristPoseSample | None:
        if self.latest_wrist is not None and self.latest_wrist.handedness != handedness:
            raise ValueError(
                f"HTS state contains {self.latest_wrist.handedness.value} data, not {handedness.value}"
            )
        return self.latest_wrist

    def diagnostics_for_timestamp(self, timestamp: float) -> HTSFrameDiagnostics | None:
        diagnostics = self.diagnostic_history.get(timestamp)
        if diagnostics is None:
            return None
        return HTSFrameDiagnostics(
            frame_timestamp=diagnostics.frame_timestamp,
            wrist_receipt_timestamp=diagnostics.wrist_receipt_timestamp,
            landmarks_receipt_timestamp=diagnostics.landmarks_receipt_timestamp,
            source_frame_id=diagnostics.source_frame_id,
            source_timestamp_ns=diagnostics.source_timestamp_ns,
            raw_wrist_position_unity=diagnostics.raw_wrist_position_unity.copy(),
            raw_wrist_quaternion_xyzw=diagnostics.raw_wrist_quaternion_xyzw.copy(),
            raw_landmarks_unity=diagnostics.raw_landmarks_unity.copy(),
        )


class HTSSource(HandTrackingSource):
    """Quest HTS UDP/TCP listener producing canonical hand frames."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        protocol: str = "udp",
        maximum_buffered_samples: int = 256,
    ) -> None:
        protocol = protocol.lower()
        if protocol not in {"udp", "tcp"}:
            raise ValueError(f"protocol must be 'udp' or 'tcp', got {protocol!r}")
        if maximum_buffered_samples <= 0:
            raise ValueError("maximum_buffered_samples must be positive")
        self.host = host
        self.port = int(port)
        self.protocol = protocol
        self._states = {
            Handedness.LEFT: _HandState(
                maximum_buffered_samples=maximum_buffered_samples,
            ),
            Handedness.RIGHT: _HandState(
                maximum_buffered_samples=maximum_buffered_samples,
            ),
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection_threads: list[threading.Thread] = []
        self._receiver_error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("HTSSource has already been started")
        self._thread = threading.Thread(target=self._run, name="dex-teleop-hts", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self.close()
            raise SourceUnavailableError(f"HTS listener did not bind to {self.protocol}://{self.host}:{self.port}")
        self.check_health()

    def read(self, handedness: Handedness) -> HandFrame | None:
        self.check_health()
        with self._lock:
            return self._states[Handedness(handedness)].frame(Handedness(handedness))

    def read_articulation(self, handedness: Handedness) -> HandArticulationSample | None:
        """Return HTS landmarks in their wrist-local MediaPipe-21 representation."""

        self.check_health()
        handedness = Handedness(handedness)
        with self._lock:
            return self._states[handedness].articulation(handedness)

    def read_wrist(self, handedness: Handedness) -> WristPoseSample | None:
        """Return the wrist component from the same receiver used by :meth:`read`."""

        self.check_health()
        handedness = Handedness(handedness)
        with self._lock:
            return self._states[handedness].wrist(handedness)

    def drain_hand_tracking(self, handedness: Handedness) -> HandTrackingSampleBatch:
        """Enable lossless buffering and atomically consume paired HTS samples.

        Latest-value clients never activate the bounded queue, preserving the
        legacy :meth:`read` contract without eventually overflowing.  The
        first drain returns the current latest pair, if one already exists;
        subsequent drains return every pair acquired since the prior drain.
        """

        self.check_health()
        handedness = Handedness(handedness)
        with self._lock:
            state = self._states[handedness]
            if state.lossless_buffering:
                samples = tuple(state.unread_samples)
            else:
                state.lossless_buffering = True
                samples = (
                    ((state.latest_articulation, state.latest_wrist),)
                    if state.latest_articulation is not None
                    and state.latest_wrist is not None
                    else ()
                )
            state.unread_samples.clear()
        return HandTrackingSampleBatch(
            articulations=tuple(articulation for articulation, _wrist in samples),
            wrists=tuple(wrist for _articulation, wrist in samples),
        )

    def diagnostics_for_frame(self, frame: HandFrame) -> HTSFrameDiagnostics | None:
        """Return a copy of the raw HTS records that produced ``frame``."""

        if frame.source != "hts":
            return None
        with self._lock:
            return self._states[frame.handedness].diagnostics_for_timestamp(frame.timestamp)

    def check_health(self) -> None:
        if self._receiver_error is not None:
            raise SourceUnavailableError(
                f"HTS {self.protocol.upper()} receiver failed on {self.host}:{self.port}: "
                f"{self._receiver_error}"
            ) from self._receiver_error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        for thread in self._connection_threads:
            if thread.is_alive():
                thread.join(timeout=0.5)

    def _pairing_key(self, record: _HTSRecord) -> _PairingKey:
        if record.source_frame_id is None or record.source_timestamp_ns is None:
            raise HTSProtocolError("HTS hand record is missing required source pairing metadata")
        return ("source", record.source_frame_id, record.source_timestamp_ns)

    def _handle_payload(
        self,
        payload: str,
        receipt_timestamp: float | None = None,
        *,
        require_complete_pairs: bool = False,
    ) -> tuple[_HTSRecord, ...]:
        receipt_timestamp = time.monotonic() if receipt_timestamp is None else receipt_timestamp
        records = [record for line in payload.splitlines() if (record := _parse_line(line)) is not None]
        if require_complete_pairs:
            kinds_by_pair: dict[tuple[Handedness, int, int], list[str]] = {}
            for record in records:
                assert record.source_frame_id is not None and record.source_timestamp_ns is not None
                pair = (record.handedness, record.source_frame_id, record.source_timestamp_ns)
                kinds_by_pair.setdefault(pair, []).append(record.kind)
            for pair, kinds in kinds_by_pair.items():
                if len(kinds) != 2 or set(kinds) != {"wrist", "landmarks"}:
                    raise HTSProtocolError(
                        f"HTS UDP datagram for {pair[0].value} pair {pair[1:]} must contain exactly one "
                        f"wrist and one landmarks record; received {kinds}"
                    )
        try:
            with self._lock:
                for record in records:
                    self._states[record.handedness].update(
                        record,
                        receipt_timestamp=receipt_timestamp,
                        pairing_key=self._pairing_key(record),
                    )
        except HTSBufferOverflowError as error:
            # Direct protocol injection (used by diagnostics/tests) does not
            # pass through ``_run``. Poison source health here as well so an
            # overflow can never be mistaken for successful acquisition.
            self._receiver_error = error
            raise
        return tuple(records)

    def _handle_line(self, line: str, receipt_timestamp: float | None = None) -> None:
        self._handle_payload(line, receipt_timestamp=receipt_timestamp)

    def _run(self) -> None:
        try:
            if self.protocol == "udp":
                self._run_udp()
            else:
                self._run_tcp()
        except Exception as error:
            self._receiver_error = error
            self._ready.set()

    def _run_udp(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.settimeout(0.5)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    payload, _ = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                receipt_timestamp = time.monotonic()
                self._handle_payload(
                    payload.decode("utf-8"),
                    receipt_timestamp,
                    require_complete_pairs=True,
                )

    def _run_tcp(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(target=self._handle_connection, args=(connection,), daemon=True)
                self._connection_threads.append(thread)
                thread.start()

    def _handle_connection(self, connection: socket.socket) -> None:
        pending_pairs: dict[tuple[Handedness, int, int], set[str]] = {}

        def track_records(records: tuple[_HTSRecord, ...]) -> None:
            for record in records:
                assert record.source_frame_id is not None and record.source_timestamp_ns is not None
                pair = (record.handedness, record.source_frame_id, record.source_timestamp_ns)
                kinds = pending_pairs.setdefault(pair, set())
                kinds.add(record.kind)
                if kinds == {"wrist", "landmarks"}:
                    del pending_pairs[pair]

        def incomplete_pair_error(reason: str) -> HTSProtocolError:
            pairs = ", ".join(
                f"{side.value} ({frame_id}, {timestamp_ns}) has {sorted(kinds)}"
                for (side, frame_id, timestamp_ns), kinds in pending_pairs.items()
            )
            return HTSProtocolError(f"HTS TCP connection {reason} with incomplete pair(s): {pairs}")

        try:
            with connection:
                connection.settimeout(0.5)
                buffer = ""
                while not self._stop.is_set():
                    try:
                        payload = connection.recv(4096)
                    except socket.timeout:
                        if buffer.strip():
                            raise HTSProtocolError("HTS TCP connection stalled with an unterminated hand record")
                        if pending_pairs:
                            raise incomplete_pair_error("stalled")
                        continue
                    receipt_timestamp = time.monotonic()
                    if not payload:
                        if buffer.strip():
                            raise HTSProtocolError("HTS TCP connection closed with an unterminated hand record")
                        if pending_pairs:
                            raise incomplete_pair_error("closed")
                        return
                    buffer += payload.decode("utf-8")
                    complete_lines = []
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        complete_lines.append(line)
                    if complete_lines:
                        track_records(self._handle_payload("\n".join(complete_lines), receipt_timestamp))
        except Exception as error:
            self._receiver_error = error

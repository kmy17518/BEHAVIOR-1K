"""Meta Quest Hand Tracking Streamer (HTS) source.

Adapted from AnyDexRetarget's ``example/input/quest3.py`` (MIT License,
Copyright (c) 2025 Shiquan Qiu). See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from dex_teleop.tracking.base import HandTrackingSource, SourceUnavailableError
from dex_teleop.types import HandFrame, Handedness, MEDIAPIPE_JOINT_NAMES

LOGGER = logging.getLogger(__name__)


class HTSProtocolError(ValueError):
    """Raised when HTS hand data does not satisfy the expected wire contract."""

# Unity LH (x right, y up, z forward) -> RH (x forward, y left, z up).
_UNITY_TO_RH = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
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
    """Map one HTS monotonic clock onto the desktop monotonic clock."""

    source_origin_ns: int | None = None
    desktop_origin: float | None = None

    def to_desktop(self, source_timestamp_ns: int, receipt_timestamp: float) -> float:
        if self.source_origin_ns is None:
            self.source_origin_ns = source_timestamp_ns
            self.desktop_origin = receipt_timestamp
        assert self.desktop_origin is not None
        return self.desktop_origin + (source_timestamp_ns - self.source_origin_ns) / 1e9


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
    diagnostic_history: OrderedDict[float, HTSFrameDiagnostics] = field(default_factory=OrderedDict)
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

        landmarks_world = (
            _rotate(pending.landmarks_local, pending.wrist_quaternion_xyzw) + pending.wrist_position
        )
        joints = dict(zip(MEDIAPIPE_JOINT_NAMES, landmarks_world, strict=True))
        self.latest = HandFrame(
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
        self.diagnostic_history[timestamp] = HTSFrameDiagnostics(
            frame_timestamp=timestamp,
            wrist_receipt_timestamp=pending.wrist_receipt_timestamp,
            landmarks_receipt_timestamp=pending.landmarks_receipt_timestamp,
            source_frame_id=pending.source_frame_id,
            source_timestamp_ns=pending.source_timestamp_ns,
            raw_wrist_position_unity=pending.raw_wrist_position_unity.copy(),
            raw_wrist_quaternion_xyzw=pending.raw_wrist_quaternion_xyzw.copy(),
            raw_landmarks_unity=pending.raw_landmarks_unity.copy(),
        )
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

    def __init__(self, host: str = "0.0.0.0", port: int = 9000, protocol: str = "udp") -> None:
        protocol = protocol.lower()
        if protocol not in {"udp", "tcp"}:
            raise ValueError(f"protocol must be 'udp' or 'tcp', got {protocol!r}")
        self.host = host
        self.port = int(port)
        self.protocol = protocol
        clock_alignment = _ClockAlignment()
        self._states = {
            Handedness.LEFT: _HandState(clock_alignment=clock_alignment),
            Handedness.RIGHT: _HandState(clock_alignment=clock_alignment),
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
        with self._lock:
            for record in records:
                self._states[record.handedness].update(
                    record,
                    receipt_timestamp=receipt_timestamp,
                    pairing_key=self._pairing_key(record),
                )
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

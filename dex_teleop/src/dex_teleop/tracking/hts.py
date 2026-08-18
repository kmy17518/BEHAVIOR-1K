"""Meta Quest Hand Tracking Streamer (HTS) source.

Adapted from AnyDexRetarget's ``example/input/quest3.py`` (MIT License,
Copyright (c) 2025 Shiquan Qiu). See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dex_teleop.tracking.base import HandTrackingSource, SourceUnavailableError
from dex_teleop.types import HandFrame, Handedness, MEDIAPIPE_JOINT_NAMES

LOGGER = logging.getLogger(__name__)

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


def _parse_line(line: str) -> tuple[Handedness, str, tuple[float, ...]] | None:
    parts = [part.strip() for part in line.split(",")]
    if not parts:
        return None
    label = parts[0].lower()
    if "wrist" not in label and "landmarks" not in label:
        return None
    if "right" in label:
        handedness = Handedness.RIGHT
    elif "left" in label:
        handedness = Handedness.LEFT
    else:
        return None
    values = []
    for part in parts[1:]:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return None
    return handedness, "wrist" if "wrist" in label else "landmarks", tuple(values)


@dataclass
class _HandState:
    wrist_position: np.ndarray | None = None
    wrist_quaternion_xyzw: np.ndarray | None = None
    landmarks_local: np.ndarray | None = None
    wrist_timestamp: float = 0.0
    landmarks_timestamp: float = 0.0

    def update_wrist(self, values: Iterable[float]) -> None:
        array = np.asarray(tuple(values), dtype=np.float64)
        if array.shape != (7,):
            raise ValueError(f"HTS wrist record must contain 7 values, got {array.size}")
        self.wrist_position = _UNITY_TO_RH @ array[:3]
        self.wrist_quaternion_xyzw = _convert_quaternion(array[3:])
        self.wrist_timestamp = time.monotonic()

    def update_landmarks(self, values: Iterable[float]) -> None:
        array = np.asarray(tuple(values), dtype=np.float64)
        if array.shape != (63,):
            raise ValueError(f"HTS landmark record must contain 63 values, got {array.size}")
        landmarks = array.reshape(21, 3)
        self.landmarks_local = (_UNITY_TO_RH @ landmarks.T).T
        self.landmarks_timestamp = time.monotonic()

    def frame(self, handedness: Handedness) -> HandFrame | None:
        if self.wrist_position is None or self.wrist_quaternion_xyzw is None or self.landmarks_local is None:
            return None
        if np.max(np.linalg.norm(self.landmarks_local - self.landmarks_local[0], axis=1)) < 1e-6:
            return None
        landmarks_world = _rotate(self.landmarks_local, self.wrist_quaternion_xyzw) + self.wrist_position
        joints = dict(zip(MEDIAPIPE_JOINT_NAMES, landmarks_world, strict=True))
        return HandFrame(
            # A fused frame is only as fresh as its older constituent record.
            timestamp=min(self.wrist_timestamp, self.landmarks_timestamp),
            handedness=handedness,
            joints=joints,
            wrist_position=self.wrist_position,
            wrist_quaternion_xyzw=self.wrist_quaternion_xyzw,
            source="hts",
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
        self._states = {Handedness.LEFT: _HandState(), Handedness.RIGHT: _HandState()}
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

    def _handle_line(self, line: str) -> None:
        parsed = _parse_line(line)
        if parsed is None:
            return
        handedness, kind, values = parsed
        try:
            with self._lock:
                if kind == "wrist":
                    self._states[handedness].update_wrist(values)
                else:
                    self._states[handedness].update_landmarks(values)
        except ValueError as error:
            LOGGER.warning("Discarding malformed HTS record: %s", error)

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
                for line in payload.decode("utf-8", errors="ignore").splitlines():
                    self._handle_line(line)

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
        with connection:
            connection.settimeout(0.5)
            buffer = ""
            while not self._stop.is_set():
                try:
                    payload = connection.recv(4096)
                except socket.timeout:
                    continue
                if not payload:
                    return
                buffer += payload.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    self._handle_line(line)

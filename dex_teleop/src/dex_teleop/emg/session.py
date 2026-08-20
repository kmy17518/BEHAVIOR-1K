"""Launcher-side lifecycle and live-preview client for the EMG sidecar."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from dex_teleop.emg.types import DecodedHandPreview, EmgPreview


class EmgSessionError(RuntimeError):
    """The EMG sidecar failed or became unavailable."""


def emg_staging_path(output_path: str | Path, *, process_id: int | None = None) -> Path:
    output_path = Path(output_path)
    process_id = os.getpid() if process_id is None else process_id
    return output_path.with_name(f".{output_path.name}.{process_id}.emg.in_progress.hdf5")


class EmgSession:
    """Own a clean Python sidecar and expose its bounded live waveform."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        device: str | None = None,
        adapter: str | None = None,
        sdk_path: str | Path | None = None,
        scan_ms: int = 5000,
        preview_capacity: int = 1000,
        visualize_decoder: bool = False,
        emg2pose_root: str | Path | None = None,
        emg2pose_checkpoint: str | Path | None = None,
        decoder_device: str = "auto",
        decoder_hand: str = "right",
        decoder_inference_hz: float = 5.0,
        filter_hpf: bool = True,
        filter_lpf: bool = True,
        filter_notch: str = "60",
    ) -> None:
        if scan_ms <= 0 or preview_capacity <= 0:
            raise ValueError("EMG scan period and preview capacity must be positive")
        self.output_path = Path(output_path).expanduser().resolve()
        self.device = device
        self.adapter = adapter
        self.sdk_path = None if sdk_path is None else Path(sdk_path).expanduser().resolve()
        self.scan_ms = int(scan_ms)
        self.preview_capacity = int(preview_capacity)
        self.visualize_decoder = bool(visualize_decoder)
        self.emg2pose_root = None if emg2pose_root is None else Path(emg2pose_root).expanduser().resolve()
        self.emg2pose_checkpoint = (
            None if emg2pose_checkpoint is None else Path(emg2pose_checkpoint).expanduser().resolve()
        )
        self.decoder_device = decoder_device
        self.decoder_hand = decoder_hand
        self.decoder_inference_hz = float(decoder_inference_hz)
        self.filter_hpf = bool(filter_hpf)
        self.filter_lpf = bool(filter_lpf)
        self.filter_notch = filter_notch
        if self.decoder_inference_hz <= 0:
            raise ValueError("EMG decoder inference rate must be positive")
        if self.filter_notch not in {"off", "50", "60", "both"}:
            raise ValueError(f"Unsupported EMG notch filter selection: {self.filter_notch}")
        self._socket: socket.socket | None = None
        self._receiver: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._preview_batches: deque[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = deque()
        self._preview_count = 0
        self._latest_impedance_ohm = np.empty(0, dtype=np.float32)
        self._decoder_vertices = np.empty((0, 3), dtype=np.float32)
        self._decoder_triangles = np.empty((0, 3), dtype=np.int32)
        self._decoder_joint_angles = np.empty(0, dtype=np.float32)
        self._decoder_sequence = 0
        self._decoder_source_sample_index: int | None = None
        self._decoder_inference_ms: float | None = None
        self._decoder_status = "disabled" if not visualize_decoder else "loading decoder"
        self._metadata: dict = {}
        self._status = "not started"
        self._error: str | None = None
        self._ready = False
        self._closed = False

    @property
    def metadata(self) -> dict:
        with self._condition:
            return dict(self._metadata)

    @property
    def channel_count(self) -> int:
        return int(self.metadata.get("channel_count", 0))

    @property
    def sample_rate_hz(self) -> float:
        return float(self.metadata.get("sample_rate_hz", 0.0))

    @property
    def status(self) -> str:
        with self._condition:
            return self._status

    def start(self, timeout: float = 45.0) -> None:
        if timeout <= 0:
            raise ValueError("EMG startup timeout must be positive")
        if self._process is not None:
            raise RuntimeError("EMG session has already been started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver_socket.bind(("127.0.0.1", 0))
        receiver_socket.settimeout(0.2)
        self._socket = receiver_socket
        self._receiver = threading.Thread(target=self._receive_loop, name="dex-teleop-emg-preview", daemon=True)
        self._receiver.start()

        command = [
            sys.executable,
            "-u",
            "-m",
            "dex_teleop.emg.sidecar",
            "--output",
            str(self.output_path),
            "--preview-port",
            str(receiver_socket.getsockname()[1]),
            "--scan-ms",
            str(self.scan_ms),
            "--filter-hpf",
            "on" if self.filter_hpf else "off",
            "--filter-lpf",
            "on" if self.filter_lpf else "off",
            "--filter-notch",
            self.filter_notch,
        ]
        if self.device:
            command.extend(("--device", self.device))
        if self.adapter:
            command.extend(("--adapter", self.adapter))
        if self.sdk_path is not None:
            command.extend(("--sdk-path", str(self.sdk_path)))
        if self.visualize_decoder:
            command.extend(("--visualize-decoder", "--decoder-device", self.decoder_device))
            command.extend(("--decoder-hand", self.decoder_hand))
            command.extend(("--decoder-inference-hz", str(self.decoder_inference_hz)))
            if self.emg2pose_root is not None:
                command.extend(("--emg2pose-root", str(self.emg2pose_root)))
            if self.emg2pose_checkpoint is not None:
                command.extend(("--emg2pose-checkpoint", str(self.emg2pose_checkpoint)))
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source_root if not python_path else os.pathsep.join((source_root, python_path))
        self._process = subprocess.Popen(command, env=environment)

        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._ready and self._error is None:
                return_code = self._process.poll()
                if return_code is not None:
                    self._error = f"EMG sidecar exited with status {return_code} during startup"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._error = f"No EMG samples arrived within {timeout:.1f}s"
                    break
                self._condition.wait(timeout=min(remaining, 0.2))
        if self._error is not None:
            error = self._error
            self.close()
            raise EmgSessionError(error)

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload, _address = self._socket.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                message = json.loads(payload.decode("utf-8"))
                self._handle_message(message)
            except Exception as error:
                with self._condition:
                    self._error = f"Invalid EMG sidecar message: {error}"
                    self._condition.notify_all()

    def _handle_message(self, message: dict) -> None:
        message_type = message.get("type")
        with self._condition:
            if message_type in ("status", "ready"):
                self._status = str(message.get("status", message_type))
                self._metadata.update({key: value for key, value in message.items() if key not in {"type", "status"}})
                if message_type == "ready":
                    self._ready = True
            elif message_type == "error":
                self._status = "error"
                self._error = str(message.get("message", "unknown EMG sidecar error"))
            elif message_type == "samples":
                indices = np.asarray(message["sample_index"], dtype=np.int64).reshape(-1)
                signal_uv = np.asarray(message["signal_uv"], dtype=np.float32)
                is_lost = np.asarray(message["is_lost"], dtype=np.bool_).reshape(-1)
                impedance_ohm = np.asarray(message["impedance_ohm"], dtype=np.float32).reshape(-1)
                if signal_uv.ndim != 2 or signal_uv.shape[0] != len(indices) or len(is_lost) != len(indices):
                    raise ValueError("EMG preview arrays have inconsistent shapes")
                if impedance_ohm.shape != (signal_uv.shape[1],):
                    raise ValueError("EMG preview impedance does not match the channel count")
                receive_ns = int(message["receive_monotonic_ns"])
                self._preview_batches.append((indices, signal_uv, is_lost, receive_ns))
                self._preview_count += len(indices)
                self._latest_impedance_ohm = impedance_ohm.copy()
                while self._preview_batches and self._preview_count > self.preview_capacity:
                    old = self._preview_batches.popleft()
                    self._preview_count -= len(old[0])
            elif message_type == "decoder_geometry":
                triangles = np.asarray(message["triangles"], dtype=np.int32)
                if triangles.ndim != 2 or triangles.shape[1:] != (3,):
                    raise ValueError("Decoded hand triangles must have shape (faces, 3)")
                self._decoder_triangles = triangles.copy()
                self._decoder_status = str(message.get("status", "buffering model window"))
            elif message_type == "decoder_frame":
                vertices = np.asarray(message["vertices"], dtype=np.float32)
                joint_angles = np.asarray(message.get("joint_angles", []), dtype=np.float32).reshape(-1)
                if vertices.ndim != 2 or vertices.shape[1:] != (3,):
                    raise ValueError("Decoded hand vertices must have shape (vertices, 3)")
                if joint_angles.shape not in {(0,), (20,)}:
                    raise ValueError("Decoded hand pose must contain 20 joint angles")
                self._decoder_vertices = vertices.copy()
                self._decoder_joint_angles = joint_angles.copy()
                self._decoder_sequence = int(message.get("sequence", 0))
                source_index = message.get("source_sample_index")
                self._decoder_source_sample_index = None if source_index is None else int(source_index)
                inference_ms = message.get("inference_ms")
                self._decoder_inference_ms = None if inference_ms is None else float(inference_ms)
                self._decoder_status = str(message.get("status", "live"))
            elif message_type == "decoder_status":
                self._decoder_status = str(message.get("status", "decoder status unavailable"))
            elif message_type == "decoder_error":
                self._decoder_status = f"error: {message.get('message', 'unknown decoder error')}"
            self._condition.notify_all()

    def preview(self, max_samples: int | None = None) -> EmgPreview:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("EMG preview sample limit must be positive")
        with self._condition:
            batches = list(self._preview_batches)
            status = self._status
            channel_count = int(self._metadata.get("channel_count", 0))
            impedance_ohm = self._latest_impedance_ohm.copy()
        if not batches:
            if impedance_ohm.shape != (channel_count,):
                impedance_ohm = np.full(channel_count, np.nan, dtype=np.float32)
            return EmgPreview(
                sample_index=np.empty(0, dtype=np.int64),
                signal_uv=np.empty((0, channel_count), dtype=np.float32),
                is_lost=np.empty(0, dtype=np.bool_),
                impedance_ohm=impedance_ohm,
                receive_monotonic_ns=None,
                status=status,
            )
        indices = np.concatenate([batch[0] for batch in batches])
        signal_uv = np.concatenate([batch[1] for batch in batches], axis=0)
        is_lost = np.concatenate([batch[2] for batch in batches])
        if max_samples is not None and len(indices) > max_samples:
            indices = indices[-max_samples:]
            signal_uv = signal_uv[-max_samples:]
            is_lost = is_lost[-max_samples:]
        return EmgPreview(
            sample_index=indices,
            signal_uv=signal_uv,
            is_lost=is_lost,
            impedance_ohm=impedance_ohm,
            receive_monotonic_ns=batches[-1][3],
            status=status,
        )

    def check_health(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise EmgSessionError(error)
        if self._process is not None and not self._closed:
            return_code = self._process.poll()
            if return_code is not None:
                raise EmgSessionError(f"EMG sidecar exited unexpectedly with status {return_code}")

    def decoded_hand(self) -> DecodedHandPreview:
        """Return a copied decoder snapshot safe for Kit's main thread."""

        with self._condition:
            return DecodedHandPreview(
                sequence=self._decoder_sequence,
                source_sample_index=self._decoder_source_sample_index,
                joint_angles=self._decoder_joint_angles,
                vertices=self._decoder_vertices,
                triangles=self._decoder_triangles,
                inference_ms=self._decoder_inference_ms,
                status=self._decoder_status,
            )

    def close(self, timeout: float = 15.0) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._receiver is not None and self._receiver.is_alive():
            self._receiver.join(timeout=1.0)

    def remove_staging_file(self) -> None:
        self.output_path.unlink(missing_ok=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()

"""VEMG2Pose inference fed by the existing Synchroni acquisition stream."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np


MODEL_SAMPLE_RATE_HZ = 2_000.0
MODEL_CHANNEL_COUNT = 16
MODEL_WINDOW_SAMPLES = 11_790
DEFAULT_INFERENCE_HZ = 5.0


def default_emg2pose_root() -> Path:
    """Return the sibling checkout used by this repository's EMG setup."""

    repository_root = Path(__file__).resolve().parents[4]
    return repository_root.parent / "emg2pose"


def required_source_samples(source_rate_hz: float) -> int:
    if source_rate_hz <= 0:
        raise ValueError("EMG source rate must be positive")
    duration_s = (MODEL_WINDOW_SAMPLES - 1) / MODEL_SAMPLE_RATE_HZ
    return math.ceil(duration_s * source_rate_hz) + 1


def interpolate_emg_channels(samples: np.ndarray, output_channels: int = MODEL_CHANNEL_COUNT) -> np.ndarray:
    """Linearly map the wristband electrodes to the checkpoint's 16 channels."""

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2:
        raise ValueError(f"Expected a 2D (time, channels) array, got {samples.shape}")
    input_channels = samples.shape[1]
    if input_channels < 2 or output_channels < 2:
        raise ValueError("At least two input and output EMG channels are required")
    if input_channels == output_channels:
        return samples.copy()

    positions = np.linspace(0.0, input_channels - 1, output_channels, dtype=np.float32)
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, input_channels - 1)
    weight = positions - lower
    return np.asarray(samples[:, lower] * (1.0 - weight) + samples[:, upper] * weight, dtype=np.float32)


def resample_emg_time(
    samples: np.ndarray,
    source_rate_hz: float,
    output_samples: int = MODEL_WINDOW_SAMPLES,
    target_rate_hz: float = MODEL_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Linearly resample a source window and align the output to its final sample."""

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2:
        raise ValueError(f"Expected a 2D (time, channels) array, got {samples.shape}")
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("EMG sample rates must be positive")
    if output_samples < 2:
        raise ValueError("At least two output samples are required")

    source_time = np.arange(samples.shape[0], dtype=np.float64) / source_rate_hz
    target_duration_s = (output_samples - 1) / target_rate_hz
    target_time = source_time[-1] - target_duration_s + np.arange(output_samples, dtype=np.float64) / target_rate_hz
    if target_time[0] < source_time[0] - 1e-9:
        raise ValueError(f"Need at least {required_source_samples(source_rate_hz)} source samples, got {len(samples)}")
    return np.asarray(
        np.column_stack(
            [np.interp(target_time, source_time, samples[:, channel]) for channel in range(samples.shape[1])]
        ),
        dtype=np.float32,
    )


def preprocess_emg(samples: np.ndarray, source_rate_hz: float) -> np.ndarray:
    """Match ``live_emg2pose.py`` preprocessing and return (16, 11790)."""

    channel_interpolated = interpolate_emg_channels(samples)
    time_interpolated = resample_emg_time(channel_interpolated, source_rate_hz)
    return np.ascontiguousarray(time_interpolated.T, dtype=np.float32)


@dataclass(frozen=True)
class DecodedMeshFrame:
    sequence: int
    source_sample_index: int
    joint_angles: np.ndarray
    vertices: np.ndarray
    inference_ms: float


class Emg2PoseDecoder:
    """Load the regression checkpoint and skin its UmeTrack hand profile."""

    def __init__(
        self,
        emg2pose_root: str | Path,
        checkpoint: str | Path | None = None,
        *,
        device_name: str = "auto",
        hand: str = "right",
    ) -> None:
        emg2pose_root = Path(emg2pose_root).expanduser().resolve()
        checkpoint = (
            emg2pose_root / "checkpoints" / "regression_vemg2pose.ckpt"
            if checkpoint is None
            else Path(checkpoint).expanduser().resolve()
        )
        if not emg2pose_root.is_dir():
            raise RuntimeError(f"emg2pose checkout not found: {emg2pose_root}")
        if not checkpoint.is_file():
            raise RuntimeError(f"emg2pose checkpoint not found: {checkpoint}")
        for path in reversed((emg2pose_root, emg2pose_root / "emg2pose" / "UmeTrack")):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        try:
            import torch
            from hydra.utils import instantiate
            from emg2pose.visualization import load_default_hand_model, mirror_profile, skin_vertices_np
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"Missing emg2pose dependency {error.name!r} in behavior_dex; install the emg2pose dependencies"
            ) from error

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested decoder device {device_name!r}, but CUDA is unavailable")
        if hand not in {"left", "right"}:
            raise ValueError(f"Unsupported decoded hand: {hand}")

        try:
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint_data = torch.load(checkpoint, map_location="cpu")
        network_conf = checkpoint_data.get("hyper_parameters", {}).get("network_conf")
        if network_conf is None:
            raise RuntimeError(f"Checkpoint has no network_conf: {checkpoint}")
        model = instantiate(network_conf, _convert_="all")
        model_state = {
            key.removeprefix("model."): value
            for key, value in checkpoint_data["state_dict"].items()
            if key.startswith("model.")
        }
        if not model_state:
            raise RuntimeError(f"Checkpoint contains no emg2pose model weights: {checkpoint}")
        model.load_state_dict(model_state, strict=True)

        self._torch = torch
        self._device = torch.device(device_name)
        self._model = model.to(self._device).eval()
        if self._model.left_context >= MODEL_WINDOW_SAMPLES:
            raise RuntimeError(
                f"Model context ({self._model.left_context}) exceeds input window ({MODEL_WINDOW_SAMPLES})"
            )
        self._skin_vertices = skin_vertices_np
        self._profile = load_default_hand_model()
        if hand == "left":
            self._profile = mirror_profile(self._profile)
        self.triangles = np.asarray(self._profile.mesh_triangles, dtype=np.int32)
        self.neutral_vertices = self.skin(np.zeros(20, dtype=np.float32))
        self.checkpoint = checkpoint
        self.device_name = str(self._device)
        self.left_context = int(self._model.left_context)

    def decode(self, emg: np.ndarray) -> tuple[np.ndarray, float]:
        if emg.shape != (MODEL_CHANNEL_COUNT, MODEL_WINDOW_SAMPLES):
            raise ValueError(
                f"Expected decoder input {(MODEL_CHANNEL_COUNT, MODEL_WINDOW_SAMPLES)}, got {emg.shape}"
            )
        torch = self._torch
        emg_tensor = torch.from_numpy(emg).unsqueeze(0).to(self._device)
        batch = {
            "emg": emg_tensor,
            "joint_angles": torch.zeros(
                (1, self._model.out_channels, MODEL_WINDOW_SAMPLES),
                dtype=emg_tensor.dtype,
                device=self._device,
            ),
            "no_ik_failure": torch.ones((1, MODEL_WINDOW_SAMPLES), dtype=torch.bool, device=self._device),
        }
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            predictions, _, _ = self._model(batch, provide_initial_pos=False)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        inference_ms = (time.perf_counter() - started) * 1_000.0
        pose = predictions[0, :, -1].detach().cpu().numpy().astype(np.float32, copy=False)
        if pose.shape != (20,) or not np.isfinite(pose).all():
            raise RuntimeError(f"Decoder produced an invalid pose with shape {pose.shape}")
        return pose, inference_ms

    def skin(self, pose: np.ndarray) -> np.ndarray:
        vertices = np.asarray(self._skin_vertices(self._profile, np.asarray(pose, dtype=np.float32)), dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise RuntimeError(f"Decoder produced invalid hand vertices with shape {vertices.shape}")
        return vertices


class DecoderRunner:
    """Maintain a native-rate rolling window and run one inference at a time."""

    def __init__(
        self,
        decoder: Emg2PoseDecoder,
        source_rate_hz: float,
        *,
        inference_hz: float = DEFAULT_INFERENCE_HZ,
    ) -> None:
        if inference_hz <= 0:
            raise ValueError("Decoder inference rate must be positive")
        self.decoder = decoder
        self.source_rate_hz = float(source_rate_hz)
        self.required_samples = required_source_samples(self.source_rate_hz)
        capacity = self.required_samples + math.ceil(2.0 * self.source_rate_hz)
        self._samples: deque[tuple[float, ...]] = deque(maxlen=capacity)
        self._last_sample_index = -1
        self.inference_hz = float(inference_hz)
        self._source_samples_per_inference = self.source_rate_hz / self.inference_hz
        self._next_submit_sample_index: float | None = None
        self._sequence = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emg2pose")
        self._future: Future[DecodedMeshFrame] | None = None

    @property
    def buffered_samples(self) -> int:
        return len(self._samples)

    def append(self, samples: np.ndarray, last_sample_index: int) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2:
            raise ValueError(f"Expected decoder samples with shape (time, channels), got {samples.shape}")
        self._samples.extend(map(tuple, samples))
        self._last_sample_index = int(last_sample_index)

    def poll(self) -> DecodedMeshFrame | None:
        result = None
        if self._future is not None and self._future.done():
            future = self._future
            self._future = None
            result = future.result()

        due = (
            self._next_submit_sample_index is None
            or self._last_sample_index >= self._next_submit_sample_index
        )
        if self._future is None and len(self._samples) >= self.required_samples and due:
            self._sequence += 1
            sequence = self._sequence
            source_sample_index = self._last_sample_index
            samples = np.asarray(self._samples, dtype=np.float32)[-self.required_samples :].copy()
            self._future = self._executor.submit(self._infer, sequence, source_sample_index, samples)
            if self._next_submit_sample_index is None:
                self._next_submit_sample_index = source_sample_index + self._source_samples_per_inference
            else:
                while self._next_submit_sample_index <= source_sample_index:
                    self._next_submit_sample_index += self._source_samples_per_inference
        return result

    def _infer(self, sequence: int, source_sample_index: int, samples: np.ndarray) -> DecodedMeshFrame:
        pose, inference_ms = self.decoder.decode(preprocess_emg(samples, self.source_rate_hz))
        return DecodedMeshFrame(
            sequence=sequence,
            source_sample_index=source_sample_index,
            joint_angles=pose,
            vertices=self.decoder.skin(pose),
            inference_ms=inference_ms,
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

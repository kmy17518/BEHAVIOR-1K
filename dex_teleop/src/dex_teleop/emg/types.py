"""Value objects shared by the EMG sidecar, recorder, and UI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmgBatch:
    """One copied SDK callback, with a single desktop receive timestamp."""

    receive_monotonic_ns: int
    sample_index: np.ndarray
    sdk_timestamp_ms: np.ndarray
    signal_uv: np.ndarray
    raw_adc: np.ndarray
    is_lost: np.ndarray
    impedance_ohm: np.ndarray
    saturation: np.ndarray

    def __post_init__(self) -> None:
        sample_index = np.asarray(self.sample_index, dtype=np.int64).reshape(-1).copy()
        sdk_timestamp_ms = np.asarray(self.sdk_timestamp_ms, dtype=np.int64).reshape(-1).copy()
        signal_uv = np.asarray(self.signal_uv, dtype=np.float32).copy()
        raw_adc = np.asarray(self.raw_adc, dtype=np.int32).copy()
        is_lost = np.asarray(self.is_lost, dtype=np.bool_).reshape(-1).copy()
        impedance_ohm = np.asarray(self.impedance_ohm, dtype=np.float32).reshape(-1).copy()
        saturation = np.asarray(self.saturation, dtype=np.float32).reshape(-1).copy()

        if self.receive_monotonic_ns < 0:
            raise ValueError("EMG receive timestamp must be non-negative")
        if signal_uv.ndim != 2 or signal_uv.shape[0] == 0 or signal_uv.shape[1] == 0:
            raise ValueError("EMG signal must have shape (samples, channels)")
        if raw_adc.shape != signal_uv.shape:
            raise ValueError("EMG raw ADC and converted signal shapes differ")
        sample_count, channel_count = signal_uv.shape
        if sample_index.shape != (sample_count,):
            raise ValueError("EMG sample indices do not match the signal length")
        if sdk_timestamp_ms.shape != (sample_count,):
            raise ValueError("EMG SDK timestamps do not match the signal length")
        if is_lost.shape != (sample_count,):
            raise ValueError("EMG loss flags do not match the signal length")
        if impedance_ohm.shape != (channel_count,) or saturation.shape != (channel_count,):
            raise ValueError("EMG channel diagnostics do not match the channel count")
        if sample_count > 1 and np.any(np.diff(sample_index) <= 0):
            raise ValueError("EMG sample indices must be strictly increasing within a batch")

        object.__setattr__(self, "sample_index", sample_index)
        object.__setattr__(self, "sdk_timestamp_ms", sdk_timestamp_ms)
        object.__setattr__(self, "signal_uv", signal_uv)
        object.__setattr__(self, "raw_adc", raw_adc)
        object.__setattr__(self, "is_lost", is_lost)
        object.__setattr__(self, "impedance_ohm", impedance_ohm)
        object.__setattr__(self, "saturation", saturation)

    @property
    def sample_count(self) -> int:
        return int(self.signal_uv.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.signal_uv.shape[1])


@dataclass(frozen=True)
class EmgPreview:
    """A bounded live waveform snapshot safe to consume from the Kit thread."""

    sample_index: np.ndarray
    signal_uv: np.ndarray
    is_lost: np.ndarray
    impedance_ohm: np.ndarray
    receive_monotonic_ns: int | None
    status: str

    def __post_init__(self) -> None:
        sample_index = np.asarray(self.sample_index, dtype=np.int64).reshape(-1).copy()
        signal_uv = np.asarray(self.signal_uv, dtype=np.float32).copy()
        is_lost = np.asarray(self.is_lost, dtype=np.bool_).reshape(-1).copy()
        impedance_ohm = np.asarray(self.impedance_ohm, dtype=np.float32).reshape(-1).copy()
        if signal_uv.ndim != 2:
            raise ValueError("EMG preview signal must have shape (samples, channels)")
        if signal_uv.shape[0] != len(sample_index) or len(is_lost) != len(sample_index):
            raise ValueError("EMG preview arrays have inconsistent lengths")
        if impedance_ohm.shape != (signal_uv.shape[1],):
            raise ValueError("EMG preview impedance does not match the channel count")
        object.__setattr__(self, "sample_index", sample_index)
        object.__setattr__(self, "signal_uv", signal_uv)
        object.__setattr__(self, "is_lost", is_lost)
        object.__setattr__(self, "impedance_ohm", impedance_ohm)


@dataclass(frozen=True)
class DecodedHandPreview:
    """Latest decoded UmeTrack mesh received from the EMG sidecar."""

    sequence: int
    source_sample_index: int | None
    joint_angles: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    inference_ms: float | None
    status: str

    def __post_init__(self) -> None:
        joint_angles = np.asarray(self.joint_angles, dtype=np.float32).reshape(-1).copy()
        vertices = np.asarray(self.vertices, dtype=np.float32).copy()
        triangles = np.asarray(self.triangles, dtype=np.int32).copy()
        if joint_angles.shape not in {(0,), (20,)}:
            raise ValueError(f"Decoded hand pose has invalid shape {joint_angles.shape}")
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError(f"Decoded hand vertices have invalid shape {vertices.shape}")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise ValueError(f"Decoded hand triangles have invalid shape {triangles.shape}")
        if len(triangles) and (not len(vertices) or triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError("Decoded hand triangles reference invalid vertices")
        object.__setattr__(self, "joint_angles", joint_angles)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)

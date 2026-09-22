"""Timestamp-aware fusion of wrist-local articulation with an external wrist."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
from typing import Mapping

import numpy as np

from dex_teleop.types import (
    FusedHandObservation,
    HandArticulationSample,
    Handedness,
    WristPoseSample,
)


@dataclass(frozen=True)
class ArticulationFrameTransform:
    """Rigid transform from one articulation basis into a wrist-pose basis."""

    source_frame: str
    target_frame: str
    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        if not self.source_frame or not self.source_frame.strip():
            raise ValueError("Articulation transform source_frame must be non-empty")
        if not self.target_frame or not self.target_frame.strip():
            raise ValueError("Articulation transform target_frame must be non-empty")
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3).copy()
        quaternion = (
            np.asarray(self.quaternion_xyzw, dtype=np.float64).reshape(4).copy()
        )
        if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
            raise ValueError("Articulation transform contains a non-finite value")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError(
                "Articulation transform quaternion must have non-zero norm"
            )
        translation.setflags(write=False)
        quaternion = quaternion / norm
        quaternion.setflags(write=False)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    def as_mapping(self) -> dict[str, str | list[float]]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "translation": self.translation.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw.tolist(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping) -> ArticulationFrameTransform:
        if value.get("schema_version") not in (None, 1):
            raise ValueError(
                f"Unsupported articulation-frame calibration schema {value.get('schema_version')!r}"
            )
        try:
            return cls(
                source_frame=str(value["source_frame"]),
                target_frame=str(value["target_frame"]),
                translation=value["translation"],
                quaternion_xyzw=value["quaternion_xyzw"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Articulation-frame calibration must define source_frame, target_frame, "
                "translation[3], and quaternion_xyzw[4]"
            ) from error

    @classmethod
    def load(cls, path: str | Path) -> ArticulationFrameTransform:
        with Path(path).expanduser().open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, Mapping):
            raise ValueError("Articulation-frame calibration must be a JSON object")
        return cls.from_mapping(document)


def _rotate_vectors(vectors: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    xyz = quaternion_xyzw[:3]
    w = quaternion_xyzw[3]
    cross = 2.0 * np.cross(xyz, vectors)
    return vectors + w * cross + np.cross(xyz, cross)


def _quaternion_multiply(left_xyzw: np.ndarray, right_xyzw: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left_xyzw
    rx, ry, rz, rw = right_xyzw
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _transform_articulation(
    sample: HandArticulationSample,
    transform: ArticulationFrameTransform,
) -> HandArticulationSample:
    if sample.coordinate_frame != transform.source_frame:
        raise ValueError(
            f"Articulation transform expects frame {transform.source_frame!r}, "
            f"received {sample.coordinate_frame!r}"
        )
    names = sample.joint_names
    positions = _rotate_vectors(sample.positions(names), transform.quaternion_xyzw)
    positions += transform.translation
    orientations = {
        name: _quaternion_multiply(
            transform.quaternion_xyzw,
            sample.joint_orientations_xyzw[name],
        )
        for name in names
        if name in sample.joint_orientations_xyzw
    }
    return HandArticulationSample(
        timestamp=sample.timestamp,
        receipt_timestamp=sample.receipt_timestamp,
        source_timestamp_ns=sample.source_timestamp_ns,
        source_frame_id=sample.source_frame_id,
        handedness=sample.handedness,
        joint_positions=dict(zip(names, positions, strict=True)),
        joint_orientations_xyzw=orientations,
        joint_validity=sample.joint_validity,
        source=sample.source,
        schema=sample.schema,
        coordinate_frame=transform.target_frame,
        confidence=sample.confidence,
        provenance=sample.provenance,
    )


def _slerp_xyzw(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path quaternion interpolation with hemisphere continuity."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = left + fraction * (right - left)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    scale = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / scale * left
        + math.sin(fraction * angle) / scale * right
    )


def _interpolate_wrist(
    before: WristPoseSample,
    after: WristPoseSample,
    timestamp: float,
) -> WristPoseSample:
    if before.source != after.source:
        raise ValueError("Cannot interpolate wrist poses from different sources")
    if before.handedness != after.handedness:
        raise ValueError("Cannot interpolate wrist poses with different handedness")
    if before.reference_frame != after.reference_frame:
        raise ValueError(
            "Cannot interpolate wrist poses from different reference frames"
        )
    if before.anatomical_frame != after.anatomical_frame:
        raise ValueError(
            "Cannot interpolate wrist poses with different anatomical frames"
        )
    duration = after.timestamp - before.timestamp
    if duration <= 0.0:
        raise ValueError("Wrist interpolation requires increasing timestamps")
    fraction = (timestamp - before.timestamp) / duration
    position = before.position + fraction * (after.position - before.position)
    quaternion = _slerp_xyzw(before.quaternion_xyzw, after.quaternion_xyzw, fraction)
    confidences = tuple(
        value for value in (before.confidence, after.confidence) if value is not None
    )
    source_timestamp_ns = None
    if before.source_timestamp_ns is not None and after.source_timestamp_ns is not None:
        source_timestamp_ns = round(
            before.source_timestamp_ns
            + fraction * (after.source_timestamp_ns - before.source_timestamp_ns)
        )
    return WristPoseSample(
        timestamp=timestamp,
        receipt_timestamp=max(before.receipt_timestamp, after.receipt_timestamp),
        source_timestamp_ns=source_timestamp_ns,
        source_frame_id=None,
        handedness=before.handedness,
        position=position,
        quaternion_xyzw=quaternion,
        source=before.source,
        reference_frame=before.reference_frame,
        anatomical_frame=before.anatomical_frame,
        confidence=min(confidences) if confidences else None,
        provenance={
            "interpolated": True,
            "before_source_frame_id": before.source_frame_id,
            "after_source_frame_id": after.source_frame_id,
            "before_source_timestamp_ns": before.source_timestamp_ns,
            "after_source_timestamp_ns": after.source_timestamp_ns,
            "before_capture_monotonic_ns": round(before.timestamp * 1e9),
            "after_capture_monotonic_ns": round(after.timestamp * 1e9),
        },
    )


class HandObservationFuser:
    """Buffer wrist samples and fuse them at articulation capture times.

    A fuser represents one configured articulation/wrist selection.  It can
    buffer several named wrist sources for comparison, but callers must select
    one explicitly when more than one source is available.  Interpolation is
    used only when both supporting wrist samples fall within the skew bound;
    otherwise the nearest in-bound sample is used.
    """

    def __init__(
        self,
        maximum_skew_seconds: float = 0.05,
        maximum_buffered_samples: int = 256,
        interpolate_wrist: bool = True,
        articulation_source: str | None = None,
        wrist_source: str | None = None,
        articulation_to_wrist: ArticulationFrameTransform | None = None,
    ) -> None:
        if not np.isfinite(maximum_skew_seconds) or maximum_skew_seconds < 0.0:
            raise ValueError("maximum_skew_seconds must be finite and non-negative")
        if maximum_buffered_samples <= 0:
            raise ValueError("maximum_buffered_samples must be positive")
        for name, value in (
            ("articulation_source", articulation_source),
            ("wrist_source", wrist_source),
        ):
            if value is not None and (not value or not value.strip()):
                raise ValueError(f"{name} must be non-empty or None")
        self.maximum_skew_seconds = float(maximum_skew_seconds)
        self.maximum_buffered_samples = int(maximum_buffered_samples)
        self.interpolate_wrist = bool(interpolate_wrist)
        self.articulation_source = articulation_source
        self.wrist_source = wrist_source
        self.articulation_to_wrist = articulation_to_wrist
        self._wrist_samples: dict[tuple[Handedness, str], list[WristPoseSample]] = (
            defaultdict(list)
        )
        self._reference_frames: dict[tuple[Handedness, str], str] = {}
        self._lock = threading.Lock()

    def add_wrist(self, sample: WristPoseSample) -> None:
        """Insert one wrist sample; out-of-order arrivals remain time sorted."""

        if self.wrist_source is not None and sample.source != self.wrist_source:
            raise ValueError(
                f"Fuser expects wrist source {self.wrist_source!r}, received {sample.source!r}"
            )
        key = (sample.handedness, sample.source)
        with self._lock:
            reference_frame = self._reference_frames.setdefault(
                key, sample.reference_frame
            )
            if sample.reference_frame != reference_frame:
                raise ValueError(
                    f"Wrist source {sample.source!r} changed reference frame from "
                    f"{reference_frame!r} to {sample.reference_frame!r}; reset the fuser first"
                )
            samples = self._wrist_samples[key]
            timestamps = [item.timestamp for item in samples]
            index = bisect_left(timestamps, sample.timestamp)
            if index < len(samples) and samples[index].timestamp == sample.timestamp:
                # A source may republish one capture.  Retain the value that
                # became available most recently instead of growing duplicates.
                if sample.receipt_timestamp >= samples[index].receipt_timestamp:
                    samples[index] = sample
            else:
                samples.insert(index, sample)
            if len(samples) > self.maximum_buffered_samples:
                del samples[: len(samples) - self.maximum_buffered_samples]

    def fuse(
        self,
        articulation: HandArticulationSample,
        *,
        wrist_source: str | None = None,
    ) -> FusedHandObservation | None:
        """Fuse one articulation with the selected wrist, or return ``None``.

        ``None`` means that no wrist sample satisfies the configured skew.  It
        is intentionally not a fallback to another source.
        """

        if (
            self.articulation_source is not None
            and articulation.source != self.articulation_source
        ):
            raise ValueError(
                f"Fuser expects articulation source {self.articulation_source!r}, "
                f"received {articulation.source!r}"
            )
        selected_source = self._resolve_wrist_source(
            articulation.handedness, wrist_source
        )
        if selected_source is None:
            return None
        with self._lock:
            samples = tuple(
                self._wrist_samples[(articulation.handedness, selected_source)]
            )
        match = self._wrist_at(samples, articulation.timestamp)
        if match is None:
            return None
        wrist, skew = match
        if self.articulation_to_wrist is not None:
            articulation = _transform_articulation(
                articulation,
                self.articulation_to_wrist,
            )
        if articulation.coordinate_frame != wrist.anatomical_frame:
            raise ValueError(
                f"Articulation frame {articulation.coordinate_frame!r} cannot be composed with "
                f"wrist anatomical frame {wrist.anatomical_frame!r}; configure an explicit "
                "articulation-to-wrist transform"
            )
        return FusedHandObservation(
            articulation=articulation,
            wrist=wrist,
            synchronization_skew_seconds=skew,
        )

    def fuse_paired(
        self,
        articulation: HandArticulationSample,
        wrist: WristPoseSample,
    ) -> FusedHandObservation:
        """Fuse components declared co-emitted by one receiver callback.

        This path deliberately does not search the timestamp buffer or apply a
        skew threshold. Pair identity and exact callback time must agree.
        """

        if (
            self.articulation_source is not None
            and articulation.source != self.articulation_source
        ):
            raise ValueError(
                f"Fuser expects articulation source {self.articulation_source!r}, "
                f"received {articulation.source!r}"
            )
        if self.wrist_source is not None and wrist.source != self.wrist_source:
            raise ValueError(
                f"Fuser expects wrist source {self.wrist_source!r}, received {wrist.source!r}"
            )
        if articulation.timestamp != wrist.timestamp:
            raise ValueError(
                "Co-emitted articulation and wrist capture timestamps must be identical"
            )
        if (
            articulation.source_timestamp_ns is not None
            or wrist.source_timestamp_ns is not None
        ) and articulation.source_timestamp_ns != wrist.source_timestamp_ns:
            raise ValueError(
                "Co-emitted articulation and wrist source timestamps must be identical"
            )
        if (
            articulation.source_frame_id is not None
            or wrist.source_frame_id is not None
        ) and articulation.source_frame_id != wrist.source_frame_id:
            raise ValueError(
                "Co-emitted articulation and wrist frame IDs must be identical"
            )
        if self.articulation_to_wrist is not None:
            articulation = _transform_articulation(
                articulation,
                self.articulation_to_wrist,
            )
        if articulation.coordinate_frame != wrist.anatomical_frame:
            raise ValueError(
                f"Articulation frame {articulation.coordinate_frame!r} cannot be composed with "
                f"wrist anatomical frame {wrist.anatomical_frame!r}; configure an explicit "
                "articulation-to-wrist transform"
            )
        return FusedHandObservation(
            articulation=articulation,
            wrist=wrist,
            synchronization_skew_seconds=0.0,
        )

    def _resolve_wrist_source(
        self, handedness: Handedness, requested: str | None
    ) -> str | None:
        if self.wrist_source is not None:
            if requested is not None and requested != self.wrist_source:
                raise ValueError(
                    f"Fuser is fixed to wrist source {self.wrist_source!r}, not {requested!r}"
                )
            return self.wrist_source
        if requested is not None:
            return requested
        with self._lock:
            candidates = sorted(
                source
                for (sample_handedness, source), samples in self._wrist_samples.items()
                if sample_handedness == handedness and samples
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Several wrist sources are buffered for {handedness.value}: {candidates}; "
                "select wrist_source explicitly"
            )
        return candidates[0] if candidates else None

    def _wrist_at(
        self, samples: tuple[WristPoseSample, ...], timestamp: float
    ) -> tuple[WristPoseSample, float] | None:
        if not samples:
            return None
        timestamps = [sample.timestamp for sample in samples]
        index = bisect_left(timestamps, timestamp)
        if index < len(samples) and samples[index].timestamp == timestamp:
            return samples[index], 0.0

        before = samples[index - 1] if index > 0 else None
        after = samples[index] if index < len(samples) else None
        if self.interpolate_wrist and before is not None and after is not None:
            before_skew = timestamp - before.timestamp
            after_skew = after.timestamp - timestamp
            if (
                before_skew <= self.maximum_skew_seconds
                and after_skew <= self.maximum_skew_seconds
            ):
                wrist = _interpolate_wrist(before, after, timestamp)
                return wrist, max(before_skew, after_skew)

        candidates = [sample for sample in (before, after) if sample is not None]
        # Prefer the earlier sample for an exact distance tie: it was already
        # available at the articulation capture time.
        nearest = min(
            candidates,
            key=lambda sample: (
                abs(sample.timestamp - timestamp),
                sample.timestamp > timestamp,
            ),
        )
        skew = abs(nearest.timestamp - timestamp)
        if skew > self.maximum_skew_seconds:
            return None
        return nearest, skew

    def reset(self) -> None:
        """Flush all temporal and reference-frame state."""

        with self._lock:
            self._wrist_samples.clear()
            self._reference_frames.clear()

    def buffered_wrist_count(
        self, handedness: Handedness | None = None, source: str | None = None
    ) -> int:
        """Return the number of buffered wrist samples, primarily for diagnostics."""

        with self._lock:
            return sum(
                len(samples)
                for (
                    sample_handedness,
                    sample_source,
                ), samples in self._wrist_samples.items()
                if (handedness is None or sample_handedness == handedness)
                and (source is None or sample_source == source)
            )

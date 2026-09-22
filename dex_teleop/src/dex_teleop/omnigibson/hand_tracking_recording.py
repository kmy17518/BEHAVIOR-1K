"""Native-rate, multi-source hand tracking stored alongside OG trajectories.

The existing :mod:`hand_pose_recording` format intentionally remains the
action-aligned MediaPipe-21 compatibility view.  This module stores every
enabled source at its own rate and uses explicit row references to associate a
selected observation and retargeted command with each simulator action.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence
import uuid

import h5py
import numpy as np

from dex_teleop.types import (
    HandArticulationSample,
    RetargetedHandCommand,
    WristPoseSample,
)


HAND_TRACKING_GROUP = "hand_tracking"
HAND_TRACKING_SCHEMA_VERSION = 1
MISSING_INTEGER_SENTINEL = -1


@dataclass(frozen=True)
class RetargetedCommandSample:
    """One retargeter result and the exact source rows that produced it."""

    command: RetargetedHandCommand
    retargeter: str
    articulation_stream: str
    articulation_row: int
    wrist_stream: str | None = None
    wrist_row: int | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.retargeter, "retargeter")
        _validate_identifier(self.articulation_stream, "articulation stream")
        if self.articulation_row < 0:
            raise ValueError("Retargeted command articulation row must be non-negative")
        if (self.wrist_stream is None) != (self.wrist_row is None):
            raise ValueError(
                "Retargeted command wrist stream and row must either both be set or both be absent"
            )
        if self.wrist_stream is not None:
            _validate_identifier(self.wrist_stream, "wrist stream")
            if self.wrist_row < 0:
                raise ValueError("Retargeted command wrist row must be non-negative")


@dataclass(frozen=True)
class ActionHandSelection:
    """Source and output rows selected for one recorded simulator action."""

    articulation_stream: str
    articulation_row: int
    wrist_stream: str
    wrist_row: int
    retargeting_stream: str
    retargeting_row: int
    synchronization_skew_seconds: float

    def __post_init__(self) -> None:
        _validate_identifier(self.articulation_stream, "articulation stream")
        _validate_identifier(self.wrist_stream, "wrist stream")
        _validate_identifier(self.retargeting_stream, "retargeting stream")
        if min(self.articulation_row, self.wrist_row, self.retargeting_row) < 0:
            raise ValueError("Action hand-selection rows must be non-negative")
        if (
            not np.isfinite(self.synchronization_skew_seconds)
            or self.synchronization_skew_seconds < 0.0
        ):
            raise ValueError(
                "Action hand-selection synchronization skew must be finite and non-negative"
            )


def _sample_key(sample: HandArticulationSample | WristPoseSample) -> tuple:
    if sample.source_frame_id is not None or sample.source_timestamp_ns is not None:
        return ("source", sample.source_frame_id, sample.source_timestamp_ns)
    return ("capture", round(sample.timestamp * 1e9))


def _articulations_equal(
    first: HandArticulationSample, second: HandArticulationSample
) -> bool:
    names = first.joint_names
    return (
        first.timestamp == second.timestamp
        and first.receipt_timestamp == second.receipt_timestamp
        and first.handedness == second.handedness
        and first.source == second.source
        and first.schema == second.schema
        and first.coordinate_frame == second.coordinate_frame
        and first.source_timestamp_ns == second.source_timestamp_ns
        and first.source_frame_id == second.source_frame_id
        and first.confidence == second.confidence
        and dict(first.provenance) == dict(second.provenance)
        and set(names) == set(second.joint_names)
        and np.array_equal(
            first.positions(names), second.positions(names), equal_nan=True
        )
        and np.array_equal(
            first.orientations_xyzw(names),
            second.orientations_xyzw(names),
            equal_nan=True,
        )
        and np.array_equal(first.validity(names), second.validity(names))
    )


def _wrists_equal(first: WristPoseSample, second: WristPoseSample) -> bool:
    return (
        first.timestamp == second.timestamp
        and first.receipt_timestamp == second.receipt_timestamp
        and first.handedness == second.handedness
        and first.source == second.source
        and first.reference_frame == second.reference_frame
        and first.anatomical_frame == second.anatomical_frame
        and first.source_timestamp_ns == second.source_timestamp_ns
        and first.source_frame_id == second.source_frame_id
        and first.confidence == second.confidence
        and dict(first.provenance) == dict(second.provenance)
        and np.array_equal(first.position, second.position)
        and np.array_equal(first.quaternion_xyzw, second.quaternion_xyzw)
    )


def _commands_equal(
    first: RetargetedCommandSample, second: RetargetedCommandSample
) -> bool:
    return (
        first.retargeter == second.retargeter
        and first.articulation_stream == second.articulation_stream
        and first.articulation_row == second.articulation_row
        and first.wrist_stream == second.wrist_stream
        and first.wrist_row == second.wrist_row
        and first.command.timestamp == second.command.timestamp
        and first.command.handedness == second.command.handedness
        and first.command.hand_model == second.command.hand_model
        and first.command.joint_names == second.command.joint_names
        and np.array_equal(
            first.command.joint_positions, second.command.joint_positions
        )
    )


class HandTrackingRecordingSession:
    """Thread-safe, session-scoped collector for the multi-source writer.

    Source callbacks may append articulation and wrist samples while the
    control thread records outputs and action selections. Duplicate polling of
    a latest source sample returns its existing stable row. ``close`` freezes
    the collector but retains its data for final HDF5 publication.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._articulation_streams: dict[str, list[HandArticulationSample]] = {}
        self._articulation_rows: dict[str, dict[tuple, int]] = {}
        self._wrist_streams: dict[str, list[WristPoseSample]] = {}
        self._wrist_rows: dict[str, dict[tuple, int]] = {}
        self._retargeting_streams: dict[str, list[RetargetedCommandSample]] = {}
        self._retargeting_rows: dict[str, dict[tuple, int]] = {}
        self._action_alignment_episodes: list[list[ActionHandSelection]] = []
        self._stream_metadata: dict[str, dict[str, Any]] = {}

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Hand-tracking recording session is closed")

    @staticmethod
    def _check_order(samples: Sequence[Any], sample: Any, stream_id: str) -> None:
        if samples and sample.timestamp <= samples[-1].timestamp:
            raise ValueError(
                f"Stream {stream_id!r} received a new sample with a non-increasing timestamp"
            )
        if samples and sample.receipt_timestamp < samples[-1].receipt_timestamp:
            raise ValueError(
                f"Stream {stream_id!r} received a sample whose receipt time moved backwards"
            )

    def append_articulation(
        self, stream_id: str, sample: HandArticulationSample
    ) -> int:
        """Append one articulation sample, returning its stable source row."""

        _validate_identifier(stream_id, "articulation stream")
        key = _sample_key(sample)
        with self._lock:
            self._require_open()
            rows = self._articulation_rows.setdefault(stream_id, {})
            samples = self._articulation_streams.setdefault(stream_id, [])
            existing = rows.get(key)
            if existing is not None:
                if not _articulations_equal(samples[existing], sample):
                    raise ValueError(
                        f"Articulation stream {stream_id!r} reused a source identity with different data"
                    )
                return existing
            self._check_order(samples, sample, stream_id)
            row = len(samples)
            samples.append(sample)
            rows[key] = row
            return row

    def append_wrist(self, stream_id: str, sample: WristPoseSample) -> int:
        """Append one wrist sample, returning its stable source row."""

        _validate_identifier(stream_id, "wrist stream")
        key = _sample_key(sample)
        with self._lock:
            self._require_open()
            rows = self._wrist_rows.setdefault(stream_id, {})
            samples = self._wrist_streams.setdefault(stream_id, [])
            existing = rows.get(key)
            if existing is not None:
                if not _wrists_equal(samples[existing], sample):
                    raise ValueError(
                        f"Wrist stream {stream_id!r} reused a source identity with different data"
                    )
                return existing
            self._check_order(samples, sample, stream_id)
            row = len(samples)
            samples.append(sample)
            rows[key] = row
            return row

    def append_retargeted(self, stream_id: str, sample: RetargetedCommandSample) -> int:
        """Append one selected or shadow retargeter output and return its row."""

        _validate_identifier(stream_id, "retargeting stream")
        key = (
            round(sample.command.timestamp * 1e9),
            sample.articulation_stream,
            sample.articulation_row,
            sample.wrist_stream,
            sample.wrist_row,
        )
        with self._lock:
            self._require_open()
            _validate_reference(
                self._articulation_streams,
                sample.articulation_stream,
                sample.articulation_row,
                f"Retargeting stream {stream_id!r}",
            )
            articulation = self._articulation_streams[sample.articulation_stream][
                sample.articulation_row
            ]
            if articulation.handedness != sample.command.handedness:
                raise ValueError(
                    f"Retargeting stream {stream_id!r} has inconsistent handedness"
                )
            if sample.wrist_stream is not None:
                _validate_reference(
                    self._wrist_streams,
                    sample.wrist_stream,
                    sample.wrist_row,
                    f"Retargeting stream {stream_id!r}",
                )
            rows = self._retargeting_rows.setdefault(stream_id, {})
            samples = self._retargeting_streams.setdefault(stream_id, [])
            existing = rows.get(key)
            if existing is not None:
                if not _commands_equal(samples[existing], sample):
                    raise ValueError(
                        f"Retargeting stream {stream_id!r} reused an output identity with different data"
                    )
                return existing
            if samples and sample.command.timestamp <= samples[-1].command.timestamp:
                raise ValueError(
                    f"Retargeting stream {stream_id!r} received a new output with a non-increasing timestamp"
                )
            row = len(samples)
            samples.append(sample)
            rows[key] = row
            return row

    def set_stream_metadata(self, stream_id: str, metadata: Mapping[str, Any]) -> None:
        """Set immutable JSON metadata such as calibration and device IDs."""

        _validate_identifier(stream_id, "stream")
        value = deepcopy(dict(metadata))
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Hand-tracking stream metadata must be JSON serializable"
            ) from error
        with self._lock:
            self._require_open()
            previous = self._stream_metadata.get(stream_id)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Metadata for stream {stream_id!r} has already been set"
                )
            self._stream_metadata[stream_id] = value

    def begin_episode(self) -> int:
        """Begin a trajectory episode and return its zero-based collector index."""

        with self._lock:
            self._require_open()
            self._action_alignment_episodes.append([])
            return len(self._action_alignment_episodes) - 1

    def append_action_selection(
        self, selection: ActionHandSelection, *, episode_index: int | None = None
    ) -> int:
        """Append one action-aligned selection and return its action row."""

        with self._lock:
            self._require_open()
            if not self._action_alignment_episodes:
                raise RuntimeError(
                    "Begin a hand-tracking recording episode before appending actions"
                )
            _validate_reference(
                self._articulation_streams,
                selection.articulation_stream,
                selection.articulation_row,
                "Action hand selection",
            )
            _validate_reference(
                self._wrist_streams,
                selection.wrist_stream,
                selection.wrist_row,
                "Action hand selection",
            )
            _validate_reference(
                self._retargeting_streams,
                selection.retargeting_stream,
                selection.retargeting_row,
                "Action hand selection",
            )
            output = self._retargeting_streams[selection.retargeting_stream][
                selection.retargeting_row
            ]
            if (
                output.articulation_stream != selection.articulation_stream
                or output.articulation_row != selection.articulation_row
                or output.wrist_stream != selection.wrist_stream
                or output.wrist_row != selection.wrist_row
            ):
                raise ValueError(
                    "Action hand selection does not match its retargeting input rows"
                )
            index = (
                len(self._action_alignment_episodes) - 1
                if episode_index is None
                else episode_index
            )
            if not 0 <= index < len(self._action_alignment_episodes):
                raise IndexError(f"Invalid hand-tracking episode index {index}")
            episode = self._action_alignment_episodes[index]
            row = len(episode)
            episode.append(selection)
            return row

    def writer_kwargs(self) -> dict[str, Any]:
        """Return detached, stable mappings accepted by the HDF5 writer."""

        with self._lock:
            return {
                "articulation_streams": {
                    key: tuple(value)
                    for key, value in self._articulation_streams.items()
                },
                "wrist_streams": {
                    key: tuple(value) for key, value in self._wrist_streams.items()
                },
                "retargeting_streams": {
                    key: tuple(value)
                    for key, value in self._retargeting_streams.items()
                },
                "action_alignment_episodes": tuple(
                    tuple(episode) for episode in self._action_alignment_episodes
                ),
                "stream_metadata": {
                    key: deepcopy(value) for key, value in self._stream_metadata.items()
                },
            }

    def close(self) -> None:
        """Freeze this session; repeated calls are harmless."""

        with self._lock:
            self._closed = True

    def write(self, input_path: str | Path) -> None:
        """Publish the frozen session into a completed trajectory recording."""

        with self._lock:
            if not self._closed:
                raise RuntimeError(
                    "Close the hand-tracking recording session before writing"
                )
        write_multi_source_hand_recording(input_path, **self.writer_kwargs())

    def __enter__(self) -> HandTrackingRecordingSession:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or "/" in value:
        raise ValueError(
            f"{label.capitalize()} must be a non-empty HDF5-safe identifier without '/'"
        )


def _episode_groups(group: h5py.Group) -> list[tuple[int, h5py.Group]]:
    episodes = []
    for name, episode in group.items():
        if not name.startswith("demo_") or not isinstance(episode, h5py.Group):
            continue
        try:
            episode_id = int(name.removeprefix("demo_"))
        except ValueError:
            continue
        episodes.append((episode_id, episode))
    return sorted(episodes)


def _seconds_to_ns(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{label} timestamps must be finite and non-negative")
    return np.rint(array * 1e9).astype(np.int64)


def _optional_integer(values: Sequence[int | None]) -> np.ndarray:
    return np.asarray(
        [MISSING_INTEGER_SENTINEL if value is None else value for value in values],
        dtype=np.int64,
    )


def _optional_confidence(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else value for value in values], dtype=np.float64
    )


def _validate_monotonic(samples: Sequence[Any], stream_id: str) -> None:
    timestamps = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
    receipts = np.asarray(
        [sample.receipt_timestamp for sample in samples], dtype=np.float64
    )
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(
            f"Stream {stream_id!r} capture timestamps must be strictly increasing"
        )
    if len(receipts) > 1 and np.any(np.diff(receipts) < 0.0):
        raise ValueError(
            f"Stream {stream_id!r} receipt timestamps must not move backwards"
        )


def _stream_metadata(
    metadata: Mapping[str, Mapping[str, Any]] | None, stream_id: str
) -> str:
    value = {} if metadata is None else dict(metadata.get(stream_id, {}))
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Metadata for stream {stream_id!r} must be JSON serializable"
        ) from error


def _validate_articulation_stream(
    stream_id: str, samples: Sequence[HandArticulationSample]
) -> tuple[str, ...]:
    _validate_identifier(stream_id, "articulation stream")
    if not samples:
        raise ValueError(
            f"Articulation stream {stream_id!r} has no samples; omit empty streams"
        )
    _validate_monotonic(samples, stream_id)
    first = samples[0]
    joint_names = tuple(first.joint_names)
    for sample in samples:
        if (
            sample.source != first.source
            or sample.handedness != first.handedness
            or sample.schema != first.schema
            or sample.coordinate_frame != first.coordinate_frame
        ):
            raise ValueError(
                f"Articulation stream {stream_id!r} mixes source, handedness, schema, "
                "or coordinate-frame values"
            )
        if set(sample.joint_names) != set(joint_names):
            raise ValueError(f"Articulation stream {stream_id!r} changes its joint set")
    return joint_names


def _validate_wrist_stream(stream_id: str, samples: Sequence[WristPoseSample]) -> None:
    _validate_identifier(stream_id, "wrist stream")
    if not samples:
        raise ValueError(
            f"Wrist stream {stream_id!r} has no samples; omit empty streams"
        )
    _validate_monotonic(samples, stream_id)
    first = samples[0]
    for sample in samples:
        if (
            sample.source != first.source
            or sample.handedness != first.handedness
            or sample.reference_frame != first.reference_frame
            or sample.anatomical_frame != first.anatomical_frame
        ):
            raise ValueError(
                f"Wrist stream {stream_id!r} mixes source, handedness, reference-frame, "
                "or anatomical-frame values"
            )


def _write_common_timing(group: h5py.Group, samples: Sequence[Any]) -> None:
    group.create_dataset(
        "capture_monotonic_ns",
        data=_seconds_to_ns([sample.timestamp for sample in samples], "Capture"),
    )
    group.create_dataset(
        "receipt_monotonic_ns",
        data=_seconds_to_ns(
            [sample.receipt_timestamp for sample in samples], "Receipt"
        ),
    )
    group.create_dataset(
        "source_timestamp_ns",
        data=_optional_integer([sample.source_timestamp_ns for sample in samples]),
    )
    group.create_dataset(
        "source_frame_id",
        data=_optional_integer([sample.source_frame_id for sample in samples]),
    )
    group.create_dataset(
        "confidence",
        data=_optional_confidence([sample.confidence for sample in samples]),
    )
    group.create_dataset(
        "provenance_json",
        data=[
            json.dumps(
                dict(sample.provenance),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for sample in samples
        ],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _write_articulation_stream(
    parent: h5py.Group,
    stream_id: str,
    samples: Sequence[HandArticulationSample],
    joint_names: tuple[str, ...],
    metadata: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    first = samples[0]
    stream = parent.create_group(stream_id)
    stream.attrs["source"] = first.source
    stream.attrs["handedness"] = first.handedness.value
    stream.attrs["schema"] = first.schema
    stream.attrs["joint_names"] = json.dumps(joint_names)
    stream.attrs["coordinate_frame"] = first.coordinate_frame
    stream.attrs["metadata"] = _stream_metadata(metadata, stream_id)
    _write_common_timing(stream, samples)
    stream.create_dataset(
        "joint_positions",
        data=np.stack([sample.positions(joint_names) for sample in samples]),
    )

    orientations = []
    orientation_available = []
    validity = []
    for sample in samples:
        orientation_mapping = sample.joint_orientations_xyzw
        orientations.append(
            np.stack(
                [
                    np.full(4, np.nan, dtype=np.float64)
                    if orientation_mapping is None or name not in orientation_mapping
                    else orientation_mapping[name]
                    for name in joint_names
                ]
            )
        )
        orientation_available.append(
            [
                orientation_mapping is not None and name in orientation_mapping
                for name in joint_names
            ]
        )
        validity.append(sample.validity(joint_names))
    stream.create_dataset("joint_orientations_xyzw", data=np.stack(orientations))
    stream.create_dataset(
        "joint_orientation_available",
        data=np.asarray(orientation_available, dtype=np.bool_),
    )
    stream.create_dataset("joint_validity", data=np.asarray(validity, dtype=np.bool_))


def _write_wrist_stream(
    parent: h5py.Group,
    stream_id: str,
    samples: Sequence[WristPoseSample],
    metadata: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    first = samples[0]
    stream = parent.create_group(stream_id)
    stream.attrs["source"] = first.source
    stream.attrs["handedness"] = first.handedness.value
    stream.attrs["reference_frame"] = first.reference_frame
    stream.attrs["anatomical_frame"] = first.anatomical_frame
    stream.attrs["metadata"] = _stream_metadata(metadata, stream_id)
    _write_common_timing(stream, samples)
    stream.create_dataset(
        "position", data=np.stack([sample.position for sample in samples])
    )
    stream.create_dataset(
        "quaternion_xyzw", data=np.stack([sample.quaternion_xyzw for sample in samples])
    )


def _validate_reference(
    streams: Mapping[str, Sequence[Any]], stream_id: str, row: int, label: str
) -> None:
    if stream_id not in streams:
        raise ValueError(f"{label} references unknown stream {stream_id!r}")
    if not 0 <= row < len(streams[stream_id]):
        raise ValueError(f"{label} references row {row} outside stream {stream_id!r}")


def _validate_retargeting_streams(
    streams: Mapping[str, Sequence[RetargetedCommandSample]],
    articulation_streams: Mapping[str, Sequence[HandArticulationSample]],
    wrist_streams: Mapping[str, Sequence[WristPoseSample]],
) -> None:
    for stream_id, samples in streams.items():
        _validate_identifier(stream_id, "retargeting stream")
        if not samples:
            raise ValueError(
                f"Retargeting stream {stream_id!r} has no samples; omit empty streams"
            )
        first = samples[0]
        command_times = []
        for row, sample in enumerate(samples):
            command = sample.command
            command_times.append(command.timestamp)
            if (
                sample.retargeter != first.retargeter
                or command.hand_model != first.command.hand_model
                or command.handedness != first.command.handedness
                or command.joint_names != first.command.joint_names
            ):
                raise ValueError(
                    f"Retargeting stream {stream_id!r} mixes pipeline or command schema values"
                )
            _validate_reference(
                articulation_streams,
                sample.articulation_stream,
                sample.articulation_row,
                f"Retargeting stream {stream_id!r} row {row}",
            )
            articulation = articulation_streams[sample.articulation_stream][
                sample.articulation_row
            ]
            if articulation.handedness != command.handedness:
                raise ValueError(
                    f"Retargeting stream {stream_id!r} row {row} has inconsistent handedness"
                )
            if sample.wrist_stream is not None:
                _validate_reference(
                    wrist_streams,
                    sample.wrist_stream,
                    sample.wrist_row,
                    f"Retargeting stream {stream_id!r} row {row}",
                )
        if len(command_times) > 1 and np.any(np.diff(command_times) <= 0.0):
            raise ValueError(
                f"Retargeting stream {stream_id!r} timestamps must be strictly increasing"
            )


def _write_retargeting_stream(
    parent: h5py.Group,
    stream_id: str,
    samples: Sequence[RetargetedCommandSample],
    metadata: Mapping[str, Mapping[str, Any]] | None,
    string_dtype,
) -> None:
    first = samples[0]
    stream = parent.create_group(stream_id)
    stream.attrs["retargeter"] = first.retargeter
    stream.attrs["hand_model"] = first.command.hand_model
    stream.attrs["handedness"] = first.command.handedness.value
    stream.attrs["joint_names"] = json.dumps(first.command.joint_names)
    stream.attrs["metadata"] = _stream_metadata(metadata, stream_id)
    stream.create_dataset(
        "capture_monotonic_ns",
        data=_seconds_to_ns(
            [sample.command.timestamp for sample in samples], "Retargeted command"
        ),
    )
    stream.create_dataset(
        "joint_positions",
        data=np.stack([sample.command.joint_positions for sample in samples]),
    )
    stream.create_dataset(
        "articulation_stream",
        data=[sample.articulation_stream for sample in samples],
        dtype=string_dtype,
    )
    stream.create_dataset(
        "articulation_row",
        data=np.asarray(
            [sample.articulation_row for sample in samples], dtype=np.int64
        ),
    )
    stream.create_dataset(
        "wrist_stream",
        data=[
            "" if sample.wrist_stream is None else sample.wrist_stream
            for sample in samples
        ],
        dtype=string_dtype,
    )
    stream.create_dataset(
        "wrist_row",
        data=np.asarray(
            [
                MISSING_INTEGER_SENTINEL
                if sample.wrist_row is None
                else sample.wrist_row
                for sample in samples
            ],
            dtype=np.int64,
        ),
    )


def _validate_action_alignment(
    episodes: Sequence[Sequence[ActionHandSelection]],
    trajectory_lengths: Sequence[tuple[int, int]],
    articulation_streams: Mapping[str, Sequence[HandArticulationSample]],
    wrist_streams: Mapping[str, Sequence[WristPoseSample]],
    retargeting_streams: Mapping[str, Sequence[RetargetedCommandSample]],
) -> None:
    if len(episodes) != len(trajectory_lengths):
        raise RuntimeError(
            f"Hand-tracking episode count {len(episodes)} does not match trajectory count {len(trajectory_lengths)}"
        )
    for selections, (episode_id, expected_steps) in zip(
        episodes, trajectory_lengths, strict=True
    ):
        if len(selections) != expected_steps:
            raise RuntimeError(
                f"Hand-tracking demo_{episode_id} has {len(selections)} steps; trajectory has {expected_steps}"
            )
        for action_row, selection in enumerate(selections):
            label = f"Hand-tracking demo_{episode_id} action {action_row}"
            _validate_reference(
                articulation_streams,
                selection.articulation_stream,
                selection.articulation_row,
                label,
            )
            _validate_reference(
                wrist_streams, selection.wrist_stream, selection.wrist_row, label
            )
            _validate_reference(
                retargeting_streams,
                selection.retargeting_stream,
                selection.retargeting_row,
                label,
            )
            output = retargeting_streams[selection.retargeting_stream][
                selection.retargeting_row
            ]
            if (
                output.articulation_stream != selection.articulation_stream
                or output.articulation_row != selection.articulation_row
                or output.wrist_stream != selection.wrist_stream
                or output.wrist_row != selection.wrist_row
            ):
                raise ValueError(
                    f"{label} does not match its selected retargeting input rows"
                )


def _write_action_episode(
    parent: h5py.Group,
    episode_id: int,
    selections: Sequence[ActionHandSelection],
    string_dtype,
) -> None:
    episode = parent.create_group(f"demo_{episode_id}")
    episode.attrs["actual_action_dataset"] = f"/data/demo_{episode_id}/action"
    episode.attrs["row_alignment"] = "one row per simulator action with the same index"
    for field in ("articulation_stream", "wrist_stream", "retargeting_stream"):
        episode.create_dataset(
            field,
            data=[getattr(item, field) for item in selections],
            dtype=string_dtype,
        )
    for field in ("articulation_row", "wrist_row", "retargeting_row"):
        episode.create_dataset(
            field,
            data=np.asarray(
                [getattr(item, field) for item in selections], dtype=np.int64
            ),
        )
    episode.create_dataset(
        "synchronization_skew_seconds",
        data=np.asarray(
            [item.synchronization_skew_seconds for item in selections], dtype=np.float64
        ),
    )


def write_multi_source_hand_recording(
    input_path: str | Path,
    *,
    articulation_streams: Mapping[str, Sequence[HandArticulationSample]],
    wrist_streams: Mapping[str, Sequence[WristPoseSample]],
    retargeting_streams: Mapping[str, Sequence[RetargetedCommandSample]],
    action_alignment_episodes: Sequence[Sequence[ActionHandSelection]],
    stream_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Append independent source streams and action selections to a completed recording.

    Articulation, wrist, and shadow-retargeter streams keep their acquisition
    rates. Repeated row references in ``action_alignment_episodes`` are valid
    when multiple simulator actions use the same source sample.
    """

    articulation_streams = {
        key: tuple(value) for key, value in articulation_streams.items()
    }
    wrist_streams = {key: tuple(value) for key, value in wrist_streams.items()}
    retargeting_streams = {
        key: tuple(value) for key, value in retargeting_streams.items()
    }
    action_alignment_episodes = tuple(
        tuple(episode) for episode in action_alignment_episodes
    )
    if stream_metadata is not None:
        try:
            stream_metadata = deepcopy(
                {key: dict(value) for key, value in stream_metadata.items()}
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Hand-tracking stream metadata must map stream IDs to JSON objects"
            ) from error
        for stream_id, value in stream_metadata.items():
            _validate_identifier(stream_id, "metadata stream")
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Metadata for stream {stream_id!r} must be JSON serializable"
                ) from error

    overlap = (
        set(articulation_streams).intersection(wrist_streams)
        | set(articulation_streams).intersection(retargeting_streams)
        | set(wrist_streams).intersection(retargeting_streams)
    )
    if overlap:
        raise ValueError(
            f"Hand-tracking stream identifiers must be globally unique: {sorted(overlap)}"
        )
    joint_names = {
        stream_id: _validate_articulation_stream(stream_id, samples)
        for stream_id, samples in articulation_streams.items()
    }
    for stream_id, samples in wrist_streams.items():
        _validate_wrist_stream(stream_id, samples)
    _validate_retargeting_streams(
        retargeting_streams, articulation_streams, wrist_streams
    )
    # Complete all conversions that can fail before opening the target file.
    # The pending-group transaction below protects against HDF5 write errors;
    # this preflight also keeps ordinary data errors entirely read-only.
    for stream_id, samples in articulation_streams.items():
        _seconds_to_ns([sample.timestamp for sample in samples], stream_id)
        _seconds_to_ns(
            [sample.receipt_timestamp for sample in samples], stream_id
        )
    for stream_id, samples in wrist_streams.items():
        _seconds_to_ns([sample.timestamp for sample in samples], stream_id)
        _seconds_to_ns(
            [sample.receipt_timestamp for sample in samples], stream_id
        )
    for stream_id, samples in retargeting_streams.items():
        _seconds_to_ns(
            [sample.command.timestamp for sample in samples], stream_id
        )

    input_path = Path(input_path)
    with h5py.File(input_path, "r+") as recording:
        if "data" not in recording:
            raise ValueError(f"Recording has no 'data' group: {input_path}")
        trajectory_episodes = _episode_groups(recording["data"])
        trajectory_lengths = []
        for episode_id, trajectory in trajectory_episodes:
            if "num_samples" not in trajectory.attrs:
                raise ValueError(
                    f"Trajectory demo_{episode_id} has no num_samples attribute"
                )
            expected_steps = int(trajectory.attrs["num_samples"])
            action = trajectory.get("action")
            if not isinstance(action, h5py.Dataset) or action.ndim < 1:
                raise ValueError(
                    f"Trajectory demo_{episode_id} has no row-aligned action dataset"
                )
            if action.shape[0] != expected_steps:
                raise RuntimeError(
                    f"Trajectory demo_{episode_id} action dataset has {action.shape[0]} "
                    f"rows; num_samples is {expected_steps}"
                )
            trajectory_lengths.append((episode_id, expected_steps))
        _validate_action_alignment(
            action_alignment_episodes,
            trajectory_lengths,
            articulation_streams,
            wrist_streams,
            retargeting_streams,
        )

        transaction_id = uuid.uuid4().hex
        pending_name = f"__{HAND_TRACKING_GROUP}_pending_{transaction_id}"
        backup_name = f"__{HAND_TRACKING_GROUP}_backup_{transaction_id}"
        try:
            hand_tracking = recording.create_group(pending_name)
            hand_tracking.attrs["schema_version"] = HAND_TRACKING_SCHEMA_VERSION
            hand_tracking.attrs["clock"] = "time.monotonic_ns"
            hand_tracking.attrs["quaternion_order"] = "xyzw"
            hand_tracking.attrs["position_unit"] = "meter"
            hand_tracking.attrs["missing_integer_sentinel"] = MISSING_INTEGER_SENTINEL
            hand_tracking.attrs["legacy_selected_pose_group"] = "/human_hand_pose"
            hand_tracking.attrs["source_timing"] = (
                "capture timestamps share the desktop monotonic domain; source timestamps retain source provenance"
            )
            string_dtype = h5py.string_dtype(encoding="utf-8")

            articulation_group = hand_tracking.create_group("articulation_streams")
            for stream_id, samples in articulation_streams.items():
                _write_articulation_stream(
                    articulation_group,
                    stream_id,
                    samples,
                    joint_names[stream_id],
                    stream_metadata,
                )

            wrist_group = hand_tracking.create_group("wrist_streams")
            for stream_id, samples in wrist_streams.items():
                _write_wrist_stream(wrist_group, stream_id, samples, stream_metadata)

            retargeting_group = hand_tracking.create_group("retargeting_streams")
            for stream_id, samples in retargeting_streams.items():
                _write_retargeting_stream(
                    retargeting_group,
                    stream_id,
                    samples,
                    stream_metadata,
                    string_dtype,
                )

            alignment_group = hand_tracking.create_group("action_alignment")
            for selections, (episode_id, _expected_steps) in zip(
                action_alignment_episodes,
                trajectory_lengths,
                strict=True,
            ):
                _write_action_episode(
                    alignment_group, episode_id, selections, string_dtype
                )
            recording.flush()
        except BaseException:
            if pending_name in recording:
                del recording[pending_name]
                recording.flush()
            raise

        old_existed = HAND_TRACKING_GROUP in recording
        try:
            if old_existed:
                recording.move(HAND_TRACKING_GROUP, backup_name)
            recording.move(pending_name, HAND_TRACKING_GROUP)
            recording.flush()
        except BaseException:
            if backup_name in recording:
                if HAND_TRACKING_GROUP in recording:
                    del recording[HAND_TRACKING_GROUP]
                recording.move(backup_name, HAND_TRACKING_GROUP)
            elif not old_existed and HAND_TRACKING_GROUP in recording:
                del recording[HAND_TRACKING_GROUP]
            if pending_name in recording:
                del recording[pending_name]
            recording.flush()
            raise
        if backup_name in recording:
            del recording[backup_name]
        recording.flush()

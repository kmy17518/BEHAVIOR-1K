"""Single-process tracking and retargeting workers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from dex_teleop.retargeting import HandRetargeter, ObservationRetargeter
from dex_teleop.tracking import (
    HandObservationFuser,
    HandTrackingSource,
    SourceUnavailableError,
    observation_to_hand_frame,
)
from dex_teleop.tracking.multimodal import HandTrackingSampleBatch
from dex_teleop.types import (
    FusedHandObservation,
    HandArticulationSample,
    HandFrame,
    Handedness,
    RetargetedHandCommand,
    WristPoseSample,
)

if TYPE_CHECKING:
    from dex_teleop.omnigibson.hand_tracking_recording import (
        ActionHandSelection,
        HandTrackingRecordingSession,
    )


@dataclass(frozen=True)
class RetargetingSnapshot:
    frame: HandFrame
    command: RetargetedHandCommand
    observation: FusedHandObservation | None = None
    articulation_stream: str | None = None
    articulation_row: int | None = None
    wrist_stream: str | None = None
    wrist_row: int | None = None
    retargeting_stream: str | None = None
    retargeting_row: int | None = None
    synchronization_skew_seconds: float | None = None
    native_wrist_stream: str | None = None
    native_wrist_row: int | None = None

    def action_selection(self) -> ActionHandSelection | None:
        """Return the exact recording rows used for this command, if attached."""

        required = (
            self.articulation_stream,
            self.articulation_row,
            self.wrist_stream,
            self.wrist_row,
            self.retargeting_stream,
            self.retargeting_row,
            self.synchronization_skew_seconds,
        )
        if all(value is None for value in required):
            return None
        if any(value is None for value in required):
            raise RuntimeError(
                "Retargeting snapshot has incomplete recording provenance"
            )
        from dex_teleop.omnigibson.hand_tracking_recording import ActionHandSelection

        return ActionHandSelection(
            articulation_stream=self.articulation_stream,
            articulation_row=self.articulation_row,
            wrist_stream=self.wrist_stream,
            wrist_row=self.wrist_row,
            retargeting_stream=self.retargeting_stream,
            retargeting_row=self.retargeting_row,
            synchronization_skew_seconds=self.synchronization_skew_seconds,
        )


class TrackingRetargetingWorker:
    """Run the selected source and optimizer in one background thread."""

    def __init__(
        self,
        source: HandTrackingSource,
        retargeter: HandRetargeter,
        handedness: Handedness,
        poll_interval: float = 0.002,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.source = source
        self.retargeter = retargeter
        self.handedness = handedness
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._first = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: RetargetingSnapshot | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TrackingRetargetingWorker has already been started")
        self.source.start()
        self._thread = threading.Thread(
            target=self._run, name="dex-teleop-retarget", daemon=True
        )
        self._thread.start()

    def snapshot(self, maximum_age: float | None = None) -> RetargetingSnapshot | None:
        self.check_health()
        with self._lock:
            latest = self._latest
        if latest is not None and maximum_age is not None:
            age = time.monotonic() - latest.frame.receipt_timestamp
            if age > maximum_age:
                raise SourceUnavailableError(
                    f"{latest.frame.source} hand frame is stale ({age:.3f}s > {maximum_age:.3f}s)"
                )
        return latest

    def wait_for_first(self, timeout: float) -> RetargetingSnapshot:
        if not self._first.wait(timeout=timeout):
            self.check_health()
            raise SourceUnavailableError(
                f"No {self.handedness.value}-hand frame received within {timeout:.1f}s"
            )
        snapshot = self.snapshot()
        assert snapshot is not None
        return snapshot

    def check_health(self) -> None:
        self.source.check_health()
        if self._error is not None:
            raise RuntimeError(
                f"Hand retargeting worker failed: {self._error}"
            ) from self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.source.close()

    def _run(self) -> None:
        last_timestamp = float("-inf")
        try:
            while not self._stop.is_set():
                frame = self.source.read(self.handedness)
                if frame is not None and frame.timestamp > last_timestamp:
                    command = self.retargeter.retarget(frame)
                    snapshot = RetargetingSnapshot(frame=frame, command=command)
                    with self._lock:
                        self._latest = snapshot
                    last_timestamp = frame.timestamp
                    self._first.set()
                self._stop.wait(self.poll_interval)
        except Exception as error:
            self._error = error
            self._first.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@dataclass
class _ControlRequest:
    operation: str
    value: Any = None
    complete: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _sample_identity(sample: HandArticulationSample | WristPoseSample) -> tuple:
    if sample.source_frame_id is not None or sample.source_timestamp_ns is not None:
        return ("source", sample.source_frame_id, sample.source_timestamp_ns)
    return ("capture", round(sample.timestamp * 1e9))


class MultiSourceTrackingWorker:
    """Fuse explicitly selected hand and wrist sources in one worker thread.

    The registry key is configuration identity; ``sample.source`` remains
    device provenance. Selected sources are never replaced by record-only
    sources. A physical receiver registered for both modalities is started,
    health-checked, and closed exactly once by object identity.
    """

    def __init__(
        self,
        sources: Mapping[str, object],
        articulation_source: str,
        wrist_source: str,
        retargeter: HandRetargeter | ObservationRetargeter,
        handedness: Handedness,
        *,
        record_articulation_sources: Iterable[str] = (),
        record_wrist_sources: Iterable[str] = (),
        fuser: HandObservationFuser | None = None,
        maximum_skew_seconds: float = 0.05,
        interpolate_wrist: bool = True,
        interpolation_wait_seconds: float | None = None,
        poll_interval: float = 0.002,
        recording_session: HandTrackingRecordingSession | None = None,
        retargeter_name: str | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if not sources:
            raise ValueError("sources must contain at least one named source")
        normalized_sources: dict[str, object] = {}
        for key, source in sources.items():
            if not isinstance(key, str) or not key or "/" in key:
                raise ValueError(
                    "Source registry keys must be non-empty and may not contain '/'"
                )
            if key in normalized_sources:
                raise ValueError(f"Duplicate source registry key {key!r}")
            normalized_sources[key] = source
        self.sources = MappingProxyType(normalized_sources)
        self.articulation_source = self._require_source_key(
            articulation_source, "selected articulation"
        )
        self.wrist_source = self._require_source_key(wrist_source, "selected wrist")
        record_articulation = _unique(record_articulation_sources)
        record_wrist = _unique(record_wrist_sources)
        for key in record_articulation:
            self._require_source_key(key, "record-only articulation")
        for key in record_wrist:
            self._require_source_key(key, "record-only wrist")
        self.record_articulation_sources = tuple(
            key for key in record_articulation if key != self.articulation_source
        )
        self.record_wrist_sources = tuple(
            key for key in record_wrist if key != self.wrist_source
        )
        self._articulation_keys = (
            self.articulation_source,
            *self.record_articulation_sources,
        )
        self._wrist_keys = (self.wrist_source, *self.record_wrist_sources)
        self._validate_source_contracts()

        if not callable(getattr(retargeter, "retarget", None)) and not callable(
            getattr(retargeter, "retarget_observation", None)
        ):
            raise TypeError(
                "retargeter must provide retarget(frame) or retarget_observation(observation)"
            )
        if not callable(getattr(retargeter, "reset", None)):
            raise TypeError("retargeter must provide reset()")
        self.retargeter = retargeter
        self.handedness = Handedness(handedness)
        self.poll_interval = float(poll_interval)
        self.fuser = fuser or HandObservationFuser(
            maximum_skew_seconds=maximum_skew_seconds,
            interpolate_wrist=interpolate_wrist,
        )
        if interpolation_wait_seconds is None:
            interpolation_wait_seconds = min(
                self.fuser.maximum_skew_seconds,
                1.0 / 60.0,
            )
        if (
            not math.isfinite(interpolation_wait_seconds)
            or interpolation_wait_seconds < 0.0
            or interpolation_wait_seconds > self.fuser.maximum_skew_seconds
        ):
            raise ValueError(
                "interpolation_wait_seconds must be finite, non-negative, and no "
                "greater than the fuser's maximum skew"
            )
        self.interpolation_wait_seconds = (
            float(interpolation_wait_seconds) if self.fuser.interpolate_wrist else 0.0
        )
        self.retargeter_name = self._normalize_retargeter_name(retargeter_name)

        self.articulation_streams = MappingProxyType(
            {key: f"articulation.{key}" for key in self._articulation_keys}
        )
        self.wrist_streams = MappingProxyType(
            {key: f"wrist.{key}" for key in self._wrist_keys}
        )
        self.control_wrist_stream = f"wrist.{self.wrist_source}.control"
        self.retargeting_stream = (
            f"{self.retargeter_name}.{self.articulation_source}+{self.wrist_source}"
        )

        self._lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._control_requests: deque[_ControlRequest] = deque()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._first = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_attempted = False
        self._closed = False
        self._started_sources: list[tuple[object, str]] = []
        self._latest: RetargetingSnapshot | None = None
        self._error: BaseException | None = None
        self._recording_session = recording_session

        self._last_articulation: dict[str, tuple[tuple, float, float]] = {}
        self._last_wrist: dict[str, tuple[tuple, float, float]] = {}
        self._seen_articulation_sources: set[str] = set()
        self._seen_wrist_sources: set[str] = set()
        self._stream_last_receipt: dict[tuple[str, str], float] = {}
        self._recorded_wrist_rows: dict[tuple[str, tuple], int] = {}
        self._pending_articulation: (
            tuple[HandArticulationSample, int | None, float] | None
        ) = None
        self._deferred_articulation: (
            tuple[HandArticulationSample, int | None] | None
        ) = None
        self._selected_wrist_source_name: str | None = None
        self._coemitted_wrists: dict[tuple, WristPoseSample] = {}
        self._source_signatures: dict[tuple[str, str], tuple] = {}
        self._validate_recording_session(recording_session)

    def _require_source_key(self, key: str, role: str) -> str:
        if key not in self.sources:
            raise ValueError(
                f"Unknown {role} source {key!r}; available sources: {list(self.sources)}"
            )
        return key

    def _validate_source_contracts(self) -> None:
        active_keys = set(self._articulation_keys).union(self._wrist_keys)
        for key in active_keys:
            source = self.sources[key]
            for method in ("start", "check_health", "close"):
                if not callable(getattr(source, method, None)):
                    raise TypeError(f"Source {key!r} must provide {method}()")
        for key in self._articulation_keys:
            if not callable(getattr(self.sources[key], "read_articulation", None)):
                raise TypeError(
                    f"Articulation source {key!r} must provide read_articulation()"
                )
        for key in self._wrist_keys:
            if not callable(getattr(self.sources[key], "read_wrist", None)):
                raise TypeError(f"Wrist source {key!r} must provide read_wrist()")

    def _normalize_retargeter_name(self, requested: str | None) -> str:
        if requested is None:
            requested = type(self.retargeter).__name__.strip("_").lower()
            if requested.endswith("retargeter") and requested != "retargeter":
                requested = requested.removesuffix("retargeter")
        if not requested or "/" in requested:
            raise ValueError(
                "retargeter_name must be non-empty and may not contain '/'"
            )
        return requested

    @staticmethod
    def _validate_recording_session(
        session: HandTrackingRecordingSession | None,
    ) -> None:
        if session is None:
            return
        for method in ("append_articulation", "append_wrist", "append_retargeted"):
            if not callable(getattr(session, method, None)):
                raise TypeError(f"recording_session must provide {method}()")
        if bool(getattr(session, "closed", False)):
            raise ValueError("Cannot attach a closed hand-tracking recording session")

    def _active_lifecycle_sources(self) -> list[tuple[object, str]]:
        active_keys = set(self._articulation_keys).union(self._wrist_keys)
        result = []
        seen = set()
        for key, source in self.sources.items():
            if key not in active_keys or id(source) in seen:
                continue
            seen.add(id(source))
            result.append((source, self._lifecycle_label(source)))
        return result

    def _roles_for_object(self, source: object) -> list[str]:
        roles = []
        for key in self._articulation_keys:
            if self.sources[key] is source:
                prefix = (
                    "selected" if key == self.articulation_source else "record-only"
                )
                roles.append(f"{prefix} articulation source {key!r}")
        for key in self._wrist_keys:
            if self.sources[key] is source:
                prefix = "selected" if key == self.wrist_source else "record-only"
                roles.append(f"{prefix} wrist source {key!r}")
        return roles

    def _lifecycle_label(self, source: object) -> str:
        return ", ".join(self._roles_for_object(source))

    def _role_label(self, modality: str, key: str) -> str:
        selected = (
            key == self.articulation_source
            if modality == "articulation"
            else key == self.wrist_source
        )
        return f"{'selected' if selected else 'record-only'} {modality} source {key!r}"

    def start(self) -> None:
        if self._start_attempted:
            raise RuntimeError("MultiSourceTrackingWorker has already been started")
        self._start_attempted = True
        lifecycle_sources = self._active_lifecycle_sources()
        started: list[tuple[object, str]] = []
        for source, label in lifecycle_sources:
            try:
                source.start()
            except Exception as error:
                rollback_errors = []
                for rollback_source, rollback_label in reversed(
                    [*started, (source, label)]
                ):
                    try:
                        rollback_source.close()
                    except Exception as close_error:
                        rollback_errors.append(f"{rollback_label}: {close_error}")
                failure = SourceUnavailableError(f"Failed to start {label}: {error}")
                if rollback_errors:
                    failure.add_note(
                        "Rollback close failures: " + "; ".join(rollback_errors)
                    )
                raise failure from error
            started.append((source, label))
        self._started_sources = started
        self._thread = threading.Thread(
            target=self._run,
            name="dex-teleop-multisource-retarget",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self, maximum_age: float | None = None) -> RetargetingSnapshot | None:
        if maximum_age is not None and maximum_age < 0:
            raise ValueError("maximum_age must be non-negative")
        self.check_health()
        with self._lock:
            latest = self._latest
        if latest is not None and maximum_age is not None:
            assert latest.observation is not None
            now = time.monotonic()
            components = (
                (
                    self._role_label("articulation", self.articulation_source),
                    latest.observation.articulation.receipt_timestamp,
                ),
                (
                    self._role_label("wrist", self.wrist_source),
                    latest.observation.wrist.receipt_timestamp,
                ),
            )
            for label, receipt_timestamp in components:
                age = now - receipt_timestamp
                if age > maximum_age:
                    raise SourceUnavailableError(
                        f"{label.capitalize()} is stale ({age:.3f}s > {maximum_age:.3f}s)"
                    )
        return latest

    def wait_for_first(self, timeout: float) -> RetargetingSnapshot:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if not self._first.wait(timeout=timeout):
            self.check_health()
            with self._lock:
                missing = [
                    self._role_label("articulation", key)
                    for key in self._articulation_keys
                    if key not in self._seen_articulation_sources
                ]
                missing.extend(
                    self._role_label("wrist", key)
                    for key in self._wrist_keys
                    if key not in self._seen_wrist_sources
                )
            missing_detail = (
                "; missing initial samples from " + ", ".join(missing)
                if missing
                else "; all configured streams produced samples, but no in-skew fused observation was available"
            )
            raise SourceUnavailableError(
                f"No ready fused {self.handedness.value}-hand observation received within {timeout:.1f}s "
                f"from selected articulation source {self.articulation_source!r} and "
                f"selected wrist source {self.wrist_source!r}{missing_detail}"
            )
        snapshot = self.snapshot()
        if snapshot is None:
            self.check_health()
            raise SourceUnavailableError(
                "Worker signaled without publishing a fused observation"
            )
        return snapshot

    def check_health(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(
                f"Multi-source hand retargeting worker failed: {error}"
            ) from error
        for source, label in self._active_lifecycle_sources():
            try:
                source.check_health()
            except Exception as source_error:
                raise SourceUnavailableError(
                    f"Unhealthy {label}: {source_error}"
                ) from source_error

    def check_stream_freshness(self, maximum_age: float) -> None:
        """Require a recent sample from every selected and record-only role."""

        if not math.isfinite(maximum_age) or maximum_age < 0.0:
            raise ValueError("maximum_age must be finite and non-negative")
        self.check_health()
        now = time.monotonic()
        with self._lock:
            receipts = dict(self._stream_last_receipt)
        problems = []
        for modality, keys in (
            ("articulation", self._articulation_keys),
            ("wrist", self._wrist_keys),
        ):
            for key in keys:
                label = self._role_label(modality, key)
                receipt = receipts.get((modality, key))
                if receipt is None:
                    problems.append(f"{label} has not produced a sample")
                    continue
                age = now - receipt
                if age > maximum_age:
                    problems.append(
                        f"{label} is stale ({age:.3f}s > {maximum_age:.3f}s)"
                    )
        if problems:
            raise SourceUnavailableError(
                "Hand-tracking stream freshness check failed: " + "; ".join(problems)
            )

    @property
    def recording_session(self) -> HandTrackingRecordingSession | None:
        with self._lock:
            return self._recording_session

    def set_recording_session(
        self,
        session: HandTrackingRecordingSession | None,
        *,
        timeout: float = 2.0,
    ) -> None:
        """Attach or detach a per-task collector at a worker-thread boundary."""

        self._validate_recording_session(session)
        if not self._start_attempted:
            with self._lock:
                self._recording_session = session
            self._clear_recording_rows()
            return
        self._submit_control("recording", session, timeout)

    def reset(self, *, timeout: float = 2.0) -> None:
        """Synchronously reset fusion and retargeter state on the worker thread."""

        self._submit_control("reset", None, timeout)

    def _submit_control(self, operation: str, value: Any, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        thread = self._thread
        if thread is None or not thread.is_alive():
            self.check_health()
            raise RuntimeError("MultiSourceTrackingWorker is not running")
        if threading.current_thread() is thread:
            self._apply_control(operation, value)
            return
        request = _ControlRequest(operation=operation, value=value)
        with self._control_lock:
            self._control_requests.append(request)
        self._wake.set()
        if not request.complete.wait(timeout):
            raise TimeoutError(f"Timed out waiting for worker operation {operation!r}")
        if request.error is not None:
            raise RuntimeError(
                f"Worker operation {operation!r} failed: {request.error}"
            ) from request.error

    def _drain_control_requests(self) -> None:
        while True:
            with self._control_lock:
                if not self._control_requests:
                    return
                request = self._control_requests.popleft()
            try:
                self._apply_control(request.operation, request.value)
            except BaseException as error:
                request.error = error
            finally:
                request.complete.set()

    def _apply_control(self, operation: str, value: Any) -> None:
        if operation == "reset":
            self._drain_native_before_boundary()
            self.fuser.reset()
            self.retargeter.reset()
            self._pending_articulation = None
            self._deferred_articulation = None
            self._coemitted_wrists.clear()
            with self._lock:
                self._latest = None
                self._seen_articulation_sources.clear()
                self._seen_wrist_sources.clear()
                self._stream_last_receipt.clear()
                self._first.clear()
            return
        if operation == "recording":
            with self._lock:
                current = self._recording_session
            if current is value:
                return
            # Samples acquired before this synchronous boundary belong to the
            # old session (if any), never to the newly attached session.
            self._drain_native_before_boundary()
            self.fuser.reset()
            self._pending_articulation = None
            self._deferred_articulation = None
            self._coemitted_wrists.clear()
            self._clear_recording_rows()
            with self._lock:
                self._recording_session = value
                self._latest = None
                self._seen_articulation_sources.clear()
                self._seen_wrist_sources.clear()
                self._stream_last_receipt.clear()
                self._first.clear()
            return
        raise ValueError(f"Unknown worker control operation {operation!r}")

    def _clear_recording_rows(self) -> None:
        self._recorded_wrist_rows.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        close_errors = []
        for source, label in reversed(self._started_sources):
            try:
                source.close()
            except Exception as error:
                close_errors.append(f"{label}: {error}")
        self._started_sources.clear()
        if close_errors:
            raise SourceUnavailableError("Failed to close " + "; ".join(close_errors))

    def _is_new_sample(
        self,
        sample: HandArticulationSample | WristPoseSample,
        key: str,
        modality: str,
    ) -> bool:
        if sample.handedness != self.handedness:
            raise ValueError(
                f"returned {sample.handedness.value} data for requested {self.handedness.value} hand"
            )
        identity = _sample_identity(sample)
        state = (
            self._last_articulation if modality == "articulation" else self._last_wrist
        )
        previous = state.get(key)
        if previous is not None and identity == previous[0]:
            return False
        if previous is not None and sample.timestamp <= previous[1]:
            raise ValueError(
                f"capture timestamp did not increase ({sample.timestamp} <= {previous[1]})"
            )
        if previous is not None and sample.receipt_timestamp < previous[2]:
            raise ValueError(
                f"receipt timestamp moved backwards ({sample.receipt_timestamp} < {previous[2]})"
            )
        signature = (
            sample.source,
            *(
                (sample.schema, sample.coordinate_frame)
                if isinstance(sample, HandArticulationSample)
                else (sample.reference_frame, sample.anatomical_frame)
            ),
        )
        signature_key = (modality, key)
        old_signature = self._source_signatures.setdefault(signature_key, signature)
        if signature != old_signature:
            raise ValueError(
                f"changed source/schema identity from {old_signature!r} to {signature!r}"
            )
        state[key] = (identity, sample.timestamp, sample.receipt_timestamp)
        return True

    def _mark_seen(self, modality: str, key: str, receipt_timestamp: float) -> None:
        with self._lock:
            if modality == "articulation":
                self._seen_articulation_sources.add(key)
            else:
                self._seen_wrist_sources.add(key)
            self._stream_last_receipt[(modality, key)] = receipt_timestamp

    def _maybe_signal_first(self) -> None:
        with self._lock:
            ready = (
                self._latest is not None
                and set(self._articulation_keys).issubset(
                    self._seen_articulation_sources
                )
                and set(self._wrist_keys).issubset(self._seen_wrist_sources)
            )
        if ready:
            self._first.set()

    def _record_articulation(
        self, key: str, sample: HandArticulationSample
    ) -> int | None:
        with self._lock:
            session = self._recording_session
        if session is None:
            return None
        return session.append_articulation(self.articulation_streams[key], sample)

    def _record_wrist(self, key: str, sample: WristPoseSample) -> int | None:
        with self._lock:
            session = self._recording_session
        if session is None:
            return None
        row = session.append_wrist(self.wrist_streams[key], sample)
        self._recorded_wrist_rows[(key, _sample_identity(sample))] = row
        return row

    def _drain_combined_sources(
        self,
    ) -> tuple[
        dict[str, tuple[HandArticulationSample, ...]],
        dict[str, tuple[WristPoseSample, ...]],
    ]:
        """Drain paired receivers once and route the batch to configured roles."""

        self._coemitted_wrists.clear()
        articulation_batches: dict[str, tuple[HandArticulationSample, ...]] = {}
        wrist_batches: dict[str, tuple[WristPoseSample, ...]] = {}
        seen_objects: set[int] = set()
        for source, label in self._active_lifecycle_sources():
            if id(source) in seen_objects:
                continue
            seen_objects.add(id(source))
            drain = getattr(source, "drain_hand_tracking", None)
            if not callable(drain):
                continue
            try:
                batch = drain(self.handedness)
            except Exception as error:
                raise SourceUnavailableError(
                    f"Failed to drain {label}: {error}"
                ) from error
            if not isinstance(batch, HandTrackingSampleBatch):
                raise TypeError(
                    f"Combined source {label} returned {type(batch).__name__}, "
                    "not HandTrackingSampleBatch"
                )
            articulations = tuple(batch.articulations)
            wrists = tuple(batch.wrists)
            if (
                self.sources[self.articulation_source] is source
                and self.sources[self.wrist_source] is source
            ):
                if len(articulations) != len(wrists):
                    raise ValueError(
                        f"Combined selected source {label} returned {len(articulations)} "
                        f"articulations and {len(wrists)} wrists"
                    )
                for articulation, wrist in zip(articulations, wrists, strict=True):
                    articulation_identity = _sample_identity(articulation)
                    wrist_identity = _sample_identity(wrist)
                    if (
                        articulation_identity != wrist_identity
                        or articulation.timestamp != wrist.timestamp
                    ):
                        raise ValueError(
                            f"Combined selected source {label} emitted a mismatched "
                            "articulation/wrist callback pair"
                        )
                    self._coemitted_wrists[articulation_identity] = wrist
            for key in self._articulation_keys:
                if self.sources[key] is source:
                    articulation_batches[key] = articulations
            for key in self._wrist_keys:
                if self.sources[key] is source:
                    wrist_batches[key] = wrists
        return articulation_batches, wrist_batches

    def _drain_native_before_boundary(self) -> None:
        """Consume pre-boundary native samples without producing a command.

        Raw samples are appended to the currently attached session.  Clearing
        fusion and pending control state immediately afterwards prevents those
        samples from leaking across a reset or session attachment boundary.
        """

        articulation_batches, wrist_batches = self._drain_combined_sources()
        self._poll_wrist_sources(wrist_batches)
        self._poll_articulation_sources(articulation_batches)

    def _poll_wrist_sources(
        self,
        preloaded: Mapping[str, tuple[WristPoseSample, ...]] | None = None,
    ) -> None:
        preloaded = {} if preloaded is None else preloaded
        drained_by_object: dict[int, tuple[WristPoseSample, ...]] = {}
        for key in self._wrist_keys:
            label = self._role_label("wrist", key)
            try:
                source = self.sources[key]
                if key in preloaded:
                    samples = preloaded[key]
                else:
                    drain = getattr(source, "drain_wrists", None)
                    if callable(drain):
                        object_id = id(source)
                        if object_id not in drained_by_object:
                            drained_by_object[object_id] = tuple(drain(self.handedness))
                        samples = drained_by_object[object_id]
                    else:
                        sample = source.read_wrist(self.handedness)
                        samples = () if sample is None else (sample,)
                for sample in samples:
                    if not isinstance(sample, WristPoseSample):
                        raise TypeError(
                            f"returned {type(sample).__name__}, not WristPoseSample"
                        )
                    if not self._is_new_sample(sample, key, "wrist"):
                        continue
                    self._record_wrist(key, sample)
                    self._mark_seen("wrist", key, sample.receipt_timestamp)
                    if key == self.wrist_source:
                        if (
                            self._selected_wrist_source_name is not None
                            and sample.source != self._selected_wrist_source_name
                        ):
                            raise ValueError(
                                f"changed sample source from {self._selected_wrist_source_name!r} "
                                f"to {sample.source!r}"
                            )
                        self._selected_wrist_source_name = sample.source
                        self.fuser.add_wrist(sample)
                    self._maybe_signal_first()
            except Exception as error:
                raise SourceUnavailableError(
                    f"{label.capitalize()} failed: {error}"
                ) from error

    def _poll_articulation_sources(
        self,
        preloaded: Mapping[str, tuple[HandArticulationSample, ...]] | None = None,
    ) -> None:
        preloaded = {} if preloaded is None else preloaded
        drained_by_object: dict[int, tuple[HandArticulationSample, ...]] = {}
        for key in self._articulation_keys:
            label = self._role_label("articulation", key)
            try:
                source = self.sources[key]
                if key in preloaded:
                    samples = preloaded[key]
                else:
                    drain = getattr(source, "drain_articulations", None)
                    if callable(drain):
                        object_id = id(source)
                        if object_id not in drained_by_object:
                            drained_by_object[object_id] = tuple(drain(self.handedness))
                        samples = drained_by_object[object_id]
                    else:
                        sample = source.read_articulation(self.handedness)
                        samples = () if sample is None else (sample,)
                newest_selected: tuple[HandArticulationSample, int | None] | None = None
                for sample in samples:
                    if not isinstance(sample, HandArticulationSample):
                        raise TypeError(
                            f"returned {type(sample).__name__}, not HandArticulationSample"
                        )
                    if not self._is_new_sample(sample, key, "articulation"):
                        continue
                    row = self._record_articulation(key, sample)
                    self._mark_seen("articulation", key, sample.receipt_timestamp)
                    if key == self.articulation_source:
                        newest_selected = (sample, row)
                    self._maybe_signal_first()
                if newest_selected is not None:
                    # Preserve every native sample in recording, but retarget
                    # one pending sample without resetting its bounded
                    # interpolation deadline. Keep only the newest sample from
                    # this native batch as the deferred control candidate.
                    sample, row = newest_selected
                    if self._pending_articulation is None:
                        self._pending_articulation = (
                            sample,
                            row,
                            time.monotonic() + self.interpolation_wait_seconds,
                        )
                    else:
                        self._deferred_articulation = (sample, row)
            except Exception as error:
                raise SourceUnavailableError(
                    f"{label.capitalize()} failed: {error}"
                ) from error

    def _record_control_wrist(
        self,
        observation: FusedHandObservation,
    ) -> tuple[str | None, int | None, str | None, int | None]:
        with self._lock:
            session = self._recording_session
        if session is None:
            return None, None, None, None
        native_stream = self.wrist_streams[self.wrist_source]
        native_row = self._recorded_wrist_rows.get(
            (self.wrist_source, _sample_identity(observation.wrist))
        )
        if native_row is not None:
            return native_stream, native_row, native_stream, native_row
        wrist = observation.wrist
        control_sample = WristPoseSample(
            timestamp=observation.articulation.timestamp,
            receipt_timestamp=observation.receipt_timestamp,
            source_timestamp_ns=wrist.source_timestamp_ns,
            source_frame_id=wrist.source_frame_id,
            handedness=wrist.handedness,
            position=wrist.position,
            quaternion_xyzw=wrist.quaternion_xyzw,
            source=f"fusion:{self.wrist_source}",
            reference_frame=wrist.reference_frame,
            anatomical_frame=wrist.anatomical_frame,
            confidence=wrist.confidence,
            provenance=wrist.provenance,
        )
        row = session.append_wrist(self.control_wrist_stream, control_sample)
        return self.control_wrist_stream, row, None, None

    def _publish_pending(self) -> None:
        if (
            self._pending_articulation is None
            or self._selected_wrist_source_name is None
        ):
            return
        articulation, articulation_row, interpolation_deadline = (
            self._pending_articulation
        )
        paired_wrist = self._coemitted_wrists.pop(
            _sample_identity(articulation),
            None,
        )
        observation = (
            self.fuser.fuse_paired(articulation, paired_wrist)
            if paired_wrist is not None
            else self.fuser.fuse(
                articulation,
                wrist_source=self._selected_wrist_source_name,
            )
        )
        if observation is None:
            return
        if (
            observation.wrist.timestamp != articulation.timestamp
            and time.monotonic() < interpolation_deadline
        ):
            # The current result is a nearest-only fallback.  Keep the
            # articulation briefly so an exact sample or future bracket can
            # arrive, then use the nearest result once the deadline expires.
            return
        self._pending_articulation = None
        frame = observation_to_hand_frame(observation)
        retarget_observation = getattr(self.retargeter, "retarget_observation", None)
        command = (
            retarget_observation(observation)
            if callable(retarget_observation)
            else self.retargeter.retarget(frame)
        )
        wrist_stream = wrist_row = native_wrist_stream = native_wrist_row = None
        retargeting_row = None
        articulation_stream = None
        with self._lock:
            session = self._recording_session
        if session is not None:
            if articulation_row is None:
                raise RuntimeError("Selected articulation is missing its recording row")
            articulation_stream = self.articulation_streams[self.articulation_source]
            (
                wrist_stream,
                wrist_row,
                native_wrist_stream,
                native_wrist_row,
            ) = self._record_control_wrist(observation)
            from dex_teleop.omnigibson.hand_tracking_recording import (
                RetargetedCommandSample,
            )

            retargeting_row = session.append_retargeted(
                self.retargeting_stream,
                RetargetedCommandSample(
                    command=command,
                    retargeter=self.retargeter_name,
                    articulation_stream=articulation_stream,
                    articulation_row=articulation_row,
                    wrist_stream=wrist_stream,
                    wrist_row=wrist_row,
                ),
            )
        snapshot = RetargetingSnapshot(
            frame=frame,
            command=command,
            observation=observation,
            articulation_stream=articulation_stream,
            articulation_row=articulation_row if session is not None else None,
            wrist_stream=wrist_stream,
            wrist_row=wrist_row,
            retargeting_stream=self.retargeting_stream if session is not None else None,
            retargeting_row=retargeting_row,
            synchronization_skew_seconds=(
                observation.synchronization_skew_seconds
                if session is not None
                else None
            ),
            native_wrist_stream=native_wrist_stream,
            native_wrist_row=native_wrist_row,
        )
        with self._lock:
            self._latest = snapshot
        self._maybe_signal_first()
        if self._deferred_articulation is not None:
            sample, row = self._deferred_articulation
            self._deferred_articulation = None
            self._pending_articulation = (
                sample,
                row,
                time.monotonic() + self.interpolation_wait_seconds,
            )

    def _fail_pending_controls(self, error: BaseException) -> None:
        with self._control_lock:
            requests = tuple(self._control_requests)
            self._control_requests.clear()
        for request in requests:
            request.error = error
            request.complete.set()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_control_requests()
                if self._stop.is_set():
                    break
                articulation_batches, wrist_batches = self._drain_combined_sources()
                self._poll_wrist_sources(wrist_batches)
                # A wrist that arrived while the previous articulation was
                # waiting should be allowed to complete it before a newer
                # articulation replaces the real-time control candidate.
                self._publish_pending()
                self._poll_articulation_sources(articulation_batches)
                self._publish_pending()
                self._wake.wait(self.poll_interval)
                self._wake.clear()
        except BaseException as error:
            with self._lock:
                self._error = error
            self._fail_pending_controls(error)
            self._first.set()
        finally:
            self._fail_pending_controls(
                RuntimeError("MultiSourceTrackingWorker has stopped")
            )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

"""Single-process tracking and retargeting worker."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from dex_teleop.retargeting import LandmarkRetargeter
from dex_teleop.tracking import HandTrackingSource, SourceUnavailableError
from dex_teleop.types import HandFrame, Handedness, RetargetedHandCommand


@dataclass(frozen=True)
class RetargetingSnapshot:
    frame: HandFrame
    command: RetargetedHandCommand


class TrackingRetargetingWorker:
    """Run the selected source and optimizer in one background thread."""

    def __init__(
        self,
        source: HandTrackingSource,
        retargeter: LandmarkRetargeter,
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
        self._thread = threading.Thread(target=self._run, name="dex-teleop-retarget", daemon=True)
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
            raise SourceUnavailableError(f"No {self.handedness.value}-hand frame received within {timeout:.1f}s")
        snapshot = self.snapshot()
        assert snapshot is not None
        return snapshot

    def check_health(self) -> None:
        self.source.check_health()
        if self._error is not None:
            raise RuntimeError(f"Hand retargeting worker failed: {self._error}") from self._error

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

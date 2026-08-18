"""Tracking-source contract used by the teleoperation runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dex_teleop.types import HandFrame, Handedness


class SourceUnavailableError(RuntimeError):
    """Raised when an explicitly selected tracking source cannot be used."""


class HandTrackingSource(ABC):
    """A lifecycle-managed producer of canonical hand frames."""

    @abstractmethod
    def start(self) -> None:
        """Start acquiring data, raising if acquisition cannot start."""

    @abstractmethod
    def read(self, handedness: Handedness) -> HandFrame | None:
        """Return the latest complete frame, or None before the first frame."""

    @abstractmethod
    def check_health(self) -> None:
        """Raise when the source's receiver has failed."""

    @abstractmethod
    def close(self) -> None:
        """Stop acquisition and release resources."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

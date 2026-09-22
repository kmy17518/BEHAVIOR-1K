"""Contracts for independently acquired hand articulation and wrist pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dex_teleop.types import HandArticulationSample, Handedness, WristPoseSample


@dataclass(frozen=True)
class HandTrackingSampleBatch:
    """All unseen samples drained atomically from one multimodal receiver.

    Batches are capture-ordered and destructive: a later drain returns only
    samples acquired after this batch.  Sources must report bounded-buffer
    overflow through ``check_health()`` instead of silently dropping samples.
    """

    articulations: tuple[HandArticulationSample, ...] = ()
    wrists: tuple[WristPoseSample, ...] = ()


@runtime_checkable
class HandArticulationSource(Protocol):
    """Lifecycle-managed producer of wrist-local named hand skeletons."""

    def start(self) -> None:
        """Start acquisition, raising if the source cannot start."""

    def read_articulation(self, handedness: Handedness) -> HandArticulationSample | None:
        """Return the latest articulation sample for one hand, if available."""

    def check_health(self) -> None:
        """Raise when acquisition has failed."""

    def close(self) -> None:
        """Stop acquisition and release resources."""


@runtime_checkable
class WristPoseSource(Protocol):
    """Lifecycle-managed producer of anatomical wrist poses."""

    def start(self) -> None:
        """Start acquisition, raising if the source cannot start."""

    def read_wrist(self, handedness: Handedness) -> WristPoseSample | None:
        """Return the latest wrist pose for one hand, if available."""

    def check_health(self) -> None:
        """Raise when acquisition has failed."""

    def close(self) -> None:
        """Stop acquisition and release resources."""


@runtime_checkable
class DrainingHandArticulationSource(HandArticulationSource, Protocol):
    """Articulation source that preserves every unseen sample in a bounded queue."""

    def drain_articulations(
        self, handedness: Handedness
    ) -> tuple[HandArticulationSample, ...]:
        """Consume and return all unseen samples in capture order."""


@runtime_checkable
class DrainingWristPoseSource(WristPoseSource, Protocol):
    """Wrist source that preserves every unseen sample in a bounded queue."""

    def drain_wrists(self, handedness: Handedness) -> tuple[WristPoseSample, ...]:
        """Consume and return all unseen samples in capture order."""


@runtime_checkable
class MultimodalHandTrackingSource(HandArticulationSource, WristPoseSource, Protocol):
    """A single physical receiver that provides both tracking modalities."""


@runtime_checkable
class DrainingMultimodalHandTrackingSource(MultimodalHandTrackingSource, Protocol):
    """Multimodal source whose paired component queues are drained atomically."""

    def drain_hand_tracking(self, handedness: Handedness) -> HandTrackingSampleBatch:
        """Consume all unseen articulation and wrist samples in one source lock."""

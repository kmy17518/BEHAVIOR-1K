"""Common retargeter contract and construction helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dex_teleop.types import FusedHandObservation, HandFrame, RetargetedHandCommand


SUPPORTED_RETARGETERS = ("adaptive", "dexpilot")


class RetargeterUnavailableError(RuntimeError):
    """Raised when an explicitly selected optional retargeter is unavailable."""


@runtime_checkable
class HandRetargeter(Protocol):
    """Stateful hand-frame to named robot-joint command contract."""

    hand_model: str
    hand_side: str
    joint_names: tuple[str, ...]

    def retarget(self, frame: HandFrame) -> RetargetedHandCommand:
        """Retarget one complete frame."""

    def reset(self) -> None:
        """Clear warm-start and filtering state."""


@runtime_checkable
class ObservationRetargeter(Protocol):
    """Optional rich-observation contract for orientation-aware backends."""

    hand_model: str
    hand_side: str
    joint_names: tuple[str, ...]

    def retarget_observation(self, observation: FusedHandObservation) -> RetargetedHandCommand:
        """Retarget without discarding the named articulation representation."""

    def reset(self) -> None:
        """Clear warm-start and filtering state."""


def create_hand_retargeter(
    name: str, hand_model: str, hand_side: str = "right"
) -> HandRetargeter | ObservationRetargeter:
    """Create one explicitly selected retargeter without importing optional backends eagerly."""

    normalized = name.lower()
    if normalized == "adaptive":
        from dex_teleop.retargeting.retargeter import LandmarkRetargeter

        return LandmarkRetargeter.from_hand_model(hand_model, hand_side=hand_side)
    if normalized == "dexpilot":
        from dex_teleop.retargeting.dexpilot import DexPilotRetargeter

        return DexPilotRetargeter.from_hand_model(hand_model, hand_side=hand_side)
    raise ValueError(f"Unsupported retargeter {name!r}; choose from {SUPPORTED_RETARGETERS}")

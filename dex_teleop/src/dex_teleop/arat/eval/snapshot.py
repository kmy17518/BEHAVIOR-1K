"""Per-step evaluation snapshot.

An :class:`AratStepSnapshot` is everything the scorer sees about one engaged control
step. The live extractor (:mod:`dex_teleop.arat.eval.live`) builds these from the
OmniGibson sim; tests build them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class HandContact:
    """One contact between a hand link and this item's object of interest."""

    finger: str  # "thumb" | "index" | "middle" | "ring" | "pinky" | "palm"
    part: str  # "pad" | "dorsal" | "palm"
    link_name: str
    other_link: str = ""
    position: tuple[float, float, float] | None = None
    normal: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SupportContact:
    """One contact between the tracked object and a non-robot body."""

    other_object: str
    other_link: str
    relation: str  # "below" | "side" | "above"


@dataclass(frozen=True)
class TrackedObjectState:
    position: tuple[float, float, float]
    aabb_lo: tuple[float, float, float]
    aabb_hi: tuple[float, float, float]
    speed: float
    supports: tuple[SupportContact, ...] = ()


@dataclass(frozen=True)
class WaterState:
    n_total: int
    n_in_source: int
    n_in_dest: int

    @property
    def n_outside(self) -> int:
        return max(0, self.n_total - self.n_in_source - self.n_in_dest)


@dataclass(frozen=True)
class TargetState:
    """Placement status with respect to this item's target, computed by the extractor."""

    at_target: bool = False
    reached_target_height: bool = False
    # Gross-movement region contact classification
    palmar_region_contact: bool = False
    dorsal_region_contact: bool = False
    # Gross movement: how far the hand has advanced toward the target region (meters)
    approach_progress_m: float = 0.0


@dataclass(frozen=True)
class AratStepSnapshot:
    """State of one engaged control step, in task time."""

    t: float
    hand_contacts: tuple[HandContact, ...] = ()
    tracked: TrackedObjectState | None = None
    apertures: Mapping[str, float] = field(default_factory=dict)  # finger -> thumb-pad distance (m)
    target: TargetState = field(default_factory=TargetState)
    water: WaterState | None = None

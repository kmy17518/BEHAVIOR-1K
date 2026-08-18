"""Stateless per-step summaries and small stateful milestone detectors."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dex_teleop.arat.eval.snapshot import AratStepSnapshot, TrackedObjectState


@dataclass(frozen=True)
class ContactSummary:
    """Which hand parts touch the object of interest this step."""

    pads: frozenset[str]
    dorsals: frozenset[str]
    palm: bool

    @property
    def any(self) -> bool:
        return bool(self.pads) or bool(self.dorsals) or self.palm

    @property
    def pads_only_fingers(self) -> frozenset[str]:
        return self.pads


def summarize_hand_contacts(snapshot: AratStepSnapshot) -> ContactSummary:
    pads = set()
    dorsals = set()
    palm = False
    for contact in snapshot.hand_contacts:
        if contact.part == "pad":
            pads.add(contact.finger)
        elif contact.part == "palm":
            palm = True
        else:
            dorsals.add(contact.finger)
    return ContactSummary(pads=frozenset(pads), dorsals=frozenset(dorsals), palm=palm)


def is_env_supported(tracked: TrackedObjectState) -> bool:
    """Whether any non-robot body supports the object from below."""
    return any(support.relation == "below" for support in tracked.supports)


def is_side_contacting(tracked: TrackedObjectState) -> bool:
    """Whether the object is laterally pressed against any non-robot body."""
    return any(support.relation == "side" for support in tracked.supports)


class VoluntaryOpeningTracker:
    """Latches whether the hand voluntarily opened to the object's size before contact.

    Tracks the maximum aperture (thumb-pad-to-finger-pad distance over the rubric's
    relevant fingers) seen while the hand was NOT touching the object — both before the
    first contact and between attempts.
    """

    def __init__(self, required_aperture: float | None, relevant_fingers: tuple[str, ...]):
        self._required = required_aperture
        self._fingers = relevant_fingers
        self.max_open_aperture = 0.0

    def step(self, snapshot: AratStepSnapshot, contacts: ContactSummary) -> None:
        if self._required is None or contacts.any:
            return
        apertures = [snapshot.apertures[f] for f in self._fingers if f in snapshot.apertures]
        if apertures:
            self.max_open_aperture = max(self.max_open_aperture, max(apertures))

    @property
    def satisfied(self) -> bool:
        if self._required is None:
            return True
        return self.max_open_aperture >= self._required


class DorsumPushTracker:
    """Latches when the object is displaced by dorsal-only contact while never held.

    Pushing the object across the surface with the dorsum of the hand does not
    constitute partial completion (scoring guide, score 1 notes).
    """

    def __init__(self, displacement_threshold_m: float = 0.03):
        self._threshold = displacement_threshold_m
        self._accumulated = 0.0
        self._previous_xy: tuple[float, float] | None = None
        self.detected = False

    def step(self, tracked: TrackedObjectState, contacts: ContactSummary, ever_held: bool) -> None:
        xy = (tracked.position[0], tracked.position[1])
        previous, self._previous_xy = self._previous_xy, xy
        if ever_held or self.detected:
            return
        dorsal_only = contacts.any and not contacts.pads and not contacts.palm
        if dorsal_only and is_env_supported(tracked) and previous is not None:
            self._accumulated += math.hypot(xy[0] - previous[0], xy[1] - previous[1])
            if self._accumulated >= self._threshold:
                self.detected = True
        elif not dorsal_only:
            self._accumulated = 0.0


class BracingTracker:
    """Flags sustained lateral bracing of the object against a fixed body during grasp
    formation (attempting to manipulate the object into the hand by stabilizing it).

    This is a heuristic and is recorded as an advisory reason, never an automatic 0.
    """

    def __init__(self, dt: float, min_duration_s: float = 0.5, min_speed: float = 0.02):
        self._dt = dt
        self._min_duration = min_duration_s
        self._min_speed = min_speed
        self._elapsed = 0.0
        self.detected = False

    def step(self, tracked: TrackedObjectState, contacts: ContactSummary, held_now: bool) -> None:
        bracing_now = (
            contacts.any and not held_now and is_side_contacting(tracked) and tracked.speed > self._min_speed
        )
        if bracing_now:
            self._elapsed += self._dt
            if self._elapsed >= self._min_duration:
                self.detected = True
        else:
            self._elapsed = 0.0


class WaterTracker:
    """Tracks water transfer and spill for the pouring item.

    ``spilled`` is the number of particles outside both cups. During the pour some
    particles are legitimately in flight, so the authoritative spill count is the value
    at completion / item end; the running maximum is kept for diagnostics only.
    """

    def __init__(self) -> None:
        self.n_total = 0
        self.n_in_source = 0
        self.n_in_dest = 0
        self.max_outside_seen = 0

    def step(self, water) -> None:
        self.n_total = water.n_total
        self.n_in_source = water.n_in_source
        self.n_in_dest = water.n_in_dest
        self.max_outside_seen = max(self.max_outside_seen, water.n_outside)

    @property
    def spilled(self) -> int:
        return max(0, self.n_total - self.n_in_source - self.n_in_dest)

    def transferred(self, empty_tolerance: int, min_transfer_fraction: float) -> bool:
        if self.n_total == 0:
            return False
        emptied = self.n_in_source <= empty_tolerance
        received = self.n_in_dest >= max(1, int(self.n_total * min_transfer_fraction))
        return emptied and received

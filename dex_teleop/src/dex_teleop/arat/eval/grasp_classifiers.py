"""Per-item hand-movement classification from pad/dorsal/palm contact sets.

Each classifier answers: does the current contact pattern constitute the "appropriate
hand movement components" the scoring guide requires for this item? Classification is
per step; the scorer aggregates over the hold window.

Reason strings are stable identifiers surfaced in the item result:
- ``wrong_opposition`` (pinch): a different finger pair or palm-hold was used — hard 0.
- ``pads_not_used`` (pinch): correct finger involved but via its dorsum/side — caps at 2.
- Other reasons describe why the pattern is not the appropriate one — caps at 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dex_teleop.arat.eval.detectors import ContactSummary
from dex_teleop.arat.eval.hand_model import FINGERS


@dataclass(frozen=True)
class GraspAssessment:
    appropriate: bool
    reasons: tuple[str, ...] = ()


_NON_THUMB = tuple(f for f in FINGERS if f != "thumb")


def _fingers_involved(contacts: ContactSummary) -> frozenset[str]:
    return frozenset(contacts.pads | contacts.dorsals)


def classify_step(grasp_config: Mapping, contacts: ContactSummary) -> GraspAssessment:
    grasp_class = grasp_config["class"]
    if grasp_class in ("opposition", "pads_opposition"):
        return _classify_opposition(contacts)
    if grasp_class == "spherical":
        return _classify_spherical(contacts)
    if grasp_class == "lateral":
        return _classify_lateral(contacts)
    if grasp_class == "cylindrical":
        return _classify_cylindrical(contacts)
    if grasp_class == "pincer":
        return _classify_pincer(contacts)
    if grasp_class == "pinch":
        return _classify_pinch(contacts, grasp_config["finger"])
    if grasp_class == "palmar":
        # Gross-movement items: no grasp requirement (the hand need not be open)
        return GraspAssessment(appropriate=True)
    raise ValueError(f"Unknown grasp class {grasp_class!r}")


def _classify_opposition(contacts: ContactSummary) -> GraspAssessment:
    """Blocks/tubes: any grasp with the thumb (pad) opposed to one or more finger pads."""
    reasons = []
    if "thumb" not in contacts.pads:
        reasons.append("thumb_not_opposed")
    if not any(f in contacts.pads for f in _NON_THUMB):
        reasons.append("no_finger_pad_opposition")
    return GraspAssessment(appropriate=not reasons, reasons=tuple(reasons))


def _classify_spherical(contacts: ContactSummary) -> GraspAssessment:
    """Cricket ball: fingers and thumb spread around the ball (thumb pad + >=2 finger pads)."""
    reasons = []
    if "thumb" not in contacts.pads:
        reasons.append("thumb_not_opposed")
    if len([f for f in _NON_THUMB if f in contacts.pads]) < 2:
        reasons.append("spherical_grasp_not_formed")
    return GraspAssessment(appropriate=not reasons, reasons=tuple(reasons))


def _classify_lateral(contacts: ContactSummary) -> GraspAssessment:
    """Sharpening stone: lateral grip between the thumb pad and the side of the index."""
    reasons = []
    if "thumb" not in contacts.pads:
        reasons.append("thumb_not_opposed")
    if "index" not in contacts.dorsals:
        if "index" in contacts.pads:
            reasons.append("pinch_instead_of_lateral")
        else:
            reasons.append("index_side_not_used")
    return GraspAssessment(appropriate=not reasons, reasons=tuple(reasons))


def _classify_cylindrical(contacts: ContactSummary) -> GraspAssessment:
    """Cups: cylindrical grasp wrapping the cup (thumb + several fingers, or palm + fingers)."""
    fingers = [f for f in _NON_THUMB if f in _fingers_involved(contacts)]
    thumb = "thumb" in _fingers_involved(contacts)
    wrapped = (thumb and len(fingers) >= 3) or (contacts.palm and len(fingers) >= 2)
    reasons = () if wrapped else ("cylindrical_grasp_not_formed",)
    return GraspAssessment(appropriate=wrapped, reasons=reasons)


def _classify_pincer(contacts: ContactSummary) -> GraspAssessment:
    """Washer: pincer or three-jaw-chuck grasp with thumb/index(/middle) pads only."""
    reasons = []
    if "thumb" not in contacts.pads:
        reasons.append("thumb_not_opposed")
    if not ("index" in contacts.pads or "middle" in contacts.pads):
        reasons.append("no_finger_pad_opposition")
    if contacts.palm or "ring" in contacts.pads or "pinky" in contacts.pads:
        reasons.append("extra_fingers_or_palm_used")
    return GraspAssessment(appropriate=not reasons, reasons=tuple(reasons))


def _classify_pinch(contacts: ContactSummary, finger: str) -> GraspAssessment:
    """Pinch items: opposition of the pads of the named finger and thumb, and only them.

    Wrong finger opposition (a different pair, or a palm hold) scores 0 for the item;
    correct opposition without using the pads caps the item at 2.
    """
    reasons = []
    other_pads = contacts.pads - {"thumb", finger}
    palm_hold = contacts.palm and len(_fingers_involved(contacts)) >= 3

    if finger in contacts.pads and "thumb" in contacts.pads:
        if other_pads:
            reasons.append("wrong_opposition")
        return GraspAssessment(appropriate=not reasons, reasons=tuple(reasons))

    if other_pads or palm_hold:
        reasons.append("wrong_opposition")
    if "thumb" not in contacts.pads:
        reasons.append("thumb_not_opposed")
    if finger not in contacts.pads:
        if finger in contacts.dorsals and "thumb" in contacts.pads and not other_pads:
            reasons.append("pads_not_used")
        else:
            reasons.append("named_finger_not_opposed")
    return GraspAssessment(appropriate=False, reasons=tuple(reasons))

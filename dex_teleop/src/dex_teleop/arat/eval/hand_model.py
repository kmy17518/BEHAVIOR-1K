"""Semantic map of the dexterous hand's links for contact-based hand-movement evaluation.

The Sharpa hand already segments the clinically relevant "parts" as links: each finger's
palmar pad is its ``*_elastomer`` link (the ``*_fingertip`` links carry no collision
geometry), the phalanx bodies (``*_PP/MP/DP`` and the proximal ``*_MC``/``*_VL``
structure) form the dorsum/sides, and ``*_hand_C_MC`` is the palm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class HandSemantics:
    """Maps robot link names to (finger, part) semantics."""

    side: str
    palm_link: str
    pad_links: Mapping[str, str] = field(repr=False)
    dorsal_links: Mapping[str, tuple[str, ...]] = field(repr=False)

    @classmethod
    def sharpa(cls, side: str = "right") -> "HandSemantics":
        pad_links = {finger: f"{side}_{finger}_elastomer" for finger in FINGERS}
        dorsal_links = {}
        for finger in FINGERS:
            if finger == "thumb":
                names = ("PP", "DP", "MC", "CMC_VL", "MCP_VL")
            else:
                names = ("PP", "MP", "DP", "MC", "MCP_VL")
            dorsal_links[finger] = tuple(f"{side}_{finger}_{name}" for name in names)
        return cls(
            side=side,
            palm_link=f"{side}_hand_C_MC",
            pad_links=pad_links,
            dorsal_links=dorsal_links,
        )

    def classify_link(self, link_name: str) -> tuple[str, str] | None:
        """Classify a robot link name as (finger-or-palm, part).

        Returns:
            (finger, "pad") for a palmar pad link, (finger, "dorsal") for any other link
            of that finger, ("palm", "palm") for the palm, or None for non-hand links
            (arm links etc.).
        """
        if link_name == self.palm_link:
            return ("palm", "palm")
        for finger in FINGERS:
            if link_name == self.pad_links[finger]:
                return (finger, "pad")
        for finger in FINGERS:
            # Any other link belonging to this finger counts as its dorsum/side,
            # including links not declared in the robot YAML (e.g. fingertips).
            if link_name.startswith(f"{self.side}_{finger}_"):
                return (finger, "dorsal")
        return None

    def all_pad_links(self) -> tuple[str, ...]:
        return tuple(self.pad_links[finger] for finger in FINGERS)

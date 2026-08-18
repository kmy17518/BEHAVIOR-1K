"""Named body-region volumes for the gross-movement targets.

The mannequin asset is a single rigid link with no sub-link annotations, so the target
regions (behind the head, top of the head, the mouth) are defined relative to the
object's live axis-aligned bounding box: vertical bands as fractions of the AABB height,
horizontal constraints in meters around the AABB center, and an optional front/back
halfspace along the facing direction (the ARAT mannequin faces world -X, toward the
robot).

The numbers below are initial estimates from standard human proportions on the 1.32 m
scaled mannequin; verify them visually in the scene and refine as needed. Exclusion
areas from the scoring guide (neck, forehead, chin) are handled by the band edges:
contacts outside every region simply do not count as reaching the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AabbRegion:
    """A region attached to an object's live AABB.

    Attributes:
        z_lo_frac / z_hi_frac: vertical band, as fractions of the AABB height measured
            from its bottom.
        horizontal_radius: maximum horizontal distance (meters) from the AABB XY center,
            or None for no constraint.
        half: "front", "back", or None — restricts the region to the halfspace along the
            facing direction.
        y_halfwidth: maximum lateral offset (meters) from the AABB Y center measured
            perpendicular to the facing direction, or None for no constraint.
    """

    z_lo_frac: float
    z_hi_frac: float
    horizontal_radius: float | None = None
    half: str | None = None
    y_halfwidth: float | None = None

    def contains(
        self,
        point: Sequence[float],
        aabb_lo: Sequence[float],
        aabb_hi: Sequence[float],
        front_dir_xy: Sequence[float] = (-1.0, 0.0),
    ) -> bool:
        height = aabb_hi[2] - aabb_lo[2]
        if height <= 0:
            return False
        z_frac = (point[2] - aabb_lo[2]) / height
        if not (self.z_lo_frac <= z_frac <= self.z_hi_frac):
            return False

        center_x = (aabb_lo[0] + aabb_hi[0]) / 2.0
        center_y = (aabb_lo[1] + aabb_hi[1]) / 2.0
        dx = point[0] - center_x
        dy = point[1] - center_y

        if self.horizontal_radius is not None and math.hypot(dx, dy) > self.horizontal_radius:
            return False

        if self.half is not None:
            along_front = dx * front_dir_xy[0] + dy * front_dir_xy[1]
            if self.half == "front" and along_front < 0:
                return False
            if self.half == "back" and along_front > 0:
                return False

        if self.y_halfwidth is not None:
            # Lateral direction is perpendicular to the facing direction
            lateral = abs(-dx * front_dir_xy[1] + dy * front_dir_xy[0])
            if lateral > self.y_halfwidth:
                return False

        return True

    def center(
        self,
        aabb_lo: Sequence[float],
        aabb_hi: Sequence[float],
        front_dir_xy: Sequence[float] = (-1.0, 0.0),
    ) -> tuple[float, float, float]:
        """A representative target point for approach-progress measurement."""
        height = aabb_hi[2] - aabb_lo[2]
        z = aabb_lo[2] + height * (self.z_lo_frac + self.z_hi_frac) / 2.0
        x = (aabb_lo[0] + aabb_hi[0]) / 2.0
        y = (aabb_lo[1] + aabb_hi[1]) / 2.0
        if self.half is not None:
            # Offset toward the requested halfspace by an approximate head radius
            sign = 1.0 if self.half == "front" else -1.0
            x += sign * front_dir_xy[0] * 0.08
            y += sign * front_dir_xy[1] * 0.08
        return (x, y, z)


# Regions for the ARAT gross-movement mannequin (nphsfp, ~1.32 m tall, head at the top).
# Scoring guide: behind the head (not the neck), top of the head (not the forehead),
# the mouth (not the chin).
MANNEQUIN_REGIONS = {
    # Top band of the head; forehead contacts fall below the band's lower edge.
    "head_top": AabbRegion(z_lo_frac=0.955, z_hi_frac=1.02, horizontal_radius=0.14),
    # Back half of the head; the neck falls below the band's lower edge.
    "head_back": AabbRegion(z_lo_frac=0.86, z_hi_frac=0.96, horizontal_radius=0.16, half="back"),
    # Front of the face around mouth height; the chin falls below the band's lower edge.
    "mouth": AabbRegion(z_lo_frac=0.885, z_hi_frac=0.93, horizontal_radius=0.14, half="front", y_halfwidth=0.07),
}

"""Named Franka + Sharpa reset configurations for ARAT teleoperation."""

from __future__ import annotations

import argparse
from typing import Mapping, Sequence


SHARPA_FINGER_DOFS = 22

# OmniGibson ``franka.yaml`` ``sharpa_right`` default_joint_pos: compact ready pose.
_COMPACT_ARM = (0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75)

# Deoxys "golden resetting joints" from UT-Austin-RPL/deoxys_control. A more
# outstretched tabletop start than the OmniGibson compact ready pose.
_EXTENDED_ARM = (
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
)


def _with_open_hand(arm: Sequence[float]) -> tuple[float, ...]:
    return tuple(arm) + (0.0,) * SHARPA_FINGER_DOFS


RESET_POSES: Mapping[str, tuple[float, ...]] = {
    "compact": _with_open_hand(_COMPACT_ARM),
    "extended": _with_open_hand(_EXTENDED_ARM),
}

DEFAULT_RESET_POSE = "extended"


def reset_joint_positions(name: str = DEFAULT_RESET_POSE) -> list[float]:
    """Return a copy of the named Franka + Sharpa reset configuration."""

    try:
        return list(RESET_POSES[name])
    except KeyError as error:
        known = ", ".join(sorted(RESET_POSES))
        raise ValueError(f"Unknown reset pose {name!r}; expected one of: {known}") from error


def add_reset_pose_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--reset-pose`` with the compact/extended aliases."""

    parser.add_argument(
        "--reset-pose",
        choices=tuple(RESET_POSES),
        default=DEFAULT_RESET_POSE,
        help=(
            "Franka + Sharpa start configuration: 'compact' is the OmniGibson asset "
            "ready pose; 'extended' is the Deoxys tabletop golden pose "
            f"(default: {DEFAULT_RESET_POSE})"
        ),
    )
    return parser

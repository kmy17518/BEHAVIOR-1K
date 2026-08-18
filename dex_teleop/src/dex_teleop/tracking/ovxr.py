"""OVXR tracking-source boundary.

The adapter is deliberately unavailable until OmniGibson's OVXR hand tracking
is restored and verified. Selection is explicit and never falls back to HTS.
"""

from dex_teleop.tracking.base import SourceUnavailableError


def create_ovxr_source():
    raise SourceUnavailableError(
        "OVXR was explicitly selected, but dex_teleop's OVXR source adapter is not implemented yet"
    )

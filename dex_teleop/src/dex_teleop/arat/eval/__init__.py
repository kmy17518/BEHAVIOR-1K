"""ARAT 0-3 scoring layer.

The scorer consumes per-step :class:`AratStepSnapshot` values, so the same scoring code
runs online (fed by :mod:`dex_teleop.arat.eval.live`, which reads the OmniGibson sim) and
in tests (fed synthetic snapshots). Everything in this package except ``live`` is
importable without OmniGibson.

Scoring follows ``ARAT_scoring_guide.md``. Only hand-movement components gate scores;
arm-movement quality is deliberately not evaluated.
"""

from dex_teleop.arat.eval.hand_model import FINGERS, HandSemantics
from dex_teleop.arat.eval.regions import AabbRegion, MANNEQUIN_REGIONS
from dex_teleop.arat.eval.rubrics import ItemRubric, ScoringConfig, load_rubrics
from dex_teleop.arat.eval.scorer import AratItemScorer, ItemResult
from dex_teleop.arat.eval.session import SUBSCALE_MAX_SCORES, AratSessionScorer
from dex_teleop.arat.eval.snapshot import (
    AratStepSnapshot,
    HandContact,
    SupportContact,
    TargetState,
    TrackedObjectState,
    WaterState,
)

__all__ = [
    "AabbRegion",
    "AratItemScorer",
    "AratSessionScorer",
    "AratStepSnapshot",
    "FINGERS",
    "HandContact",
    "HandSemantics",
    "ItemResult",
    "ItemRubric",
    "MANNEQUIN_REGIONS",
    "SUBSCALE_MAX_SCORES",
    "ScoringConfig",
    "SupportContact",
    "TargetState",
    "TrackedObjectState",
    "WaterState",
    "load_rubrics",
]

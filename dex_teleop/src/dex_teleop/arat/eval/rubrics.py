"""Per-item scoring rubrics and shared scoring thresholds.

``rubrics.yaml`` (next to this module) declares, for each of the 19 ARAT activities:
which BDDL instance is the tracked object, what the placement target is, which grasp
class constitutes appropriate hand movement, and the object size used by the
voluntary-opening check. ``ScoringConfig`` holds the shared thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml


RUBRICS_PATH = Path(__file__).with_name("rubrics.yaml")

VALID_GRASP_CLASSES = (
    "opposition",
    "pads_opposition",
    "spherical",
    "lateral",
    "cylindrical",
    "pincer",
    "pinch",
    "palmar",
)
VALID_TARGET_KINDS = ("shelf", "shelf_or_tin", "tin", "pegged", "pour", "region")


@dataclass(frozen=True)
class ScoringConfig:
    """Shared thresholds. Times are seconds of engaged task time; lengths are meters."""

    time_limit_s: float = 60.0
    score3_time_s: float = 5.0
    lift_height_m: float = 0.02
    hold_min_s: float = 0.2
    release_confirm_s: float = 0.5
    settle_speed: float = 0.10
    opening_factor: float = 0.9
    grasp_ok_fraction: float = 0.6
    region_contact_confirm_s: float = 0.1
    gross_start_progress_m: float = 0.08
    # Pouring
    spill_tolerance_particles: int = 0
    source_empty_tolerance_particles: int = 2
    min_transfer_fraction: float = 0.5
    control_dt: float = 1.0 / 30.0


@dataclass(frozen=True)
class ItemRubric:
    activity: str
    subscale: str
    tracked_instance: str | None
    target: Mapping
    grasp: Mapping
    object_size_m: float | None = None
    label: str = ""
    extras: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.target["kind"] not in VALID_TARGET_KINDS:
            raise ValueError(f"{self.activity}: unknown target kind {self.target['kind']!r}")
        if self.grasp["class"] not in VALID_GRASP_CLASSES:
            raise ValueError(f"{self.activity}: unknown grasp class {self.grasp['class']!r}")
        if self.grasp["class"] == "pinch" and "finger" not in self.grasp:
            raise ValueError(f"{self.activity}: pinch grasp requires a 'finger'")
        if self.target["kind"] == "region" and "region" not in self.target:
            raise ValueError(f"{self.activity}: region target requires a 'region'")
        if self.target["kind"] != "region" and self.tracked_instance is None:
            raise ValueError(f"{self.activity}: non-region targets require a tracked_instance")

    @property
    def relevant_opening_fingers(self) -> tuple[str, ...]:
        if self.grasp["class"] == "pinch":
            return (self.grasp["finger"],)
        return ("index", "middle", "ring", "pinky")


def load_rubrics(path: Path = RUBRICS_PATH) -> dict[str, ItemRubric]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    rubrics = {}
    for activity, data in raw.items():
        rubrics[activity] = ItemRubric(
            activity=activity,
            subscale=str(data["subscale"]),
            tracked_instance=data.get("tracked_instance"),
            target=dict(data["target"]),
            grasp=dict(data["grasp"]),
            object_size_m=data.get("object_size_m"),
            label=str(data.get("label", activity)),
            extras={k: v for k, v in data.items() if k not in
                    {"subscale", "tracked_instance", "target", "grasp", "object_size_m", "label"}},
        )
    return rubrics

"""Calibration from MANUS Core world to the dex_teleop reference frame."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from dex_teleop.tracking.transforms import RigidTransform, compose_transforms
from dex_teleop.types import Handedness


MANUS_CORE_CALIBRATION_SCHEMA_VERSION = 1
MANUS_CORE_WORLD_FRAME = "manus_core_world_y_up_rh_z_to_viewer_m"
MANUS_BAKED_WRIST_SEMANTICS = "core_baked_skeleton_wrist"


@dataclass(frozen=True)
class ManusWristCalibration:
    """Per-hand mount semantics and residual correction for a baked Core wrist."""

    handedness: Handedness
    skeleton_wrist_to_anatomical_wrist: RigidTransform
    tracker_id: str | None = None
    tracker_offset_preset: str | None = None
    tracker_to_anatomical_wrist: RigidTransform | None = None
    core_tracker_offset_applied: bool = True
    source_pose: str = MANUS_BAKED_WRIST_SEMANTICS

    def __post_init__(self) -> None:
        object.__setattr__(self, "handedness", Handedness(self.handedness))
        if not isinstance(self.skeleton_wrist_to_anatomical_wrist, RigidTransform) or (
            self.tracker_to_anatomical_wrist is not None
            and not isinstance(self.tracker_to_anatomical_wrist, RigidTransform)
        ):
            raise TypeError("MANUS wrist transforms must be RigidTransform values")
        if self.source_pose != MANUS_BAKED_WRIST_SEMANTICS:
            raise ValueError(
                f"MANUS source_pose must be {MANUS_BAKED_WRIST_SEMANTICS!r}"
            )
        if not isinstance(self.core_tracker_offset_applied, bool):
            raise ValueError("core_tracker_offset_applied must be boolean")
        if not self.core_tracker_offset_applied:
            raise ValueError(
                "Raw-skeleton wrist control requires Core's tracker-to-wrist offset "
                "to be applied; configure the Ultimate preset in MANUS Core"
            )
        for field, value in (
            ("tracker_id", self.tracker_id),
            ("tracker_offset_preset", self.tracker_offset_preset),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"MANUS {field} must be a non-empty string or null")

    @classmethod
    def from_mapping(cls, value: Mapping, *, field: str) -> "ManusWristCalibration":
        try:
            policy_fields = {
                "tracker_type",
                "tracker_system",
                "tracker_user_id",
            }.intersection(value)
            if policy_fields:
                raise ValueError(
                    "tracker identity policy does not belong in geometric calibration; "
                    "remove " + ", ".join(sorted(policy_fields))
                )
            offset_applied = value["core_tracker_offset_applied"]
            if not isinstance(offset_applied, bool):
                raise ValueError("core_tracker_offset_applied must be boolean")
            tracker_transform = value.get("tracker_to_anatomical_wrist")
            return cls(
                handedness=Handedness(value["handedness"]),
                source_pose=value.get("source_pose", MANUS_BAKED_WRIST_SEMANTICS),
                core_tracker_offset_applied=offset_applied,
                tracker_id=value.get("tracker_id"),
                tracker_offset_preset=value.get("tracker_offset_preset"),
                tracker_to_anatomical_wrist=(
                    None
                    if tracker_transform is None
                    else RigidTransform.from_mapping(
                        tracker_transform,
                        field=f"{field}.tracker_to_anatomical_wrist",
                    )
                ),
                skeleton_wrist_to_anatomical_wrist=RigidTransform.from_mapping(
                    value["skeleton_wrist_to_anatomical_wrist"],
                    field=f"{field}.skeleton_wrist_to_anatomical_wrist",
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid MANUS wrist calibration {field}: {error}"
            ) from error

    def as_mapping(self) -> dict:
        value = {
            "handedness": self.handedness.value,
            "source_pose": self.source_pose,
            "core_tracker_offset_applied": self.core_tracker_offset_applied,
            "tracker_id": self.tracker_id,
            "tracker_offset_preset": self.tracker_offset_preset,
            "tracker_to_anatomical_wrist": (
                None
                if self.tracker_to_anatomical_wrist is None
                else self.tracker_to_anatomical_wrist.as_mapping()
            ),
            "skeleton_wrist_to_anatomical_wrist": (
                self.skeleton_wrist_to_anatomical_wrist.as_mapping()
            ),
        }
        return value


@dataclass(frozen=True)
class ManusCoreCalibration:
    """Core-world extrinsic plus explicit per-hand anatomical-wrist semantics."""

    reference_frame: str
    reference_from_core_world: RigidTransform
    wrists: tuple[ManusWristCalibration, ...]
    core_world_frame: str = MANUS_CORE_WORLD_FRAME

    def __post_init__(self) -> None:
        if self.core_world_frame != MANUS_CORE_WORLD_FRAME:
            raise ValueError(
                f"MANUS core_world_frame must be {MANUS_CORE_WORLD_FRAME!r}"
            )
        if (
            not isinstance(self.reference_frame, str)
            or not self.reference_frame.strip()
        ):
            raise ValueError("MANUS calibration reference_frame must be non-empty")
        if not isinstance(self.reference_from_core_world, RigidTransform):
            raise TypeError("MANUS reference_from_core_world must be a RigidTransform")
        object.__setattr__(self, "wrists", tuple(self.wrists))
        if not self.wrists:
            raise ValueError("MANUS calibration must contain at least one wrist")
        if not all(isinstance(wrist, ManusWristCalibration) for wrist in self.wrists):
            raise TypeError(
                "MANUS calibration wrists must contain ManusWristCalibration values"
            )
        sides = [wrist.handedness for wrist in self.wrists]
        if len(set(sides)) != len(sides):
            raise ValueError("MANUS calibration contains duplicate handedness entries")

    def wrist(self, handedness: Handedness | str) -> ManusWristCalibration:
        side = Handedness(handedness)
        for wrist in self.wrists:
            if wrist.handedness == side:
                return wrist
        raise KeyError(f"MANUS calibration has no {side.value}-hand wrist entry")

    def apply(
        self,
        handedness: Handedness | str,
        core_world_from_skeleton_wrist: RigidTransform,
    ) -> RigidTransform:
        """Map a baked raw-skeleton wrist into the calibrated reference."""

        wrist = self.wrist(handedness)
        return compose_transforms(
            compose_transforms(
                self.reference_from_core_world,
                core_world_from_skeleton_wrist,
            ),
            wrist.skeleton_wrist_to_anatomical_wrist,
        )

    @classmethod
    def from_mapping(cls, value: Mapping) -> "ManusCoreCalibration":
        if value.get("schema_version") != MANUS_CORE_CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MANUS Core calibration schema {value.get('schema_version')!r}; "
                f"expected {MANUS_CORE_CALIBRATION_SCHEMA_VERSION}"
            )
        raw_wrists = value.get("wrists")
        if not isinstance(raw_wrists, list):
            raise ValueError("MANUS Core calibration wrists must be a list")
        try:
            return cls(
                core_world_frame=value["core_world_frame"],
                reference_frame=value["reference_frame"],
                reference_from_core_world=RigidTransform.from_mapping(
                    value["reference_from_core_world"],
                    field="reference_from_core_world",
                ),
                wrists=tuple(
                    ManusWristCalibration.from_mapping(item, field=f"wrists[{index}]")
                    for index, item in enumerate(raw_wrists)
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid MANUS Core calibration: {error}") from error

    @classmethod
    def load(cls, path: str | Path) -> "ManusCoreCalibration":
        with Path(path).expanduser().open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, Mapping):
            raise ValueError("MANUS Core calibration must be a JSON object")
        return cls.from_mapping(document)

    def as_mapping(self) -> dict:
        return {
            "schema_version": MANUS_CORE_CALIBRATION_SCHEMA_VERSION,
            "core_world_frame": self.core_world_frame,
            "reference_frame": self.reference_frame,
            "reference_from_core_world": self.reference_from_core_world.as_mapping(),
            "wrists": [wrist.as_mapping() for wrist in self.wrists],
        }


__all__ = [
    "MANUS_BAKED_WRIST_SEMANTICS",
    "MANUS_CORE_CALIBRATION_SCHEMA_VERSION",
    "MANUS_CORE_WORLD_FRAME",
    "ManusCoreCalibration",
    "ManusWristCalibration",
]

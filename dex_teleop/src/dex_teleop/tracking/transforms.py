"""Source-neutral rigid-transform primitives for tracking calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RigidTransform:
    """A child's pose expressed in its parent (``parent_T_child``), using xyzw."""

    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3).copy()
        quaternion = (
            np.asarray(self.quaternion_xyzw, dtype=np.float64).reshape(4).copy()
        )
        if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
            raise ValueError("RigidTransform contains a non-finite value")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0.0:
            raise ValueError("RigidTransform quaternion must have non-zero norm")
        quaternion /= norm
        translation.setflags(write=False)
        quaternion.setflags(write=False)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    @classmethod
    def from_mapping(cls, value: Mapping, *, field: str) -> "RigidTransform":
        try:
            return cls(value["translation"], value["quaternion_xyzw"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{field} must contain translation[3] and quaternion_xyzw[4]"
            ) from error

    def as_mapping(self) -> dict[str, list[float]]:
        return {
            "translation": self.translation.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw.tolist(),
        }


def compose_transforms(
    parent_from_middle: RigidTransform, middle_from_child: RigidTransform
) -> RigidTransform:
    """Compose ``parent_T_middle * middle_T_child``."""

    parent_rotation = Rotation.from_quat(parent_from_middle.quaternion_xyzw)
    child_rotation = Rotation.from_quat(middle_from_child.quaternion_xyzw)
    return RigidTransform(
        parent_from_middle.translation
        + parent_rotation.apply(middle_from_child.translation),
        (parent_rotation * child_rotation).as_quat(),
    )


def inverse_transform(transform: RigidTransform) -> RigidTransform:
    """Return ``child_T_parent`` for ``parent_T_child``."""

    rotation = Rotation.from_quat(transform.quaternion_xyzw)
    inverse = rotation.inv()
    return RigidTransform(
        inverse.apply(-transform.translation),
        inverse.as_quat(),
    )


__all__ = ["RigidTransform", "compose_transforms", "inverse_transform"]

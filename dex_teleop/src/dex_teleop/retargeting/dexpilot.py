"""Optional DexPilot backend powered by the public ``dex-retargeting`` package.

The dependency is imported only when this backend is selected.  Version 0.4.6
is used because it supports the NumPy 1.x ABI required by Isaac Sim 5.1.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dex_teleop.hands import get_hand_profile
from dex_teleop.retargeting.base import RetargeterUnavailableError
from dex_teleop.tracking.openxr import articulation_to_mediapipe21
from dex_teleop.types import (
    FusedHandObservation,
    HandFrame,
    Handedness,
    MEDIAPIPE_JOINT_NAMES,
    OPENXR_ANATOMICAL_WRIST_FRAME,
    RetargetedHandCommand,
)


_RESOURCE_ROOT = Path(__file__).resolve().parent
_DEX_RETARGETING_VERSION = "0.4.6"
_HANDTRACKING_TO_SHARPA = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


class DexPilotRetargeter:
    """Retarget MediaPipe-21 positions with DexPilot's vector objective."""

    accepted_schema = "mediapipe21"

    def __init__(
        self,
        hand_model: str,
        hand_side: str,
        *,
        config_path: str | Path,
        urdf_path: str | Path,
        handtracking_to_baselink: np.ndarray = _HANDTRACKING_TO_SHARPA,
    ) -> None:
        hand_model = hand_model.lower()
        hand_side = hand_side.lower()
        if hand_model != "sharpa":
            raise ValueError(
                "The packaged DexPilot backend currently supports only the Sharpa hand"
            )
        if hand_side not in {"left", "right"}:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")

        try:
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as error:
            raise RetargeterUnavailableError(
                f"DexPilot requires dex-retargeting=={_DEX_RETARGETING_VERSION} and all of its dependencies. "
                "Install the optional dex-teleop[dexpilot] extra in the behavior_dex environment. "
                f"Import failed: {error}"
            ) from error
        try:
            installed_version = metadata.version("dex-retargeting")
        except metadata.PackageNotFoundError as error:
            raise RetargeterUnavailableError(
                "The dex_retargeting module is importable but has no dex-retargeting distribution metadata; "
                "install the pinned dex-teleop[dexpilot] extra instead of adding a source tree to PYTHONPATH"
            ) from error
        if installed_version != _DEX_RETARGETING_VERSION:
            raise RetargeterUnavailableError(
                f"DexPilot requires dex-retargeting=={_DEX_RETARGETING_VERSION}, found {installed_version}. "
                "Install the pinned dex-teleop[dexpilot] extra; backend versions are never substituted implicitly."
            )

        config_path = Path(config_path).expanduser().resolve()
        urdf_path = Path(urdf_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing DexPilot configuration: {config_path}")
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Missing Sharpa retargeting URDF: {urdf_path}")

        transform = np.asarray(handtracking_to_baselink, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError(
                "handtracking_to_baselink must be a finite 3x3 rotation matrix"
            )
        if not np.allclose(transform.T @ transform, np.eye(3), atol=1e-6):
            raise ValueError("handtracking_to_baselink must be orthonormal")
        if not np.isclose(np.linalg.det(transform), 1.0, atol=1e-6):
            raise ValueError(
                "handtracking_to_baselink must be a proper rotation with determinant +1"
            )

        self.hand_model = hand_model
        self.hand_side = hand_side
        self.hand_profile = get_hand_profile(hand_model)
        self.handtracking_to_baselink = transform.copy()
        config = RetargetingConfig.load_from_file(
            config_path,
            override={"urdf_path": str(urdf_path)},
        )
        self._retargeting = config.build()
        self.joint_names = tuple(self._retargeting.optimizer.robot.dof_joint_names)
        if len(self.joint_names) != self.hand_profile.degrees_of_freedom:
            raise ValueError(
                f"DexPilot Sharpa URDF exposes {len(self.joint_names)} joints; "
                f"expected {self.hand_profile.degrees_of_freedom}"
            )

    @classmethod
    def from_hand_model(
        cls, hand_model: str, hand_side: str = "right"
    ) -> "DexPilotRetargeter":
        hand_model = hand_model.lower()
        hand_side = hand_side.lower()
        return cls(
            hand_model,
            hand_side,
            config_path=_RESOURCE_ROOT
            / "configs"
            / f"{hand_model}_{hand_side}_dexpilot.yaml",
            urdf_path=_RESOURCE_ROOT
            / "urdf"
            / hand_model
            / f"{hand_side}_{hand_model}_wave.urdf",
        )

    def retarget(self, frame: HandFrame) -> RetargetedHandCommand:
        if frame.handedness.value != self.hand_side:
            raise ValueError(
                f"Retargeter is configured for {self.hand_side}, received {frame.handedness.value} frame"
            )

        # HandFrame landmarks are in the tracking world.  Row-vector
        # multiplication by the wrist rotation maps them back into the local
        # wrist frame used by the DexPilot objective.
        positions = frame.mediapipe_landmarks().astype(np.float64, copy=True)
        positions -= positions[0]
        wrist_rotation = Rotation.from_quat(frame.wrist_quaternion_xyzw).as_matrix()
        local_positions = positions @ wrist_rotation
        return self._retarget_local_positions(
            local_positions, frame.timestamp, frame.handedness
        )

    def retarget_observation(
        self, observation: FusedHandObservation
    ) -> RetargetedHandCommand:
        """Consume the rich named representation without a world-frame round trip."""

        if observation.articulation.coordinate_frame != OPENXR_ANATOMICAL_WRIST_FRAME:
            raise ValueError(
                "The packaged Sharpa DexPilot transform expects articulation in "
                f"{OPENXR_ANATOMICAL_WRIST_FRAME!r}, received "
                f"{observation.articulation.coordinate_frame!r}; configure an explicit "
                "ArticulationFrameTransform before retargeting"
            )
        articulation = articulation_to_mediapipe21(observation.articulation)
        invalid = [
            name
            for name in MEDIAPIPE_JOINT_NAMES
            if not articulation.joint_validity[name]
        ]
        if invalid:
            raise ValueError(
                f"DexPilot requires valid MediaPipe-equivalent joints: {invalid}"
            )
        positions = articulation.positions(MEDIAPIPE_JOINT_NAMES).astype(
            np.float64, copy=True
        )
        positions -= positions[0]
        return self._retarget_local_positions(
            positions, observation.timestamp, observation.handedness
        )

    def _retarget_local_positions(
        self,
        local_positions: np.ndarray,
        timestamp: float,
        handedness: Handedness,
    ) -> RetargetedHandCommand:
        if handedness.value != self.hand_side:
            raise ValueError(
                f"Retargeter is configured for {self.hand_side}, received {handedness.value} observation"
            )
        target_positions = local_positions @ self.handtracking_to_baselink

        human_indices = self._retargeting.optimizer.target_link_human_indices
        if human_indices.ndim != 2 or human_indices.shape[0] != 2:
            raise RuntimeError(
                "DexPilot optimizer returned an unexpected target_link_human_indices shape "
                f"{human_indices.shape}"
            )
        reference = (
            target_positions[human_indices[1]] - target_positions[human_indices[0]]
        )
        try:
            import torch
        except ImportError as error:
            raise RetargeterUnavailableError(
                "DexPilot's optimizer requires PyTorch; install the complete dex-teleop[dexpilot] extra"
            ) from error
        # DexPilot differentiates its objective with torch. Make that contract
        # explicit even if a simulator caller has entered inference mode.
        with torch.enable_grad(), torch.inference_mode(False):
            joint_positions = np.asarray(
                self._retargeting.retarget(reference), dtype=np.float64
            )

        command = RetargetedHandCommand(
            timestamp=timestamp,
            handedness=handedness,
            hand_model=self.hand_model,
            joint_names=self.joint_names,
            joint_positions=joint_positions,
        )
        self.hand_profile.validate(command)
        return command

    def reset(self) -> None:
        self._retargeting.reset()
        low_pass_filter = getattr(self._retargeting, "filter", None)
        if low_pass_filter is not None:
            low_pass_filter.reset()

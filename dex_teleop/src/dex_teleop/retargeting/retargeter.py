"""High-level landmark retargeting API.

Adapted from AnyDexRetarget's unified retargeter (MIT License, Copyright (c)
2025 Shiquan Qiu). See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dex_teleop.hands import get_hand_profile
from dex_teleop.retargeting.mediapipe import apply_mediapipe_transformations
from dex_teleop.retargeting.optimizer import BaseOptimizer, LPFilter
from dex_teleop.types import HandFrame, RetargetedHandCommand

SUPPORTED_HAND_MODELS = ("shadow", "sharpa", "wuji")
_RESOURCE_ROOT = Path(__file__).resolve().parent


class LandmarkRetargeter:
    """Retarget canonical 21-landmark frames to one supported robot hand."""

    def __init__(self, hand_model: str, config: dict, hand_side: str = "right") -> None:
        hand_model = hand_model.lower()
        hand_side = hand_side.lower()
        if hand_model not in SUPPORTED_HAND_MODELS:
            raise ValueError(f"Unsupported hand model {hand_model!r}; choose from {SUPPORTED_HAND_MODELS}")
        if hand_side not in {"left", "right"}:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")

        self.hand_model = hand_model
        self.hand_profile = get_hand_profile(hand_model)
        self.hand_side = hand_side
        self.config = deepcopy(config)
        self.config.setdefault("optimizer", {})["hand_side"] = hand_side
        self.config.setdefault("robot", {})["type"] = hand_model
        robot_defaults = BaseOptimizer.ROBOT_CONFIGS[hand_model]
        urdf_name = robot_defaults["urdf_file"][hand_side]
        urdf_path = _RESOURCE_ROOT / robot_defaults["urdf_subdir"] / urdf_name
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Missing {hand_model} retargeting URDF: {urdf_path}")
        self.config["robot"]["urdf_path"] = str(urdf_path)

        self.optimizer = BaseOptimizer.from_config(self.config)
        alpha = float(self.config.get("retarget", {}).get("lp_alpha", 0.2))
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"retarget.lp_alpha must be in (0, 1], got {alpha}")
        self.filter = LPFilter(alpha)
        self.rotation_xyz = self.config.get("retarget", {}).get("mediapipe_rotation", {})
        self.joint_names = tuple(self.optimizer.robot.dof_joint_names)
        if len(self.joint_names) != self.hand_profile.degrees_of_freedom:
            raise ValueError(
                f"{hand_model} URDF exposes {len(self.joint_names)} joints; "
                f"expected {self.hand_profile.degrees_of_freedom}"
            )

    @classmethod
    def from_hand_model(cls, hand_model: str, hand_side: str = "right") -> "LandmarkRetargeter":
        config_path = _RESOURCE_ROOT / "configs" / f"{hand_model.lower()}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing retargeting configuration: {config_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        return cls(hand_model=hand_model, config=config, hand_side=hand_side)

    def retarget(self, frame: HandFrame, apply_filter: bool = True) -> RetargetedHandCommand:
        if frame.handedness.value != self.hand_side:
            raise ValueError(
                f"Retargeter is configured for {self.hand_side}, received {frame.handedness.value} frame"
            )
        positions = self.retarget_landmarks(frame.mediapipe_landmarks(), apply_filter=apply_filter)
        command = RetargetedHandCommand(
            timestamp=frame.timestamp,
            handedness=frame.handedness,
            hand_model=self.hand_model,
            joint_names=self.joint_names,
            joint_positions=positions,
        )
        self.hand_profile.validate(command)
        return command

    def retarget_landmarks(self, landmarks: np.ndarray, apply_filter: bool = True) -> np.ndarray:
        landmarks = np.asarray(landmarks, dtype=np.float64)
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks with shape (21, 3), got {landmarks.shape}")
        transformed = apply_mediapipe_transformations(landmarks, self.hand_side)
        if self.rotation_xyz:
            angles = [float(self.rotation_xyz.get(axis, 0.0)) for axis in "xyz"]
            transformed = transformed @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix().T
        positions = self.optimizer.solve(transformed)
        return self.filter.next(positions) if apply_filter else positions

    def reset(self) -> None:
        self.filter.reset()
        self.optimizer.last_qpos = None

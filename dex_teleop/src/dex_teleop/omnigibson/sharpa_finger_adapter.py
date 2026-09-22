"""Finger-only Sharpa execution for a hand whose arm the simulator holds still."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import numpy as np
import torch as th

from dex_teleop.hands import SHARPA_ACTION_JOINTS, get_hand_profile
from dex_teleop.runtime import RetargetingSnapshot, SafetyConfig, SafetyFilter


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharpaFingerAdapterConfig:
    control_hz: float = 30.0
    finger_target_scale: float = 1.0


class SharpaFingerActionAdapter:
    """Map named Sharpa joint targets to a 22-D finger action.

    The robot must expose its arm through a zero-width controller (OmniGibson's
    ``NullJointController``) so that ``robot.action_dim`` is exactly the 22
    Sharpa finger joints in :data:`SHARPA_ACTION_JOINTS` order.  The wrist pose
    carried by the snapshot is ignored by design.
    """

    def __init__(
        self,
        robot,
        config: SharpaFingerAdapterConfig = SharpaFingerAdapterConfig(),
        safety_config: SafetyConfig = SafetyConfig(),
    ) -> None:
        get_hand_profile("sharpa")
        if not math.isfinite(config.control_hz) or config.control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")
        if not 0.0 < config.finger_target_scale <= 1.0:
            raise ValueError("finger_target_scale must be in (0, 1]")
        self.robot = robot
        self.config = config
        self.arm_name = robot.arm_names[0]
        self.action_dimension = int(robot.action_dim)
        if self.action_dimension != len(SHARPA_ACTION_JOINTS):
            raise ValueError(
                "Finger-only Sharpa execution requires the arm to be held by a NullJointController "
                f"so the action is {len(SHARPA_ACTION_JOINTS)}-D; got {self.action_dimension}-D"
            )
        controller_indices = getattr(robot, "controller_joint_idx", None)
        if not isinstance(controller_indices, dict):
            raise RuntimeError("Robot does not expose controller_joint_idx; cannot prove Sharpa command order")
        indices = controller_indices.get(f"gripper_{self.arm_name}")
        if indices is None:
            raise RuntimeError(f"Robot has no gripper controller for arm {self.arm_name}")
        raw_indices = indices.tolist() if hasattr(indices, "tolist") else indices
        self.finger_dof_indices = tuple(int(index) for index in raw_indices)
        actual_names = tuple(robot.dof_names_ordered[index] for index in self.finger_dof_indices)
        if actual_names != SHARPA_ACTION_JOINTS:
            raise ValueError(
                "Sharpa controller joint order differs from the declared execution contract:\n"
                f"expected={SHARPA_ACTION_JOINTS}\nactual={actual_names}"
            )
        self.safety = SafetyFilter(0, len(SHARPA_ACTION_JOINTS), 1.0 / config.control_hz, safety_config)
        self._last_live_fingers: dict[str, float] | None = None
        self.reset()

    def reset(self) -> None:
        """Re-seed the rate limiter from the measured finger joints."""

        current = self.robot.get_joint_positions()[list(self.finger_dof_indices)]
        self.safety.reset(np.asarray(current.cpu().numpy(), dtype=np.float64))
        self._last_live_fingers = None

    def action(self, snapshot: RetargetingSnapshot) -> th.Tensor:
        """Build one finger action from the retargeted command; the wrist is ignored."""

        command = snapshot.command
        if command.hand_model != "sharpa" or command.handedness.value != "right":
            raise ValueError("Finger-only execution supports only right-hand Sharpa commands")
        positions_by_name = dict(zip(command.joint_names, command.joint_positions, strict=True))
        if set(positions_by_name) != set(SHARPA_ACTION_JOINTS):
            missing = set(SHARPA_ACTION_JOINTS).difference(positions_by_name)
            extra = set(positions_by_name).difference(SHARPA_ACTION_JOINTS)
            raise ValueError(f"Sharpa command joint mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        targets = np.array(
            [positions_by_name[name] for name in SHARPA_ACTION_JOINTS], dtype=np.float64
        ) * self.config.finger_target_scale
        self._last_live_fingers = dict(zip(SHARPA_ACTION_JOINTS, targets.tolist()))
        filtered = self.safety.apply(targets, logger=LOGGER)
        return th.as_tensor(filtered, dtype=th.float32)

    @property
    def last_live_fingers(self) -> dict[str, float] | None:
        """The most recent unfiltered finger targets by joint name, set by :meth:`action`."""

        return None if self._last_live_fingers is None else dict(self._last_live_fingers)

    @property
    def measured_fingers(self) -> dict[str, float]:
        """The robot's current measured finger joint positions by joint name."""

        positions = self.robot.get_joint_positions()[list(self.finger_dof_indices)]
        return dict(zip(SHARPA_ACTION_JOINTS, positions.tolist()))

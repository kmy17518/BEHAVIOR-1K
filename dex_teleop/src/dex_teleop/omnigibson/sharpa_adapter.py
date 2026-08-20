"""Canonical hand command to Franka-mounted Sharpa OmniGibson action."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Mapping

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T

from dex_teleop.hands import SHARPA_ACTION_JOINTS, get_hand_profile
from dex_teleop.runtime import RetargetingSnapshot, SafetyConfig, SafetyFilter


LOGGER = logging.getLogger(__name__)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion, dtype=np.float64)[[3, 0, 1, 2]]


def _wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray(quaternion, dtype=np.float64)[[1, 2, 3, 0]]


def _inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    xyz = np.array([x, y, z], dtype=np.float64)
    cross = 2.0 * np.cross(xyz, vector)
    return vector + w * cross + np.cross(xyz, cross)


def _geodesic_angle_xyzw(first: np.ndarray, second: np.ndarray) -> float:
    """Rotation angle between two unit quaternions; sign flips (q vs -q) measure zero."""
    dot = abs(float(np.dot(first, second)))
    return 2.0 * math.acos(min(1.0, dot))


@dataclass(frozen=True)
class SharpaAdapterConfig:
    control_hz: float = 60.0
    position_sensitivity: float = 1.0
    tracking_to_world_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    shoulder_position: tuple[float, float, float] = (0.0, 0.0, 1.194)
    shoulder_reach: float = 0.80
    minimum_z: float = 0.82
    maximum_z: float = 1.75
    maximum_position_step: float = 0.02
    finger_target_scale: float = 1.0
    # Wrist-orientation glitch gate: a tracked-wrist rotation larger than this between
    # consecutive frames (45 degrees; hand trackers flip ~180 degrees on closed-fist /
    # edge-on poses) is impossible for a human and is rejected -- the last accepted
    # orientation is held instead. Tracking resumes as soon as a frame lands back within
    # the gate of the held orientation.
    maximum_orientation_step: float = math.radians(45.0)
    # For stale frames the allowance grows at this rate (fast human wrist bursts across
    # dropped frames), evaluated over at most 0.15 s of frame gap...
    maximum_orientation_rate: float = math.radians(600.0)
    # ...but never beyond this hard cap, so a flip cannot slip through as two stale jumps.
    maximum_orientation_allowance: float = math.radians(60.0)
    # Trackers can also slew to a flipped estimate in sub-threshold steps. Sustained
    # rotation faster than this, measured against the accepted orientation from up to
    # orientation_slew_horizon_s ago (floored by maximum_orientation_allowance so real
    # short bursts pass), is rejected the same way as a single-frame jump.
    maximum_orientation_slew_rate: float = math.radians(150.0)
    orientation_slew_horizon_s: float = 0.5
    # If every frame keeps being rejected for this long, accept the new orientation --
    # the estimate is persistent, and holding forever would lock the wrist. Raise this if
    # tracking flips typically persist longer; SPACE re-anchoring always resets the gate.
    orientation_resync_s: float = 2.0


@dataclass(frozen=True)
class WristActionDiagnostics:
    """One wrist sample at the important stages of the OmniGibson action path."""

    frame_timestamp: float
    input_quaternion_xyzw: np.ndarray
    gated_quaternion_xyzw: np.ndarray
    target_quaternion_xyzw: np.ndarray
    target_position: np.ndarray
    target_axis_angle: np.ndarray
    filtered_position: np.ndarray
    filtered_axis_angle: np.ndarray
    eef_position_before: np.ndarray
    eef_quaternion_before_xyzw: np.ndarray
    gate_decision: str
    gate_reason: str | None
    hemisphere_corrected: bool


@dataclass(frozen=True)
class _WristGateResult:
    quaternion_xyzw: np.ndarray
    decision: str
    reason: str | None
    hemisphere_corrected: bool


class SharpaActionAdapter:
    """Map source-independent wrist + named Sharpa joints to one OG action."""

    def __init__(
        self,
        robot,
        config: SharpaAdapterConfig = SharpaAdapterConfig(),
        safety_config: SafetyConfig = SafetyConfig(),
    ) -> None:
        get_hand_profile("sharpa")
        if config.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not math.isfinite(config.position_sensitivity) or config.position_sensitivity <= 0:
            raise ValueError("position_sensitivity must be finite and positive")
        if not 0.0 < config.finger_target_scale <= 1.0:
            raise ValueError("finger_target_scale must be in (0, 1]")
        self.robot = robot
        self.config = config
        self.arm_name = robot.arm_names[0]
        self.action_dimension = int(robot.action_dim)
        if self.action_dimension != 28:
            raise ValueError(f"Sharpa execution requires a 28-D action, got {self.action_dimension}")
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

        self.safety = SafetyFilter(6, 22, 1.0 / config.control_hz, safety_config)
        self._position_offset: np.ndarray | None = None
        self._orientation_offset_wxyz: np.ndarray | None = None
        self._previous_target_position: np.ndarray | None = None
        self._last_live_fingers: dict[str, float] | None = None
        self._accepted_wrist_quaternion: np.ndarray | None = None
        self._accepted_wrist_timestamp: float | None = None
        self._accepted_wrist_history: deque[tuple[float, np.ndarray]] = deque()
        self._rejection_started: float | None = None
        self._last_wrist_diagnostics: WristActionDiagnostics | None = None
        self.request_anchor()

    def request_anchor(self) -> None:
        self._position_offset = None
        self._orientation_offset_wxyz = None
        self._previous_target_position = None
        self._accepted_wrist_quaternion = None
        self._accepted_wrist_timestamp = None
        self._accepted_wrist_history = deque()
        self._rejection_started = None
        self._last_wrist_diagnostics = None
        base_position, base_quaternion = self.robot.get_position_orientation()
        eef_position, eef_quaternion = self.robot.eef_links[self.arm_name].get_position_orientation()
        relative_position, relative_quaternion = T.relative_pose_transform(
            eef_position, eef_quaternion, base_position, base_quaternion
        )
        current_fingers = self.robot.get_joint_positions()[list(self.finger_dof_indices)]
        current = np.concatenate(
            [
                relative_position.cpu().numpy(),
                T.quat2axisangle(relative_quaternion).cpu().numpy(),
                current_fingers.cpu().numpy(),
            ]
        )
        self.safety.reset(current)

    def action(self, snapshot: RetargetingSnapshot, frozen_fingers: Mapping[str, float] | None = None) -> th.Tensor:
        """Build one action; ``frozen_fingers`` overrides those joints' targets by name.

        The live (unfrozen) finger targets remain readable through
        :attr:`last_live_fingers`, and the override happens before the safety filter so
        engaging or releasing a freeze is rate-limited into a smooth ramp.
        """
        command = snapshot.command
        if command.hand_model != "sharpa" or command.handedness.value != "right":
            raise ValueError("Initial OmniGibson execution supports only right-hand Sharpa commands")
        positions_by_name = dict(zip(command.joint_names, command.joint_positions, strict=True))
        if set(positions_by_name) != set(SHARPA_ACTION_JOINTS):
            missing = set(SHARPA_ACTION_JOINTS).difference(positions_by_name)
            extra = set(positions_by_name).difference(SHARPA_ACTION_JOINTS)
            raise ValueError(f"Sharpa command joint mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

        base_position, base_quaternion = self.robot.get_position_orientation()
        eef_position, eef_quaternion = self.robot.eef_links[self.arm_name].get_position_orientation()
        eef_relative_position, eef_relative_quaternion = T.relative_pose_transform(
            eef_position, eef_quaternion, base_position, base_quaternion
        )
        base_position_np = base_position.cpu().numpy()
        base_quaternion_np = base_quaternion.cpu().numpy()

        frame = snapshot.frame
        tracking_to_world = np.asarray(self.config.tracking_to_world_quaternion_xyzw, dtype=np.float64)
        mapped_position_world = (
            _rotate_xyzw(tracking_to_world, frame.wrist_position) * self.config.position_sensitivity
        )
        wrist_position_robot = mapped_position_world - base_position_np
        if self._position_offset is None:
            self._position_offset = eef_relative_position.cpu().numpy() - wrist_position_robot
        target_position = self._clamp_position(wrist_position_robot + self._position_offset)

        gate = self._gate_wrist_orientation(frame.wrist_quaternion_xyzw, frame.timestamp)
        wrist_quaternion_xyzw = gate.quaternion_xyzw
        tracking_to_world_wxyz = _xyzw_to_wxyz(tracking_to_world)
        wrist_world_wxyz = _quat_multiply_wxyz(tracking_to_world_wxyz, _xyzw_to_wxyz(wrist_quaternion_xyzw))
        wrist_robot_wxyz = _quat_multiply_wxyz(
            _inverse_wxyz(_xyzw_to_wxyz(base_quaternion_np)), wrist_world_wxyz
        )
        if self._orientation_offset_wxyz is None:
            self._orientation_offset_wxyz = _quat_multiply_wxyz(
                _inverse_wxyz(wrist_robot_wxyz),
                _xyzw_to_wxyz(eef_relative_quaternion.cpu().numpy()),
            )
        target_quaternion_xyzw = _wxyz_to_xyzw(
            _quat_multiply_wxyz(wrist_robot_wxyz, self._orientation_offset_wxyz)
        )

        hand_positions = np.array(
            [positions_by_name[name] for name in SHARPA_ACTION_JOINTS], dtype=np.float64
        ) * self.config.finger_target_scale
        self._last_live_fingers = dict(zip(SHARPA_ACTION_JOINTS, hand_positions.tolist()))
        if frozen_fingers:
            unknown = set(frozen_fingers).difference(SHARPA_ACTION_JOINTS)
            if unknown:
                raise ValueError(f"Unknown frozen finger joints: {sorted(unknown)}")
            for index, name in enumerate(SHARPA_ACTION_JOINTS):
                if name in frozen_fingers:
                    hand_positions[index] = frozen_fingers[name]
        target = np.concatenate(
            [
                target_position,
                T.quat2axisangle(th.as_tensor(target_quaternion_xyzw, dtype=th.float32)).cpu().numpy(),
                hand_positions,
            ]
        )
        filtered = self.safety.apply(target, logger=LOGGER)
        self._last_wrist_diagnostics = WristActionDiagnostics(
            frame_timestamp=frame.timestamp,
            input_quaternion_xyzw=np.asarray(frame.wrist_quaternion_xyzw, dtype=np.float64).copy(),
            gated_quaternion_xyzw=wrist_quaternion_xyzw.copy(),
            target_quaternion_xyzw=target_quaternion_xyzw.copy(),
            target_position=target_position.copy(),
            target_axis_angle=target[3:6].copy(),
            filtered_position=filtered[:3].copy(),
            filtered_axis_angle=filtered[3:6].copy(),
            eef_position_before=eef_relative_position.cpu().numpy().copy(),
            eef_quaternion_before_xyzw=eef_relative_quaternion.cpu().numpy().copy(),
            gate_decision=gate.decision,
            gate_reason=gate.reason,
            hemisphere_corrected=gate.hemisphere_corrected,
        )
        return th.as_tensor(filtered, dtype=th.float32)

    def _gate_wrist_orientation(self, quaternion_xyzw: np.ndarray, timestamp: float) -> _WristGateResult:
        """Reject impossibly fast tracked-wrist rotations and hold the last good orientation.

        Hand trackers flip the estimated palm orientation by ~180 degrees on closed-fist or
        edge-on poses; chased through the IK this swings the whole arm. Two checks catch
        this: a per-frame jump gate (scaled for stale frames, hard-capped), and a slew gate
        that bounds sustained rotation against the accepted orientation from up to
        ``orientation_slew_horizon_s`` ago -- flips reached in sub-threshold steps trip the
        latter. Tracking resumes when a frame returns within the gates of the held
        orientation, when the new orientation persists for ``orientation_resync_s``, or on
        re-anchoring.
        """
        quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
        accepted = self._accepted_wrist_quaternion
        hemisphere_corrected = False
        if accepted is not None and float(np.dot(quaternion, accepted)) < 0.0:
            # Same rotation, opposite hemisphere: keep the stream sign-continuous so the
            # downstream axis-angle representation cannot jump across a 2*pi wrap.
            quaternion = -quaternion
            hemisphere_corrected = True
        if accepted is None:
            self._accept_wrist(quaternion, timestamp, reseed=True)
            return _WristGateResult(quaternion, "anchored", None, hemisphere_corrected)

        rejection_reason = None
        frame_gap = min(max(timestamp - self._accepted_wrist_timestamp, 0.0), 0.15)
        allowed = min(
            max(self.config.maximum_orientation_step, self.config.maximum_orientation_rate * frame_gap),
            self.config.maximum_orientation_allowance,
        )
        jump = _geodesic_angle_xyzw(quaternion, accepted)
        if jump > allowed:
            rejection_reason = (
                f"jumped {math.degrees(jump):.0f} deg in one frame (limit {math.degrees(allowed):.0f} deg)"
            )
        else:
            # Sub-threshold steps can still slew to a flipped estimate; bound the sustained
            # rotation against the oldest accepted orientation within the slew horizon.
            baseline_timestamp, baseline_quaternion = self._accepted_wrist_history[0]
            # Clamp at the horizon: while frames are being rejected the baseline ages, and
            # an unclamped budget would eventually inflate enough to accept the flip.
            elapsed = min(max(timestamp - baseline_timestamp, 0.0), self.config.orientation_slew_horizon_s)
            slew = _geodesic_angle_xyzw(quaternion, baseline_quaternion)
            slew_allowed = max(
                self.config.maximum_orientation_allowance,
                self.config.maximum_orientation_slew_rate * elapsed,
            )
            if slew > slew_allowed:
                rejection_reason = (
                    f"slewed {math.degrees(slew):.0f} deg over {elapsed:.2f} s "
                    f"(limit {math.degrees(slew_allowed):.0f} deg)"
                )

        if rejection_reason is None:
            self._accept_wrist(quaternion, timestamp)
            decision = "accepted_sign_corrected" if hemisphere_corrected else "accepted"
            return _WristGateResult(quaternion, decision, None, hemisphere_corrected)
        if self._rejection_started is None:
            self._rejection_started = timestamp
            LOGGER.warning("Tracked wrist orientation %s; holding the last orientation", rejection_reason)
        elif timestamp - self._rejection_started >= self.config.orientation_resync_s:
            LOGGER.warning(
                "Tracked wrist orientation stayed away for %.1f s (%s); re-synchronizing",
                timestamp - self._rejection_started,
                rejection_reason,
            )
            self._accept_wrist(quaternion, timestamp, reseed=True)
            return _WristGateResult(quaternion, "resynchronized", rejection_reason, hemisphere_corrected)
        return _WristGateResult(accepted.copy(), "held", rejection_reason, hemisphere_corrected)

    def _accept_wrist(self, quaternion: np.ndarray, timestamp: float, reseed: bool = False) -> None:
        self._accepted_wrist_quaternion = quaternion.copy()
        self._accepted_wrist_timestamp = timestamp
        self._rejection_started = None
        if reseed:
            self._accepted_wrist_history = deque()
        self._accepted_wrist_history.append((timestamp, self._accepted_wrist_quaternion))
        # Keep one entry at (or just beyond) the slew horizon as the comparison baseline.
        horizon = self.config.orientation_slew_horizon_s
        while len(self._accepted_wrist_history) >= 2 and self._accepted_wrist_history[1][0] <= timestamp - horizon:
            self._accepted_wrist_history.popleft()

    @property
    def last_live_fingers(self) -> dict[str, float] | None:
        """The most recent unfrozen finger targets by joint name, set by :meth:`action`."""
        return None if self._last_live_fingers is None else dict(self._last_live_fingers)

    @property
    def last_wrist_diagnostics(self) -> WristActionDiagnostics | None:
        """The most recent wrist transform and gate outcome, for diagnostic tooling."""

        diagnostics = self._last_wrist_diagnostics
        if diagnostics is None:
            return None
        return WristActionDiagnostics(
            frame_timestamp=diagnostics.frame_timestamp,
            input_quaternion_xyzw=diagnostics.input_quaternion_xyzw.copy(),
            gated_quaternion_xyzw=diagnostics.gated_quaternion_xyzw.copy(),
            target_quaternion_xyzw=diagnostics.target_quaternion_xyzw.copy(),
            target_position=diagnostics.target_position.copy(),
            target_axis_angle=diagnostics.target_axis_angle.copy(),
            filtered_position=diagnostics.filtered_position.copy(),
            filtered_axis_angle=diagnostics.filtered_axis_angle.copy(),
            eef_position_before=diagnostics.eef_position_before.copy(),
            eef_quaternion_before_xyzw=diagnostics.eef_quaternion_before_xyzw.copy(),
            gate_decision=diagnostics.gate_decision,
            gate_reason=diagnostics.gate_reason,
            hemisphere_corrected=diagnostics.hemisphere_corrected,
        )

    @property
    def measured_fingers(self) -> dict[str, float]:
        """The robot's current measured finger joint positions by joint name."""
        positions = self.robot.get_joint_positions()[list(self.finger_dof_indices)]
        return dict(zip(SHARPA_ACTION_JOINTS, positions.tolist()))

    def _clamp_position(self, position: np.ndarray) -> np.ndarray:
        target = np.asarray(position, dtype=np.float64).copy()
        target[2] = np.clip(target[2], self.config.minimum_z, self.config.maximum_z)
        shoulder = np.asarray(self.config.shoulder_position, dtype=np.float64)
        displacement = target - shoulder
        distance = np.linalg.norm(displacement)
        if distance > self.config.shoulder_reach:
            target = shoulder + displacement * (self.config.shoulder_reach / distance)
        if self._previous_target_position is not None:
            step = target - self._previous_target_position
            step_size = np.linalg.norm(step)
            if step_size > self.config.maximum_position_step:
                target = self._previous_target_position + step * (self.config.maximum_position_step / step_size)
        self._previous_target_position = target.copy()
        return target

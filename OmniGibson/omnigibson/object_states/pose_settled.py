from collections import deque

import torch as th

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)

# Number of consecutive simulator steps the object must be settled for PoseSettled to evaluate True.
m.DEFAULT_N_STEPS = 10

# --- Velocity ("motionless right now") thresholds ---------------------------------------------
# Root lin/ang mirror the *strict* thresholds in
# scripts/sampling/utils.py::_validate_object_state_stability (the B1K-canonical "is this
# object/robot still" check, and what BaseObject.keep_still() zeroes). The joint-velocity threshold
# matches VEL_TOLERANCE in controllers/multi_finger_gripper_controller.py -- the in-practice
# "this finger joint isn't moving" heuristic -- for consistency across the codebase.
m.DEFAULT_LIN_VEL_TOLERANCE = 0.001  # root linear velocity, m/s
m.DEFAULT_ANG_VEL_TOLERANCE = 0.005  # root angular velocity, rad/s
m.DEFAULT_JOINT_VEL_TOLERANCE = 0.02  # per-joint velocity, rad/s (matches gripper VEL_TOLERANCE)

# --- Configuration-drift thresholds (deviation from the window anchor) -------------------------
m.DEFAULT_POS_TOLERANCE = 0.01  # root position drift, m
m.DEFAULT_ORI_TOLERANCE = 0.05  # root orientation drift, rad
m.DEFAULT_JOINT_TOLERANCE = 0.05  # per-joint position drift, rad (or m for prismatic)


class PoseSettled(AbsoluteObjectState, BooleanStateMixin, UpdateStateMixin):
    """
    Temporal object state: ``True`` when an object (or robot) has come to rest -- i.e. it has been
    motionless for ``n_steps`` consecutive simulator steps.

    Stillness is checked over the **full articulated configuration**, not just the root link, so it
    is correct for mobile manipulators: a robot whose base is stationary while its arm/gripper is
    still moving is NOT settled. The per-step measurement combines the two signals B1K uses to
    decide an object is "settled" (compare ``scripts/sampling/utils.py::_validate_object_state_stability``
    and ``BaseObject.keep_still()``):

      1. **Velocity** -- the object's root linear & angular velocity AND *all* joint velocities are
         below their thresholds ("motionless right now").
      2. **Configuration drift** -- across the trailing ``n_steps`` window, the root pose AND all
         joint positions have stayed within tolerance of their values at the start of the window
         (catches slow drift that integrates from velocities too small to trip the velocity check).

    ``settled`` requires BOTH: velocity-still on every one of the ``n_steps`` steps AND the
    configuration drift staying within tolerance across the whole window.

    Unlike most object states this one accumulates history across simulator steps via ``_update()``
    (called once per ``og.sim.step()``), maintaining a fixed-length ring buffer of the last
    ``n_steps`` ``(root_pose, joint_positions, velocity_still)`` samples.

    To avoid imposing per-step cost on the (many) objects that carry this default state but are not
    referenced by any ``pose_settled`` goal, the state is inert until :meth:`activate` is called:
    its ``_update`` early-returns and ``_get_value`` returns ``False``. The BEHAVIOR task
    (``omnigibson.tasks.behavior_task``) activates only the objects that appear in a
    ``(pose_settled ...)`` goal atom and clears their history on episode reset.

    The cross-cutting "were the other goal conditions satisfied during the settled window" and the
    latency read-out live in the task, not here -- this state only owns the per-object ``settled``
    boolean and exposes the window bounds (:attr:`window_indices`, :attr:`settle_start_index`).

    Config (set via :meth:`set_config`; module-macro defaults otherwise):
        n_steps: number of consecutive settled steps required for ``settled`` to be True.
        lin_vel_tolerance / ang_vel_tolerance / joint_vel_tolerance: per-step velocity thresholds.
        pos_tolerance / ori_tolerance / joint_tolerance: window configuration-drift thresholds.
    """

    def __init__(self, obj):
        super().__init__(obj)
        self._n_steps = int(m.DEFAULT_N_STEPS)
        # Velocity thresholds
        self._lin_vel_tolerance = float(m.DEFAULT_LIN_VEL_TOLERANCE)
        self._ang_vel_tolerance = float(m.DEFAULT_ANG_VEL_TOLERANCE)
        self._joint_vel_tolerance = float(m.DEFAULT_JOINT_VEL_TOLERANCE)
        # Configuration-drift thresholds
        self._pos_tolerance = float(m.DEFAULT_POS_TOLERANCE)
        self._ori_tolerance = float(m.DEFAULT_ORI_TOLERANCE)
        self._joint_tolerance = float(m.DEFAULT_JOINT_TOLERANCE)

        # Whether this state actively tracks history (set by the task for pose_settled-goal objects)
        self._active = False
        # Ring buffer of (t, root_pos, root_quat, joint_pos_or_None, velocity_still) for the last
        # n_steps *distinct* simulator steps.
        self._history = deque(maxlen=self._n_steps)

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Any object/robot can settle / come to rest.
        return True, None

    # ------------------------------------------------------------------
    # External configuration / activation
    # ------------------------------------------------------------------

    def set_config(
        self,
        n_steps=None,
        lin_vel_tolerance=None,
        ang_vel_tolerance=None,
        joint_vel_tolerance=None,
        pos_tolerance=None,
        ori_tolerance=None,
        joint_tolerance=None,
    ):
        """Configure the settled check. Any argument left as ``None`` keeps the current (or
        module-default) value. Resizing ``n_steps`` clears the accumulated history."""
        if n_steps is not None:
            n_steps = int(n_steps)
            if n_steps != self._n_steps:
                self._n_steps = n_steps
                self._history = deque(maxlen=self._n_steps)
        if lin_vel_tolerance is not None:
            self._lin_vel_tolerance = float(lin_vel_tolerance)
        if ang_vel_tolerance is not None:
            self._ang_vel_tolerance = float(ang_vel_tolerance)
        if joint_vel_tolerance is not None:
            self._joint_vel_tolerance = float(joint_vel_tolerance)
        if pos_tolerance is not None:
            self._pos_tolerance = float(pos_tolerance)
        if ori_tolerance is not None:
            self._ori_tolerance = float(ori_tolerance)
        if joint_tolerance is not None:
            self._joint_tolerance = float(joint_tolerance)

    def activate(self, **config):
        """Enable per-step tracking (optionally (re)configuring via :meth:`set_config`) and clear
        any accumulated history."""
        if config:
            self.set_config(**config)
        self._active = True
        self.reset_history()

    def deactivate(self):
        """Disable per-step tracking and clear history."""
        self._active = False
        self.reset_history()

    def reset_history(self):
        """Clear the accumulated history (call at episode reset)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Read-only accessors (consumed by the task instrumentation)
    # ------------------------------------------------------------------

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def is_active(self):
        return self._active

    @property
    def is_settled(self):
        """Whether the object is currently settled (motionless for the last ``n_steps`` steps).
        Equivalent to ``get_value()`` but reads the accumulated history directly (no cache)."""
        return self._compute_settled()

    @property
    def window_indices(self):
        """Simulator-step indices spanned by the current full window, or ``[]`` if the buffer is
        not yet full (i.e. the object is not settled)."""
        if len(self._history) < self._n_steps:
            return []
        return [entry[0] for entry in self._history]

    @property
    def settle_start_index(self):
        """Simulator-step index of the start (anchor) of the current settled window, or ``-1`` if
        the object is not currently settled."""
        if len(self._history) < self._n_steps:
            return -1
        return self._history[0][0]

    # ------------------------------------------------------------------
    # Stillness measurement
    # ------------------------------------------------------------------

    @staticmethod
    def _ori_error(quat, anchor_quat):
        """Scalar orientation error (radians, full quaternion angular distance)."""
        dot = th.abs(th.sum(quat * anchor_quat))
        dot = th.clamp(dot, -1.0, 1.0)
        return (2.0 * th.acos(dot)).item()

    def _read_sample(self):
        """Read the current ``(root_pos, root_quat, joint_pos_or_None, velocity_still)`` sample.

        ``velocity_still`` is True iff the root linear & angular velocity AND every joint velocity
        are within their thresholds. Joint getters are only called for articulated objects (they
        assert ``n_joints > 0``); for rigid objects the joint terms are skipped.
        """
        pos, quat = self.obj.get_position_orientation()
        lin_ok = th.norm(self.obj.get_linear_velocity()).item() <= self._lin_vel_tolerance
        ang_ok = th.norm(self.obj.get_angular_velocity()).item() <= self._ang_vel_tolerance

        joint_pos = None
        joint_ok = True
        if self.obj.n_joints > 0:
            joint_pos = self.obj.get_joint_positions()
            joint_vel = self.obj.get_joint_velocities()
            joint_ok = joint_vel.numel() == 0 or th.max(th.abs(joint_vel)).item() <= self._joint_vel_tolerance

        velocity_still = bool(lin_ok and ang_ok and joint_ok)
        return pos.clone(), quat.clone(), (None if joint_pos is None else joint_pos.clone()), velocity_still

    def _compute_settled(self):
        """``True`` iff active, the buffer is full, every sample was velocity-still, and the root
        pose + joint positions stayed within tolerance of the window anchor (oldest sample)."""
        if not self._active or len(self._history) < self._n_steps:
            return False

        # (1) Velocity: every step in the window must be motionless.
        if not all(entry[4] for entry in self._history):
            return False

        # (2) Configuration drift: every sample within tolerance of the window anchor.
        _, anchor_pos, anchor_quat, anchor_jpos, _ = self._history[0]
        for _, pos, quat, jpos, _ in self._history:
            if th.norm(pos - anchor_pos).item() > self._pos_tolerance:
                return False
            if self._ori_error(quat, anchor_quat) > self._ori_tolerance:
                return False
            if anchor_jpos is not None and jpos is not None and jpos.numel() == anchor_jpos.numel():
                if th.max(th.abs(jpos - anchor_jpos)).item() > self._joint_tolerance:
                    return False
        return True

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------

    def _update(self):
        # Inert unless activated for a pose_settled goal object (keeps cost off unrelated objects).
        if not self._active:
            return
        t = og.sim.current_time_step_index
        # Record at most one sample per distinct simulator step (robust to >1 internal sub-step
        # within a single env step).
        if len(self._history) > 0 and self._history[-1][0] == t:
            return
        self._history.append((t, *self._read_sample()))

    def _get_value(self):
        return self._compute_settled()

    def _set_value(self, new_value, **kwargs):
        log.warning(
            f"PoseSettled.set_value({new_value!r}) on {self.obj.name} is a no-op; PoseSettled is a "
            f"temporal, read-only state computed from accumulated motion history."
        )
        return False

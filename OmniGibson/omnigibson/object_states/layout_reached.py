from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin
from omnigibson.utils.constants import JointType
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


class LayoutReached(AbsoluteObjectState, BooleanStateMixin):
    """
    Object state for the ``(layout_reached ?obj)`` predicate on an **articulated object**: True when every
    configured joint is within tolerance of its target position, i.e. the object's internal configuration (the
    pose of every moving link relative to the base link, which the joint positions parametrize exactly) matches
    the goal configuration.

    The group form of ``layout_reached`` (relative poses between the members of a declared group) is not an
    object state; it is evaluated by the iSpatialGym group entity (``ispatialgym.eval.group_entity``).

    All goal parameters come from an external loader (the TRO ``articulation_targets`` record) via
    :meth:`set_goal`. The state never mutates the simulator; ``_set_value`` is a no-op returning ``False``.

    Goal parameters (set_goal):
        joint_targets: ``{joint_name: {"position": float, "tolerance": float}}``. Revolute targets are radians,
            prismatic targets are meters. Every joint name must exist on the object (strict; unknown names raise).
        require_all_movable: if True (default), every non-fixed joint of the object must be listed so an
            unlisted door / drawer cannot drift unnoticed; a missing joint raises ``ValueError``.
    """

    def __init__(self, obj):
        super().__init__(obj)
        self._joint_targets = None  # dict: joint_name -> (position, tolerance)

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        return True, None

    # ------------------------------------------------------------------
    # External configuration
    # ------------------------------------------------------------------

    def set_goal(self, joint_targets=None, require_all_movable=True):
        """Configure the per-joint targets. Passing ``None`` clears the goal."""
        # A new goal invalidates any value cached earlier in this simulator step.
        self.clear_cache()
        if joint_targets is None:
            self._joint_targets = None
            return

        joints = getattr(self.obj, "joints", None) or {}
        targets = {}
        for name, spec in joint_targets.items():
            if name not in joints:
                raise ValueError(
                    f"LayoutReached: joint '{name}' not found on {self.obj.name}; available: {sorted(joints)}"
                )
            if isinstance(spec, dict):
                position, tolerance = spec.get("position"), spec.get("tolerance")
            else:
                position, tolerance = spec
            if position is None or tolerance is None:
                raise ValueError(f"LayoutReached: joint '{name}' on {self.obj.name} needs both position and tolerance")
            if not joints[name].is_single_dof:
                raise ValueError(f"LayoutReached: joint '{name}' on {self.obj.name} is not a single-DOF joint")
            targets[name] = (float(position), float(tolerance))

        if require_all_movable:
            movable = {n for n, j in joints.items() if j.joint_type != JointType.JOINT_FIXED}
            missing = sorted(movable - set(targets))
            if missing:
                raise ValueError(
                    f"LayoutReached: every non-fixed joint of {self.obj.name} must have a target; missing {missing}"
                )
        self._joint_targets = targets

    def clear_goal(self):
        self.set_goal(None)

    @property
    def has_goal(self):
        return bool(self._joint_targets)

    @property
    def joint_targets(self):
        return None if self._joint_targets is None else dict(self._joint_targets)

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------

    def get_error(self):
        """Per-joint absolute error.

        Returns:
            dict or None: ``{joint_name: {"error": float, "position": float, "goal": float, "tolerance": float,
            "joint_type": str, "within_tolerance": bool}}`` or ``None`` if no goal is configured. ``error`` is in
            radians for revolute joints and meters for prismatic joints.
        """
        if not self._joint_targets:
            return None
        errors = {}
        for name, (goal, tol) in self._joint_targets.items():
            joint = self.obj.joints[name]
            position = joint.get_state()[0].item()
            err = abs(position - goal)
            errors[name] = {
                "error": err,
                "position": position,
                "goal": goal,
                "tolerance": tol,
                "joint_type": joint.joint_type,
                "within_tolerance": err <= tol,
            }
        return errors

    def get_error_report(self):
        """Aggregate read-out: max error split by joint type (prismatic -> position, revolute -> orientation)."""
        errors = self.get_error()
        if errors is None:
            return None
        pos = [e["error"] for e in errors.values() if e["joint_type"] == JointType.JOINT_PRISMATIC]
        ori = [e["error"] for e in errors.values() if e["joint_type"] != JointType.JOINT_PRISMATIC]
        return {
            "position_error": max(pos) if pos else None,
            "orientation_error": max(ori) if ori else None,
            "within_tolerance": all(e["within_tolerance"] for e in errors.values()),
            "joints": errors,
        }

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------

    def _get_value(self):
        errors = self.get_error()
        if errors is None:
            return False
        return all(e["within_tolerance"] for e in errors.values())

    def _set_value(self, new_value, **kwargs):
        log.warning(
            f"LayoutReached.set_value({new_value!r}) on {self.obj.name} is a no-op; the goal configuration is read "
            f"from the TRO (use set_goal from the loader)."
        )
        return False

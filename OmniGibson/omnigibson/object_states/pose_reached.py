import math

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


_VALID_POS_AXES = (None, "z", "xy")
_VALID_ORI_AXES = (None, "yaw")


def _as_tensor(value):
    """Coerce a position/quaternion-like value (list, np.array, torch.Tensor) into a float32 tensor, or None."""
    if value is None:
        return None
    if isinstance(value, th.Tensor):
        return value.detach().to(dtype=th.float32)
    return th.tensor(value, dtype=th.float32)


class PoseReached(AbsoluteObjectState, BooleanStateMixin):
    """
    Object state that checks whether an object's (or a specified link's) pose matches a goal pose
    within tolerance, in a configurable reference frame and along configurable axes.

    All goal/init parameters are populated externally (typically from a TRO state file) via
    ``set_goal`` and ``set_init``. The state itself never mutates the simulator; ``_set_value``
    always returns ``False`` with a warning, since placement is handled by the loader that
    reads the TRO file.

    Goal parameters (set_goal):
        goal_pos: th.tensor([x, y, z]) target position in the configured ``frame``, or None to skip
            position checking.
        goal_ori: th.tensor([x, y, z, w]) target quaternion in the configured ``frame``, or None to
            skip orientation checking.
        pos_axes: which position axes to compare. One of:
            * ``None`` (default) -- compare full 3D position.
            * ``"z"``           -- compare only the z component.
            * ``"xy"``          -- compare only the (x, y) components.
        ori_axes: which orientation axes to compare. One of:
            * ``None`` (default) -- compare full quaternion (angular distance).
            * ``"yaw"``          -- compare only the yaw (z-axis Euler) component.
        link: name of a sensor or link on ``self.obj`` whose world pose to use as the current pose.
            ``None`` (default) means use ``self.obj.get_position_orientation()`` (root pose).
        frame: reference frame in which ``goal_pos`` / ``goal_ori`` are expressed and in which the
            comparison is performed. One of:
            * ``None`` or ``"world"`` (default) -- world frame, no transform applied.
            * ``"<obj_name>"`` -- root pose of that scene object becomes the comparison frame.
            * ``"<obj_name>::<link_or_sensor>"`` -- pose of that link/sensor on that scene
              object becomes the comparison frame.
            Resolution is strict: an unknown object/link raises ``ValueError``.
        pos_tolerance: float (meters) -- position error must be <= this for the goal to count as
            reached. If ``None``, the position component is skipped during the boolean check
            (``get_error`` still reports it).
        ori_tolerance: float (radians) -- orientation error must be <= this for the goal to count
            as reached. If ``None``, the orientation component is skipped during the boolean
            check (``get_error`` still reports it).

    Init parameters (set_init): read-only state kept for external loaders to consume when placing
    the object/robot at episode start. They are NOT used by this state's own ``get_value`` /
    ``set_value``.
        init_pos: th.tensor([x, y, z]) -- world-frame initial position.
        init_ori: th.tensor([x, y, z, w]) -- world-frame initial quaternion.
        init_joint_config: th.tensor or list -- initial joint positions (e.g. for the robot).
    """

    def __init__(self, obj):
        super().__init__(obj)
        # Goal parameters
        self._goal_pos = None
        self._goal_ori = None
        self._pos_axes = None
        self._ori_axes = None
        self._link = None
        self._frame = "world"
        self._pos_tolerance = None
        self._ori_tolerance = None

        # Init parameters (read-only, consumed by external loaders)
        self._init_pos = None
        self._init_ori = None
        self._init_joint_config = None

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        return True, None

    # ------------------------------------------------------------------
    # External setters
    # ------------------------------------------------------------------

    def set_goal(
        self,
        goal_pos=None,
        goal_ori=None,
        pos_axes=None,
        ori_axes=None,
        link=None,
        frame="world",
        pos_tolerance=None,
        ori_tolerance=None,
    ):
        """Populate the goal parameters used by ``get_value`` / ``get_error``."""
        assert pos_axes in _VALID_POS_AXES, f"pos_axes must be one of {_VALID_POS_AXES}, got {pos_axes!r}"
        assert ori_axes in _VALID_ORI_AXES, f"ori_axes must be one of {_VALID_ORI_AXES}, got {ori_axes!r}"

        self._goal_pos = _as_tensor(goal_pos)
        self._goal_ori = _as_tensor(goal_ori)
        self._pos_axes = pos_axes
        self._ori_axes = ori_axes
        self._link = link
        self._frame = frame if frame is not None else "world"
        self._pos_tolerance = float(pos_tolerance) if pos_tolerance is not None else None
        self._ori_tolerance = float(ori_tolerance) if ori_tolerance is not None else None

    def set_init(self, init_pos=None, init_ori=None, init_joint_config=None):
        """Populate the read-only init parameters. External loaders read these when placing
        the object/robot at episode start; this state itself does not consume them."""
        self._init_pos = _as_tensor(init_pos)
        self._init_ori = _as_tensor(init_ori)
        self._init_joint_config = _as_tensor(init_joint_config)

    # Read-only accessors (kept for external loaders / debugging) -------------------------------

    @property
    def goal_pos(self):
        return self._goal_pos

    @property
    def goal_ori(self):
        return self._goal_ori

    @property
    def pos_axes(self):
        return self._pos_axes

    @property
    def ori_axes(self):
        return self._ori_axes

    @property
    def link(self):
        return self._link

    @property
    def frame(self):
        return self._frame

    @property
    def pos_tolerance(self):
        return self._pos_tolerance

    @property
    def ori_tolerance(self):
        return self._ori_tolerance

    @property
    def init_pos(self):
        return self._init_pos

    @property
    def init_ori(self):
        return self._init_ori

    @property
    def init_joint_config(self):
        return self._init_joint_config

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_link_pose(obj, name):
        """Look up world pose of a sensor or link by name on ``obj``.

        Returns ``(pos, quat)`` torch tensors, or ``(None, None)`` if not found.
        Cameras prefer ``camera_parameters["cameraViewTransform"]`` when populated
        (mirrors the convention used by the iSpatialGym eval/sampling pipeline).
        """
        sensors = getattr(obj, "sensors", None)
        if sensors and name in sensors:
            camera = sensors[name]
            direct_cam_pose = camera.camera_parameters.get("cameraViewTransform")
            if direct_cam_pose is not None and not np.allclose(direct_cam_pose, np.zeros(16)):
                cam_mat = np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T)
                return T.mat2pose(th.tensor(cam_mat, dtype=th.float32))
            return camera.get_position_orientation()
        links = getattr(obj, "links", None)
        if links and name in links:
            return links[name].get_position_orientation()
        return None, None

    def _get_current_pose(self):
        """World pose of the link/sensor on ``self.obj`` (or root pose if ``link`` is None).

        Strict: an unknown ``link`` raises rather than silently falling back to the root pose.
        """
        if self._link is None:
            return self.obj.get_position_orientation()
        pos, quat = self._lookup_link_pose(self.obj, self._link)
        if pos is None:
            raise ValueError(
                f"PoseReached: link '{self._link}' not found on {self.obj.name}"
            )
        return pos, quat

    def _resolve_frame_pose(self):
        """World pose of the comparison frame's origin, or ``None`` for the world frame.

        ``frame`` is interpreted strictly:
        * ``None`` or ``"world"`` -> world frame (no transform).
        * ``"<obj_name>"`` -> root pose of that scene object.
        * ``"<obj_name>::<link_or_sensor>"`` -> pose of that link/sensor on that scene object.

        Any unresolved name raises ``ValueError`` -- no fallbacks.
        """
        if self._frame is None or self._frame == "world":
            return None

        if "::" in self._frame:
            obj_name, link_name = self._frame.split("::", 1)
        else:
            obj_name, link_name = self._frame, None

        other = self.obj.scene.object_registry("name", obj_name)
        if other is None:
            raise ValueError(
                f"PoseReached: frame '{self._frame}' references unknown object '{obj_name}'"
            )

        if link_name is None:
            return other.get_position_orientation()

        pos, quat = self._lookup_link_pose(other, link_name)
        if pos is None:
            raise ValueError(
                f"PoseReached: frame '{self._frame}' references unknown link/sensor '{link_name}' on '{obj_name}'"
            )
        return pos, quat

    def _pose_in_frame(self, pos_world, quat_world):
        """Express a world pose in the configured comparison frame."""
        frame_pose = self._resolve_frame_pose()
        if frame_pose is None:
            return pos_world, quat_world
        frame_pos, frame_quat = frame_pose
        return T.relative_pose_transform(pos_world, quat_world, frame_pos, frame_quat)

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------

    def _pos_error(self, current_pos):
        """Scalar position error (meters) in the configured axes."""
        if self._goal_pos is None:
            return None
        delta = current_pos - self._goal_pos
        if self._pos_axes is None:
            return th.norm(delta).item()
        if self._pos_axes == "z":
            return abs(delta[2].item())
        if self._pos_axes == "xy":
            return th.norm(delta[:2]).item()
        raise ValueError(f"Unsupported pos_axes={self._pos_axes!r}")

    def _ori_error(self, current_quat):
        """Scalar orientation error (radians) in the configured axes."""
        if self._goal_ori is None:
            return None
        if self._ori_axes is None:
            # Full quaternion angular distance.
            dot = th.abs(th.sum(current_quat * self._goal_ori))
            dot = th.clamp(dot, -1.0, 1.0)
            return (2.0 * th.acos(dot)).item()
        if self._ori_axes == "yaw":
            current_yaw = T.quat2euler(current_quat)[2].item()
            goal_yaw = T.quat2euler(self._goal_ori)[2].item()
            diff = (current_yaw - goal_yaw + math.pi) % (2.0 * math.pi) - math.pi
            return abs(diff)
        raise ValueError(f"Unsupported ori_axes={self._ori_axes!r}")

    def get_error(self):
        """Return ``(pos_error, ori_error)`` in the configured frame and axes.

        Each component is ``None`` when its corresponding goal is not set (so a depth-only goal
        returns ``(pos_error, None)`` and an orientation-only goal returns ``(None, ori_error)``).
        Returns ``(None, None)`` when no goal has been configured.
        """
        if self._goal_pos is None and self._goal_ori is None:
            return None, None
        current_pos, current_quat = self._get_current_pose()
        current_pos, current_quat = self._pose_in_frame(current_pos, current_quat)
        return self._pos_error(current_pos), self._ori_error(current_quat)

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------

    def _get_value(self):
        # No goal configured -> condition is not (yet) reached.
        if self._goal_pos is None and self._goal_ori is None:
            return False

        pos_error, ori_error = self.get_error()

        if pos_error is not None and self._pos_tolerance is not None and pos_error > self._pos_tolerance:
            return False
        if ori_error is not None and self._ori_tolerance is not None and ori_error > self._ori_tolerance:
            return False
        return True

    def _set_value(self, new_value, **kwargs):
        log.warning(
            f"PoseReached.set_value({new_value!r}) on {self.obj.name} is a no-op; "
            f"PoseReached doesn't support _set_value this way -- it should be read from the TRO file "
            f"(use set_goal / set_init from the loader)."
        )
        return False

"""Lightweight commanded and measured end-effector pose markers."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation


X_AXIS = (1.0, 0.1, 0.1, 1.0)
Y_AXIS = (0.1, 1.0, 0.1, 1.0)
Z_AXIS = (0.1, 0.4, 1.0, 1.0)
AXIS_COLORS = (X_AXIS, Y_AXIS, Z_AXIS)
TARGET = (1.0, 0.0, 1.0, 1.0)
ACTUAL = (0.0, 0.85, 1.0, 1.0)
GREEN = (0.15, 0.9, 0.25, 1.0)
AMBER = (1.0, 0.65, 0.0, 1.0)
RED = (1.0, 0.1, 0.1, 1.0)


def _as_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _normalized_quaternion_xyzw(value) -> np.ndarray:
    quaternion = _as_vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("quaternion must have non-zero norm")
    return quaternion / norm


def xyz_frame_segments(position, quaternion_xyzw, axis_length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return three line segments representing a pose's local XYZ axes."""

    position = _as_vector(position, 3, "frame position")
    if axis_length_m <= 0.0:
        raise ValueError("axis_length_m must be positive")
    rotation = Rotation.from_quat(_normalized_quaternion_xyzw(quaternion_xyzw))
    starts = np.repeat(position[None, :], 3, axis=0)
    ends = starts + rotation.apply(np.eye(3) * axis_length_m)
    return starts, ends


def pose_ring_segments(
    position,
    quaternion_xyzw,
    radius_m: float,
    segments_per_ring: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return three pose-oriented circle outlines in the local XY, XZ, and YZ planes."""

    position = _as_vector(position, 3, "ring position")
    if radius_m <= 0.0 or segments_per_ring < 4:
        raise ValueError("ring radius must be positive and each ring needs at least four segments")
    angles = np.linspace(0.0, 2.0 * np.pi, segments_per_ring, endpoint=False)
    cosine = np.cos(angles) * radius_m
    sine = np.sin(angles) * radius_m
    local_rings = np.stack(
        (
            np.column_stack((cosine, sine, np.zeros_like(cosine))),
            np.column_stack((cosine, np.zeros_like(cosine), sine)),
            np.column_stack((np.zeros_like(cosine), cosine, sine)),
        )
    )
    rotation = Rotation.from_quat(_normalized_quaternion_xyzw(quaternion_xyzw))
    world_rings = rotation.apply(local_rings.reshape(-1, 3)).reshape(3, segments_per_ring, 3) + position
    return world_rings.reshape(-1, 3), np.roll(world_rings, -1, axis=1).reshape(-1, 3)


def tracking_error_color(error_m: float, *, warning_error_m: float, critical_error_m: float) -> tuple:
    """Color the target-to-EEF connector by positional tracking error."""

    if error_m > critical_error_m:
        return RED
    if error_m > warning_error_m:
        return AMBER
    return GREEN


@dataclass(frozen=True)
class ArmMarkerVisualizationConfig:
    draw_frequency_hz: float = 10.0
    target_axis_length_m: float = 0.12
    eef_axis_length_m: float = 0.10
    target_axis_line_size: float = 4.0
    eef_axis_line_size: float = 3.0
    ring_radius_m: float = 0.045
    ring_line_size: float = 2.0
    segments_per_ring: int = 24
    connector_line_size: float = 2.0
    origin_point_size: float = 18.0
    warning_tracking_error_m: float = 0.02
    critical_tracking_error_m: float = 0.05

    def __post_init__(self) -> None:
        if self.draw_frequency_hz <= 0.0:
            raise ValueError("draw_frequency_hz must be positive")
        if min(
            self.target_axis_length_m,
            self.eef_axis_length_m,
            self.target_axis_line_size,
            self.eef_axis_line_size,
            self.ring_radius_m,
            self.ring_line_size,
            self.connector_line_size,
            self.origin_point_size,
        ) <= 0.0:
            raise ValueError("marker dimensions must be positive")
        if self.segments_per_ring < 4:
            raise ValueError("segments_per_ring must be at least four")
        if not 0.0 <= self.warning_tracking_error_m < self.critical_tracking_error_m:
            raise ValueError("tracking-error thresholds are invalid")


class ArmPoseMarkerVisualizer:
    """Draw independently toggleable commanded-target and measured-EEF frames."""

    def __init__(
        self,
        draw_interface,
        *,
        config: ArmMarkerVisualizationConfig = ArmMarkerVisualizationConfig(),
        target_visible: bool = True,
        eef_visible: bool = True,
    ) -> None:
        self.draw = draw_interface
        self.config = config
        self.target_visible = target_visible
        self.eef_visible = eef_visible
        self._last_draw = float("-inf")
        self._closed = False

    def update_due(self) -> bool:
        """Return whether simulator state should be read for the next redraw."""

        return (
            not self._closed
            and (self.target_visible or self.eef_visible)
            and time.monotonic() - self._last_draw >= 1.0 / self.config.draw_frequency_hz
        )

    def toggle_target(self) -> bool:
        self.target_visible = not self.target_visible
        self._invalidate()
        return self.target_visible

    def toggle_eef(self) -> bool:
        self.eef_visible = not self.eef_visible
        self._invalidate()
        return self.eef_visible

    def reset(self) -> None:
        self._invalidate()

    def update(
        self,
        *,
        target_position_robot,
        target_quaternion_robot_xyzw,
        actual_position_world,
        actual_quaternion_world_xyzw,
        base_position_world,
        base_quaternion_world_xyzw,
    ) -> None:
        if not self.update_due():
            return
        self._last_draw = time.monotonic()

        base_position = _as_vector(base_position_world, 3, "base_position_world")
        base_rotation = Rotation.from_quat(_normalized_quaternion_xyzw(base_quaternion_world_xyzw))
        target_position_world = base_rotation.apply(
            _as_vector(target_position_robot, 3, "target_position_robot")
        ) + base_position
        target_rotation_world = base_rotation * Rotation.from_quat(
            _normalized_quaternion_xyzw(target_quaternion_robot_xyzw)
        )
        actual_position_world = _as_vector(actual_position_world, 3, "actual_position_world")
        actual_quaternion_world = _normalized_quaternion_xyzw(actual_quaternion_world_xyzw)

        starts = []
        ends = []
        colors = []
        sizes = []
        origins = []
        origin_colors = []
        origin_sizes = []
        if self.target_visible:
            target_starts, target_ends = xyz_frame_segments(
                target_position_world,
                target_rotation_world.as_quat(),
                self.config.target_axis_length_m,
            )
            starts.extend(target_starts)
            ends.extend(target_ends)
            colors.extend(AXIS_COLORS)
            sizes.extend([self.config.target_axis_line_size] * 3)
            ring_starts, ring_ends = pose_ring_segments(
                target_position_world,
                target_rotation_world.as_quat(),
                self.config.ring_radius_m,
                self.config.segments_per_ring,
            )
            starts.extend(ring_starts)
            ends.extend(ring_ends)
            colors.extend([TARGET] * len(ring_starts))
            sizes.extend([self.config.ring_line_size] * len(ring_starts))
            origins.append(target_position_world)
            origin_colors.append(TARGET)
            origin_sizes.append(self.config.origin_point_size)
        if self.eef_visible:
            eef_starts, eef_ends = xyz_frame_segments(
                actual_position_world,
                actual_quaternion_world,
                self.config.eef_axis_length_m,
            )
            starts.extend(eef_starts)
            ends.extend(eef_ends)
            colors.extend(AXIS_COLORS)
            sizes.extend([self.config.eef_axis_line_size] * 3)
            ring_starts, ring_ends = pose_ring_segments(
                actual_position_world,
                actual_quaternion_world,
                self.config.ring_radius_m,
                self.config.segments_per_ring,
            )
            starts.extend(ring_starts)
            ends.extend(ring_ends)
            colors.extend([ACTUAL] * len(ring_starts))
            sizes.extend([self.config.ring_line_size] * len(ring_starts))
            origins.append(actual_position_world)
            origin_colors.append(ACTUAL)
            origin_sizes.append(self.config.origin_point_size)
        if self.target_visible and self.eef_visible:
            starts.append(actual_position_world)
            ends.append(target_position_world)
            colors.append(
                tracking_error_color(
                    float(np.linalg.norm(target_position_world - actual_position_world)),
                    warning_error_m=self.config.warning_tracking_error_m,
                    critical_error_m=self.config.critical_tracking_error_m,
                )
            )
            sizes.append(self.config.connector_line_size)

        self.draw.clear_lines()
        self.draw.clear_points()
        if starts:
            self.draw.draw_lines(
                [tuple(point) for point in starts],
                [tuple(point) for point in ends],
                colors,
                sizes,
            )
        if origins:
            self.draw.draw_points([tuple(point) for point in origins], origin_colors, origin_sizes)

    def _invalidate(self) -> None:
        self._last_draw = float("-inf")
        self.draw.clear_lines()
        self.draw.clear_points()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.draw.clear_lines()
        self.draw.clear_points()

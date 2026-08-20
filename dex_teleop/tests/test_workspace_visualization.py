import numpy as np
from scipy.spatial.transform import Rotation

from dex_teleop.omnigibson.workspace_visualization import (
    ACTUAL,
    AMBER,
    AXIS_COLORS,
    GREEN,
    RED,
    TARGET,
    ArmPoseMarkerVisualizer,
    pose_ring_segments,
    tracking_error_color,
    xyz_frame_segments,
)


def test_xyz_frame_segments_follow_pose_orientation():
    quaternion = Rotation.from_euler("z", 90.0, degrees=True).as_quat()

    starts, ends = xyz_frame_segments([1.0, 2.0, 3.0], quaternion, 0.1)

    assert np.allclose(starts, [[1.0, 2.0, 3.0]] * 3)
    assert np.allclose(
        ends,
        [
            [1.0, 2.1, 3.0],
            [0.9, 2.0, 3.0],
            [1.0, 2.0, 3.1],
        ],
    )


def test_tracking_error_colors():
    kwargs = {"warning_error_m": 0.02, "critical_error_m": 0.05}

    assert tracking_error_color(0.01, **kwargs) == GREEN
    assert tracking_error_color(0.03, **kwargs) == AMBER
    assert tracking_error_color(0.06, **kwargs) == RED


def test_pose_ring_segments_follow_wrist_orientation():
    quaternion = Rotation.from_euler("z", 90.0, degrees=True).as_quat()

    starts, ends = pose_ring_segments([1.0, 2.0, 3.0], quaternion, 0.05, 8)

    assert starts.shape == ends.shape == (24, 3)
    assert np.allclose(np.linalg.norm(starts - [1.0, 2.0, 3.0], axis=1), 0.05)
    assert np.allclose(starts[0], [1.0, 2.05, 3.0])


class _FakeDraw:
    def __init__(self):
        self.points = None
        self.lines = None

    def clear_points(self):
        self.points = None

    def clear_lines(self):
        self.lines = None

    def draw_points(self, points, colors, sizes):
        self.points = (points, colors, sizes)

    def draw_lines(self, starts, ends, colors, sizes):
        self.lines = (starts, ends, colors, sizes)


def test_live_renderer_draws_target_and_measured_xyz_frames():
    draw = _FakeDraw()
    visualizer = ArmPoseMarkerVisualizer(draw)
    visualizer.update(
        target_position_robot=[0.1, 0.2, 0.3],
        target_quaternion_robot_xyzw=[0.0, 0.0, 0.0, 1.0],
        actual_position_world=[1.0, 2.0, 3.0],
        actual_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        base_position_world=[1.0, 1.0, 1.0],
        base_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    points, point_colors, _point_sizes = draw.points
    starts, ends, line_colors, _line_sizes = draw.lines
    assert np.allclose(points, [[1.1, 1.2, 1.3], [1.0, 2.0, 3.0]])
    assert point_colors == [TARGET, ACTUAL]
    assert line_colors[:3] == list(AXIS_COLORS)
    assert line_colors[3:75] == [TARGET] * 72
    assert line_colors[75:78] == list(AXIS_COLORS)
    assert line_colors[78:150] == [ACTUAL] * 72
    assert line_colors[-1] == RED
    assert np.allclose(starts[:3], [[1.1, 1.2, 1.3]] * 3)
    assert np.allclose(ends[75], [1.1, 2.0, 3.0])
    visualizer.close()


def test_target_and_eef_frames_toggle_independently():
    draw = _FakeDraw()
    visualizer = ArmPoseMarkerVisualizer(draw)

    assert visualizer.toggle_target() is False
    visualizer.update(
        target_position_robot=[0.1, 0.2, 0.3],
        target_quaternion_robot_xyzw=[0.0, 0.0, 0.0, 1.0],
        actual_position_world=[1.0, 2.0, 3.0],
        actual_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        base_position_world=[1.0, 1.0, 1.0],
        base_quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    _points, point_colors, _point_sizes = draw.points
    _starts, _ends, line_colors, _line_sizes = draw.lines
    assert point_colors == [ACTUAL]
    assert line_colors[:3] == list(AXIS_COLORS)
    assert line_colors[3:] == [ACTUAL] * 72
    assert visualizer.toggle_eef() is False
    assert visualizer.update_due() is False
    assert draw.points is None
    assert draw.lines is None
    visualizer.close()

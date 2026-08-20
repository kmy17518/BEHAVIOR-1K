import json
import math
from types import SimpleNamespace

import numpy as np

from dex_teleop.diagnostics import WristFlipDiagnosticConfig, WristFlipRecorder
from dex_teleop.tracking.hts import HTSSource
from dex_teleop.types import HandFrame, Handedness, MEDIAPIPE_JOINT_NAMES


def _quaternion_z(degrees):
    angle = math.radians(degrees)
    return np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def _frame(timestamp, quaternion):
    landmarks = np.arange(63, dtype=np.float64).reshape(21, 3) / 1000.0
    return HandFrame(
        timestamp=timestamp,
        receipt_timestamp=timestamp,
        handedness=Handedness.RIGHT,
        joints=dict(zip(MEDIAPIPE_JOINT_NAMES, landmarks, strict=True)),
        wrist_position=np.zeros(3),
        wrist_quaternion_xyzw=quaternion,
        source="hts",
    )


def _source_diagnostics(timestamp, quaternion, skew=0.0):
    return SimpleNamespace(
        raw_wrist_quaternion_xyzw=quaternion,
        raw_wrist_position_unity=np.zeros(3),
        raw_landmarks_unity=np.arange(63, dtype=np.float64).reshape(21, 3) / 1000.0,
        wrist_receipt_timestamp=timestamp,
        landmarks_receipt_timestamp=timestamp + skew,
        pair_skew_seconds=abs(skew),
    )


def _action_diagnostics(frame, *, gate_decision="accepted", target=None):
    target = frame.wrist_quaternion_xyzw if target is None else target
    return SimpleNamespace(
        target_quaternion_xyzw=target,
        target_position=np.zeros(3),
        filtered_axis_angle=np.zeros(3),
        filtered_position=np.zeros(3),
        gated_quaternion_xyzw=target,
        target_axis_angle=np.zeros(3),
        eef_position_before=np.zeros(3),
        eef_quaternion_before_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        gate_decision=gate_decision,
        gate_reason="synthetic jump" if gate_decision == "held" else None,
        hemisphere_corrected=False,
    )


def test_hts_exposes_raw_records_and_receive_skew_for_published_frame():
    source = HTSSource()
    landmarks = ", ".join(str(value) for value in np.arange(63) / 1000.0)
    source._handle_line(
        "Right wrist | f = 7 | t = 1000:, 1, 2, 3, 0, 0, 0, 2",
        receipt_timestamp=10.0,
    )
    source._handle_line(
        f"Right landmarks | f = 7 | t = 1000:, {landmarks}",
        receipt_timestamp=10.025,
    )

    frame = source.read(Handedness.RIGHT)
    assert frame is not None
    diagnostics = source.diagnostics_for_frame(frame)
    assert diagnostics is not None
    assert diagnostics.source_frame_id == 7
    assert np.isclose(diagnostics.pair_skew_seconds, 0.025)
    assert np.allclose(diagnostics.raw_wrist_position_unity, [1, 2, 3])
    assert np.allclose(diagnostics.raw_wrist_quaternion_xyzw, [0, 0, 0, 1])


def test_recorder_classifies_a_tracker_jump_that_the_gate_contains(tmp_path):
    recorder = WristFlipRecorder(
        tmp_path,
        WristFlipDiagnosticConfig(flip_threshold_degrees=90.0, post_event_seconds=0.0),
    )
    first = _frame(10.0, _quaternion_z(0))
    second = _frame(10.02, _quaternion_z(170))
    recorder.observe(
        step=0,
        snapshot=SimpleNamespace(frame=first),
        source_diagnostics=_source_diagnostics(10.0, first.wrist_quaternion_xyzw),
        action_diagnostics=_action_diagnostics(first),
        eef_position_after=np.zeros(3),
        eef_quaternion_after_xyzw=_quaternion_z(0),
    )
    recorder.observe(
        step=1,
        snapshot=SimpleNamespace(frame=second),
        source_diagnostics=_source_diagnostics(10.02, second.wrist_quaternion_xyzw),
        action_diagnostics=_action_diagnostics(second, gate_decision="held", target=_quaternion_z(0)),
        eef_position_after=np.zeros(3),
        eef_quaternion_after_xyzw=_quaternion_z(0),
    )
    recorder.close()

    events = json.loads((tmp_path / "events.json").read_text())
    assert events[0]["classification"] == "tracker_jump_contained"
    assert "raw_tracker_rotation_jump" in events[0]["triggers"]
    assert "tracker_jump_contained" in (tmp_path / "summary.md").read_text()


def test_manual_marker_classifies_a_large_multiframe_tracker_slew(tmp_path):
    recorder = WristFlipRecorder(tmp_path, WristFlipDiagnosticConfig(flip_threshold_degrees=90.0))
    for step, degrees in enumerate((0, 50, 100)):
        frame = _frame(20.0 + step * 0.02, _quaternion_z(degrees))
        if step == 2:
            recorder.mark_manual_event()
        recorder.observe(
            step=step,
            snapshot=SimpleNamespace(frame=frame),
            source_diagnostics=_source_diagnostics(frame.timestamp, frame.wrist_quaternion_xyzw),
            action_diagnostics=_action_diagnostics(frame),
            eef_position_after=np.zeros(3),
            eef_quaternion_after_xyzw=frame.wrist_quaternion_xyzw,
        )
    recorder.close()

    events = json.loads((tmp_path / "events.json").read_text())
    assert len(events) == 1
    assert events[0]["classification"] == "tracker_orientation_slew"


def test_recorder_classifies_franka_joint_space_rollover(tmp_path):
    recorder = WristFlipRecorder(
        tmp_path,
        WristFlipDiagnosticConfig(
            joint_rotation_threshold_degrees=90.0,
            joint_rotation_window_seconds=5.0,
        ),
    )
    lower_limits = np.full(7, -math.pi)
    upper_limits = np.full(7, math.pi)
    joint_positions = (
        np.radians([0, 0, 0, 0, 0, 170, 0]),
        np.radians([0, 0, 0, 0, 50, 179, -50]),
        np.radians([0, 0, 0, 0, 100, 180, -100]),
    )
    for step, positions in enumerate(joint_positions):
        frame = _frame(30.0 + step * 0.02, _quaternion_z(step))
        recorder.observe(
            step=step,
            snapshot=SimpleNamespace(frame=frame),
            source_diagnostics=_source_diagnostics(frame.timestamp, frame.wrist_quaternion_xyzw),
            action_diagnostics=_action_diagnostics(frame),
            eef_position_after=np.zeros(3),
            eef_quaternion_after_xyzw=frame.wrist_quaternion_xyzw,
            arm_joint_positions=positions,
            arm_joint_lower_limits=lower_limits,
            arm_joint_upper_limits=upper_limits,
        )
    recorder.close()

    events = json.loads((tmp_path / "events.json").read_text())
    assert len(events) == 1
    assert events[0]["classification"] == "ik_joint_space_rollover"
    assert "J5 accumulated 100.0°" in events[0]["evidence"]

    trace = (tmp_path / "trace.csv").read_text()
    assert "franka_j5_cumulative_rotation_deg" in trace.splitlines()[0]
    assert "franka_j6_limit_margin_deg" in trace.splitlines()[0]
    summary = (tmp_path / "summary.md").read_text()
    assert "## Franka joint motion and limits" in summary
    assert "| J5 | 50.00° | 100.00° | 80.00° |" in summary
    assert "| J6 | 9.00° | 10.00° | 0.00° |" in summary

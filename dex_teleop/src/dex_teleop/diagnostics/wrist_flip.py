"""Trace and classify wrist-orientation discontinuities during live teleoperation."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


def _normalized(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("Quaternion must have non-zero norm")
    return value / norm


def _rotation_delta_degrees(first: Any | None, second: Any | None) -> float | None:
    if first is None or second is None:
        return None
    dot = abs(float(np.dot(_normalized(first), _normalized(second))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def _signed_dot(first: Any | None, second: Any | None) -> float | None:
    if first is None or second is None:
        return None
    return float(np.dot(_normalized(first), _normalized(second)))


def _axis_angle_to_quaternion(axis_angle: Any) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    xyz = vector / angle * math.sin(angle / 2.0)
    return np.concatenate([xyz, [math.cos(angle / 2.0)]])


def _vector_delta_centimeters(first: Any | None, second: Any | None) -> float | None:
    if first is None or second is None:
        return None
    return float(np.linalg.norm(np.asarray(second) - np.asarray(first)) * 100.0)


def _rms_delta_centimeters(first: Any | None, second: Any | None) -> float | None:
    if first is None or second is None:
        return None
    difference = np.asarray(second) - np.asarray(first)
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=-1))) * 100.0)


def _number(value: float | int | None) -> float | int | str:
    return "" if value is None else value


def _put_vector(row: dict[str, Any], prefix: str, value: Any | None, labels: str) -> None:
    array = None if value is None else np.asarray(value, dtype=np.float64).reshape(-1)
    for index, label in enumerate(labels):
        row[f"{prefix}_{label}"] = "" if array is None else float(array[index])


@dataclass(frozen=True)
class WristFlipDiagnosticConfig:
    flip_threshold_degrees: float = 90.0
    conversion_tolerance_degrees: float = 1.0
    pair_skew_threshold_ms: float = 20.0
    sign_only_tolerance_degrees: float = 2.0
    post_event_seconds: float = 2.0
    continue_after_event: bool = False
    tracking_yaw_degrees: float = 0.0
    joint_rotation_threshold_degrees: float = 150.0
    joint_rotation_window_seconds: float = 5.0
    joint_limit_margin_threshold_degrees: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.flip_threshold_degrees <= 180.0:
            raise ValueError("flip_threshold_degrees must be in (0, 180]")
        if self.conversion_tolerance_degrees <= 0.0:
            raise ValueError("conversion_tolerance_degrees must be positive")
        if self.pair_skew_threshold_ms < 0.0:
            raise ValueError("pair_skew_threshold_ms must be non-negative")
        if self.sign_only_tolerance_degrees <= 0.0:
            raise ValueError("sign_only_tolerance_degrees must be positive")
        if self.post_event_seconds < 0.0:
            raise ValueError("post_event_seconds must be non-negative")
        if not math.isfinite(self.tracking_yaw_degrees):
            raise ValueError("tracking_yaw_degrees must be finite")
        if self.joint_rotation_threshold_degrees <= 0.0:
            raise ValueError("joint_rotation_threshold_degrees must be positive")
        if self.joint_rotation_window_seconds <= 0.0:
            raise ValueError("joint_rotation_window_seconds must be positive")
        if self.joint_limit_margin_threshold_degrees < 0.0:
            raise ValueError("joint_limit_margin_threshold_degrees must be non-negative")


_FRANKA_JOINT_LABELS = tuple(f"j{index}" for index in range(1, 8))
_FRANKA_WRIST_JOINT_START = 4
_FRANKA_JOINT_METRICS = (
    "position_rad",
    "lower_limit_rad",
    "upper_limit_rad",
    "lower_margin_deg",
    "upper_margin_deg",
    "limit_margin_deg",
    "step_rotation_deg",
    "cumulative_rotation_deg",
    "window_rotation_deg",
)


_VECTOR_COLUMNS = {
    "raw_position_unity": "xyz",
    "raw_quaternion_unity": "xyzw",
    "canonical_position": "xyz",
    "canonical_quaternion": "xyzw",
    "gated_quaternion": "xyzw",
    "target_position": "xyz",
    "target_quaternion": "xyzw",
    "target_axis_angle": "xyz",
    "command_position": "xyz",
    "command_axis_angle": "xyz",
    "eef_position_before": "xyz",
    "eef_quaternion_before": "xyzw",
    "eef_position_after": "xyz",
    "eef_quaternion_after": "xyzw",
}

_SCALAR_COLUMNS = (
    "sample",
    "step",
    "elapsed_s",
    "segment",
    "new_tracking_frame",
    "source_diagnostics_available",
    "frame_timestamp",
    "receipt_timestamp",
    "frame_age_ms",
    "source_frame_id",
    "source_timestamp_ns",
    "source_dt_ms",
    "receipt_dt_ms",
    "wrist_receipt_timestamp",
    "landmarks_receipt_timestamp",
    "pair_skew_ms",
    "raw_signed_dot",
    "raw_rotation_delta_deg",
    "canonical_signed_dot",
    "canonical_rotation_delta_deg",
    "conversion_delta_error_deg",
    "canonical_position_delta_cm",
    "local_landmark_rms_delta_cm",
    "target_rotation_delta_deg",
    "target_position_delta_cm",
    "command_rotation_delta_deg",
    "command_position_delta_cm",
    "eef_rotation_delta_deg",
    "eef_position_delta_cm",
    "franka_joint_positions_available",
    "franka_max_joint_step_rotation_deg",
    "franka_max_joint_step_rotation_joint",
    "franka_max_joint_window_rotation_deg",
    "franka_max_joint_window_rotation_joint",
    "franka_max_wrist_joint_window_rotation_deg",
    "franka_max_wrist_joint_window_rotation_joint",
    "franka_total_cumulative_rotation_deg",
    "franka_min_joint_limit_margin_deg",
    "franka_min_joint_limit_margin_joint",
    "gate_decision",
    "gate_reason",
    "hemisphere_corrected",
    "manual_mark",
    "triggers",
)

_FRANKA_JOINT_COLUMNS = tuple(
    f"franka_{joint}_{metric}" for joint in _FRANKA_JOINT_LABELS for metric in _FRANKA_JOINT_METRICS
)

TRACE_COLUMNS = (
    tuple(_SCALAR_COLUMNS)
    + _FRANKA_JOINT_COLUMNS
    + tuple(f"{prefix}_{label}" for prefix, labels in _VECTOR_COLUMNS.items() for label in labels)
)


_CLASSIFICATION_EXPLANATIONS = {
    "tracker_orientation_jump": (
        "The raw Unity quaternion and canonical quaternion made the same large physical rotation. "
        "The left/right-handed conversion preserved the motion, so the jump originated upstream in HTS/Quest tracking."
    ),
    "tracker_orientation_slew": (
        "No single input frame crossed the jump threshold, but the raw and canonical wrist orientations moved across "
        "a large angle during the manual-event window. The change therefore originated upstream of the adapter; "
        "compare "
        "the trace with the operator's intended physical wrist rotation."
    ),
    "persistent_tracker_flip_resynchronized": (
        "The adapter initially rejected the changed tracker orientation, but the estimate persisted long enough "
        "to trigger "
        "its resynchronization timeout. The simulated wrist can then move to the persistent flipped estimate."
    ),
    "coordinate_conversion_discontinuity": (
        "The physical rotation delta changed across the Unity-to-right-handed conversion. A basis change should "
        "preserve "
        "rotation angle, so this points to conversion or raw-sample alignment rather than normal tracking motion."
    ),
    "quaternion_representation_discontinuity": (
        "The tracker changed between q and -q, which represents the same rotation, but a later command stage jumped. "
        "This points to missing quaternion hemisphere continuity or an axis-angle wrap downstream."
    ),
    "adapter_transform_discontinuity": (
        "The canonical wrist orientation stayed continuous while the calibrated pre-safety target jumped. Inspect the "
        "anchor/orientation-offset transform in SharpaActionAdapter."
    ),
    "action_filter_discontinuity": (
        "The pre-safety target stayed continuous while the emitted arm command jumped. Inspect axis-angle "
        "conversion and "
        "SafetyFilter state/reset behavior."
    ),
    "ik_controller_or_physics_discontinuity": (
        "The emitted arm command stayed continuous while the measured simulated end effector jumped. Inspect "
        "OmniGibson's "
        "IK controller, joint state, collisions, and physics response."
    ),
    "ik_joint_space_rollover": (
        "One or more Franka joints accumulated a large rotation while the Cartesian wrist command remained "
        "comparatively continuous. This is an IK configuration-space rollover, commonly caused by an infeasible "
        "pose near a workspace boundary, a kinematic singularity, or a joint limit."
    ),
    "wrist_landmark_pairing_risk": (
        "The wrist and landmark records used for a frame arrived farther apart than the configured skew threshold. "
        "A lost or delayed record may have produced a temporally inconsistent hand sample."
    ),
    "tracker_jump_contained": (
        "The adapter's wrist gate held the preceding accepted orientation. The trace saw a suspicious tracker "
        "estimate, "
        "but that estimate was not sent directly to the arm controller."
    ),
    "unclassified_manual_event": (
        "No recorded stage crossed the configured thresholds near the manual marker. Lower the threshold or inspect "
        "the "
        "trace for a slower accumulated rotation, position jump, or controller oscillation."
    ),
}


class WristFlipRecorder:
    """Write a stage-by-stage wrist trace and produce an automatic cause report."""

    def __init__(self, output_dir: str | Path, config: WristFlipDiagnosticConfig) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.trace_path = self.output_dir / "trace.csv"
        self.events_path = self.output_dir / "events.json"
        self.summary_path = self.output_dir / "summary.md"
        self.session_path = self.output_dir / "session.json"
        self._trace_file = self.trace_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._trace_file, fieldnames=TRACE_COLUMNS)
        self._writer.writeheader()
        self._started = time.monotonic()
        self._closed = False
        self._sample_count = 0
        self._tracking_frame_count = 0
        self._missing_source_diagnostic_frames = 0
        self._segment = 0
        self._manual_lock = threading.Lock()
        self._manual_mark_pending = False
        self._event_started_at: float | None = None
        self._last_event_at = float("-inf")
        self.events: list[dict[str, Any]] = []
        self._recent_rows: deque[dict[str, Any]] = deque(maxlen=300)
        self._maxima = {
            "raw_rotation_delta_deg": 0.0,
            "canonical_rotation_delta_deg": 0.0,
            "conversion_delta_error_deg": 0.0,
            "pair_skew_ms": 0.0,
            "target_rotation_delta_deg": 0.0,
            "command_rotation_delta_deg": 0.0,
            "eef_rotation_delta_deg": 0.0,
        }
        self._max_joint_step_rotation_degrees = np.zeros(7, dtype=np.float64)
        self._max_joint_cumulative_rotation_degrees = np.zeros(7, dtype=np.float64)
        self._minimum_joint_limit_margin_degrees = np.full(7, np.inf, dtype=np.float64)
        self._previous_raw_quaternion = None
        self._previous_raw_landmarks = None
        self._previous_canonical_quaternion = None
        self._previous_canonical_position = None
        self._previous_source_timestamp_ns = None
        self._previous_receipt_timestamp = None
        self._previous_frame_timestamp = None
        self._previous_target_quaternion = None
        self._previous_target_position = None
        self._previous_command_quaternion = None
        self._previous_command_position = None
        self._previous_eef_quaternion = None
        self._previous_eef_position = None
        self._previous_arm_joint_positions = None
        self._joint_cumulative_rotation_degrees = np.zeros(7, dtype=np.float64)
        self._joint_rotation_history: deque[tuple[float, np.ndarray]] = deque()
        self._write_session("running")

    def _write_session(self, status: str) -> None:
        payload = {
            "status": status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "trace": self.trace_path.name,
            "events": self.events_path.name,
            "summary": self.summary_path.name,
        }
        self.session_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def mark_manual_event(self) -> None:
        """Mark the next captured simulator step as a visually observed flip."""

        with self._manual_lock:
            self._manual_mark_pending = True

    def reset_segment(self) -> None:
        """Start a new comparison segment after an environment reset or re-anchor."""

        self._segment += 1
        self._previous_raw_quaternion = None
        self._previous_raw_landmarks = None
        self._previous_canonical_quaternion = None
        self._previous_canonical_position = None
        self._previous_source_timestamp_ns = None
        self._previous_receipt_timestamp = None
        self._previous_frame_timestamp = None
        self._previous_target_quaternion = None
        self._previous_target_position = None
        self._previous_command_quaternion = None
        self._previous_command_position = None
        self._previous_eef_quaternion = None
        self._previous_eef_position = None
        self._previous_arm_joint_positions = None
        self._joint_cumulative_rotation_degrees = np.zeros(7, dtype=np.float64)
        self._joint_rotation_history = deque()

    @property
    def should_stop(self) -> bool:
        if self.config.continue_after_event or self._event_started_at is None:
            return False
        return time.monotonic() - self._event_started_at >= self.config.post_event_seconds

    def observe(
        self,
        *,
        step: int,
        snapshot,
        source_diagnostics,
        action_diagnostics,
        eef_position_after,
        eef_quaternion_after_xyzw,
        arm_joint_positions=None,
        arm_joint_lower_limits=None,
        arm_joint_upper_limits=None,
    ) -> dict[str, Any]:
        """Record one normal teleoperation step and return the written scalar row."""

        if self._closed:
            raise RuntimeError("WristFlipRecorder is closed")
        now = time.monotonic()
        elapsed = now - self._started
        frame = snapshot.frame
        new_tracking_frame = frame.timestamp != self._previous_frame_timestamp
        if new_tracking_frame:
            self._tracking_frame_count += 1

        raw_quaternion = None
        raw_position = None
        raw_landmarks = None
        wrist_receipt = None
        landmarks_receipt = None
        pair_skew_ms = None
        if source_diagnostics is not None:
            raw_quaternion = source_diagnostics.raw_wrist_quaternion_xyzw
            raw_position = source_diagnostics.raw_wrist_position_unity
            raw_landmarks = source_diagnostics.raw_landmarks_unity
            wrist_receipt = source_diagnostics.wrist_receipt_timestamp
            landmarks_receipt = source_diagnostics.landmarks_receipt_timestamp
            pair_skew_ms = source_diagnostics.pair_skew_seconds * 1000.0
        elif new_tracking_frame:
            self._missing_source_diagnostic_frames += 1

        raw_signed_dot = None
        raw_delta = None
        canonical_signed_dot = None
        canonical_delta = None
        conversion_error = None
        canonical_position_delta = None
        landmark_delta = None
        source_dt_ms = None
        receipt_dt_ms = None
        if new_tracking_frame:
            raw_signed_dot = _signed_dot(self._previous_raw_quaternion, raw_quaternion)
            raw_delta = _rotation_delta_degrees(self._previous_raw_quaternion, raw_quaternion)
            canonical_signed_dot = _signed_dot(
                self._previous_canonical_quaternion, frame.wrist_quaternion_xyzw
            )
            canonical_delta = _rotation_delta_degrees(
                self._previous_canonical_quaternion, frame.wrist_quaternion_xyzw
            )
            if raw_delta is not None and canonical_delta is not None:
                conversion_error = abs(raw_delta - canonical_delta)
            canonical_position_delta = _vector_delta_centimeters(
                self._previous_canonical_position, frame.wrist_position
            )
            landmark_delta = _rms_delta_centimeters(self._previous_raw_landmarks, raw_landmarks)
            if frame.source_timestamp_ns is not None and self._previous_source_timestamp_ns is not None:
                source_dt_ms = (frame.source_timestamp_ns - self._previous_source_timestamp_ns) / 1e6
            if self._previous_receipt_timestamp is not None:
                receipt_dt_ms = (frame.receipt_timestamp - self._previous_receipt_timestamp) * 1000.0

        target_quaternion = action_diagnostics.target_quaternion_xyzw
        target_position = action_diagnostics.target_position
        command_quaternion = _axis_angle_to_quaternion(action_diagnostics.filtered_axis_angle)
        command_position = action_diagnostics.filtered_position
        eef_quaternion_after = _normalized(eef_quaternion_after_xyzw)
        eef_position_after = np.asarray(eef_position_after, dtype=np.float64).reshape(3)

        target_delta = _rotation_delta_degrees(self._previous_target_quaternion, target_quaternion)
        target_position_delta = _vector_delta_centimeters(self._previous_target_position, target_position)
        command_delta = _rotation_delta_degrees(self._previous_command_quaternion, command_quaternion)
        command_position_delta = _vector_delta_centimeters(self._previous_command_position, command_position)
        eef_delta = _rotation_delta_degrees(self._previous_eef_quaternion, eef_quaternion_after)
        eef_position_delta = _vector_delta_centimeters(self._previous_eef_position, eef_position_after)

        joint_positions = self._joint_vector(arm_joint_positions, "arm_joint_positions")
        joint_lower_limits = self._joint_vector(arm_joint_lower_limits, "arm_joint_lower_limits")
        joint_upper_limits = self._joint_vector(arm_joint_upper_limits, "arm_joint_upper_limits")
        joint_step_rotation = None
        joint_window_rotation = None
        joint_lower_margin = None
        joint_upper_margin = None
        joint_limit_margin = None
        if joint_positions is not None:
            if self._previous_arm_joint_positions is None:
                joint_step_rotation = np.zeros(7, dtype=np.float64)
            else:
                joint_step_rotation = np.degrees(np.abs(joint_positions - self._previous_arm_joint_positions))
                self._joint_cumulative_rotation_degrees += joint_step_rotation
            self._joint_rotation_history.append((elapsed, self._joint_cumulative_rotation_degrees.copy()))
            window_start = elapsed - self.config.joint_rotation_window_seconds
            while len(self._joint_rotation_history) >= 2 and self._joint_rotation_history[1][0] <= window_start:
                self._joint_rotation_history.popleft()
            joint_window_rotation = (
                self._joint_cumulative_rotation_degrees - self._joint_rotation_history[0][1]
            )
            if (joint_lower_limits is None) != (joint_upper_limits is None):
                raise ValueError("Both arm joint limit arrays must be provided together")
            if joint_lower_limits is not None:
                if np.any(joint_lower_limits >= joint_upper_limits):
                    raise ValueError("Each arm joint lower limit must be below its upper limit")
                joint_lower_margin = np.degrees(joint_positions - joint_lower_limits)
                joint_upper_margin = np.degrees(joint_upper_limits - joint_positions)
                joint_limit_margin = np.minimum(joint_lower_margin, joint_upper_margin)
        elif joint_lower_limits is not None or joint_upper_limits is not None:
            raise ValueError("Arm joint positions are required when arm joint limits are provided")

        with self._manual_lock:
            manual_mark = self._manual_mark_pending
            self._manual_mark_pending = False

        triggers: list[str] = []
        threshold = self.config.flip_threshold_degrees
        if new_tracking_frame and raw_delta is not None and raw_delta >= threshold:
            triggers.append("raw_tracker_rotation_jump")
        if new_tracking_frame and canonical_delta is not None and canonical_delta >= threshold:
            triggers.append("canonical_rotation_jump")
        if (
            new_tracking_frame
            and conversion_error is not None
            and conversion_error > self.config.conversion_tolerance_degrees
        ):
            triggers.append("coordinate_conversion_mismatch")
        if (
            new_tracking_frame
            and raw_signed_dot is not None
            and raw_signed_dot < 0.0
            and raw_delta is not None
            and raw_delta <= self.config.sign_only_tolerance_degrees
        ):
            triggers.append("raw_quaternion_sign_change")
        if (
            new_tracking_frame
            and canonical_signed_dot is not None
            and canonical_signed_dot < 0.0
            and canonical_delta is not None
            and canonical_delta <= self.config.sign_only_tolerance_degrees
        ):
            triggers.append("canonical_quaternion_sign_change")
        if pair_skew_ms is not None and pair_skew_ms > self.config.pair_skew_threshold_ms:
            triggers.append("wrist_landmark_pair_skew")
        if action_diagnostics.gate_decision == "held":
            triggers.append("gate_held")
        elif action_diagnostics.gate_decision == "resynchronized":
            triggers.append("gate_resynchronized")
        if target_delta is not None and target_delta >= threshold:
            triggers.append("target_rotation_jump")
        if command_delta is not None and command_delta >= threshold:
            triggers.append("command_rotation_jump")
        if eef_delta is not None and eef_delta >= threshold:
            triggers.append("eef_rotation_jump")
        if (
            joint_window_rotation is not None
            and float(np.max(joint_window_rotation[_FRANKA_WRIST_JOINT_START:]))
            >= self.config.joint_rotation_threshold_degrees
        ):
            triggers.append("joint_space_rotation_slew")
        if (
            joint_limit_margin is not None
            and float(np.min(joint_limit_margin)) <= self.config.joint_limit_margin_threshold_degrees
        ):
            triggers.append("joint_near_limit")
        if manual_mark:
            triggers.append("manual_visual_mark")

        max_joint_step_index = None if joint_step_rotation is None else int(np.argmax(joint_step_rotation))
        max_joint_window_index = None if joint_window_rotation is None else int(np.argmax(joint_window_rotation))
        max_wrist_joint_window_index = (
            None
            if joint_window_rotation is None
            else _FRANKA_WRIST_JOINT_START + int(np.argmax(joint_window_rotation[_FRANKA_WRIST_JOINT_START:]))
        )
        min_joint_margin_index = None if joint_limit_margin is None else int(np.argmin(joint_limit_margin))

        row: dict[str, Any] = {
            "sample": self._sample_count,
            "step": step,
            "elapsed_s": elapsed,
            "segment": self._segment,
            "new_tracking_frame": int(new_tracking_frame),
            "source_diagnostics_available": int(source_diagnostics is not None),
            "frame_timestamp": frame.timestamp,
            "receipt_timestamp": frame.receipt_timestamp,
            "frame_age_ms": (now - frame.receipt_timestamp) * 1000.0,
            "source_frame_id": _number(frame.source_frame_id),
            "source_timestamp_ns": _number(frame.source_timestamp_ns),
            "source_dt_ms": _number(source_dt_ms),
            "receipt_dt_ms": _number(receipt_dt_ms),
            "wrist_receipt_timestamp": _number(wrist_receipt),
            "landmarks_receipt_timestamp": _number(landmarks_receipt),
            "pair_skew_ms": _number(pair_skew_ms),
            "raw_signed_dot": _number(raw_signed_dot),
            "raw_rotation_delta_deg": _number(raw_delta),
            "canonical_signed_dot": _number(canonical_signed_dot),
            "canonical_rotation_delta_deg": _number(canonical_delta),
            "conversion_delta_error_deg": _number(conversion_error),
            "canonical_position_delta_cm": _number(canonical_position_delta),
            "local_landmark_rms_delta_cm": _number(landmark_delta),
            "target_rotation_delta_deg": _number(target_delta),
            "target_position_delta_cm": _number(target_position_delta),
            "command_rotation_delta_deg": _number(command_delta),
            "command_position_delta_cm": _number(command_position_delta),
            "eef_rotation_delta_deg": _number(eef_delta),
            "eef_position_delta_cm": _number(eef_position_delta),
            "franka_joint_positions_available": int(joint_positions is not None),
            "franka_max_joint_step_rotation_deg": _number(
                None if max_joint_step_index is None else joint_step_rotation[max_joint_step_index]
            ),
            "franka_max_joint_step_rotation_joint": (
                "" if max_joint_step_index is None else _FRANKA_JOINT_LABELS[max_joint_step_index].upper()
            ),
            "franka_max_joint_window_rotation_deg": _number(
                None if max_joint_window_index is None else joint_window_rotation[max_joint_window_index]
            ),
            "franka_max_joint_window_rotation_joint": (
                "" if max_joint_window_index is None else _FRANKA_JOINT_LABELS[max_joint_window_index].upper()
            ),
            "franka_max_wrist_joint_window_rotation_deg": _number(
                None
                if max_wrist_joint_window_index is None
                else joint_window_rotation[max_wrist_joint_window_index]
            ),
            "franka_max_wrist_joint_window_rotation_joint": (
                ""
                if max_wrist_joint_window_index is None
                else _FRANKA_JOINT_LABELS[max_wrist_joint_window_index].upper()
            ),
            "franka_total_cumulative_rotation_deg": _number(
                None if joint_positions is None else float(np.sum(self._joint_cumulative_rotation_degrees))
            ),
            "franka_min_joint_limit_margin_deg": _number(
                None if min_joint_margin_index is None else joint_limit_margin[min_joint_margin_index]
            ),
            "franka_min_joint_limit_margin_joint": (
                "" if min_joint_margin_index is None else _FRANKA_JOINT_LABELS[min_joint_margin_index].upper()
            ),
            "gate_decision": action_diagnostics.gate_decision,
            "gate_reason": action_diagnostics.gate_reason or "",
            "hemisphere_corrected": int(action_diagnostics.hemisphere_corrected),
            "manual_mark": int(manual_mark),
            "triggers": "|".join(triggers),
        }
        for index, joint in enumerate(_FRANKA_JOINT_LABELS):
            values = {
                "position_rad": None if joint_positions is None else joint_positions[index],
                "lower_limit_rad": None if joint_lower_limits is None else joint_lower_limits[index],
                "upper_limit_rad": None if joint_upper_limits is None else joint_upper_limits[index],
                "lower_margin_deg": None if joint_lower_margin is None else joint_lower_margin[index],
                "upper_margin_deg": None if joint_upper_margin is None else joint_upper_margin[index],
                "limit_margin_deg": None if joint_limit_margin is None else joint_limit_margin[index],
                "step_rotation_deg": None if joint_step_rotation is None else joint_step_rotation[index],
                "cumulative_rotation_deg": (
                    None if joint_positions is None else self._joint_cumulative_rotation_degrees[index]
                ),
                "window_rotation_deg": None if joint_window_rotation is None else joint_window_rotation[index],
            }
            for metric, value in values.items():
                row[f"franka_{joint}_{metric}"] = _number(value)
        _put_vector(row, "raw_position_unity", raw_position, "xyz")
        _put_vector(row, "raw_quaternion_unity", raw_quaternion, "xyzw")
        _put_vector(row, "canonical_position", frame.wrist_position, "xyz")
        _put_vector(row, "canonical_quaternion", frame.wrist_quaternion_xyzw, "xyzw")
        _put_vector(row, "gated_quaternion", action_diagnostics.gated_quaternion_xyzw, "xyzw")
        _put_vector(row, "target_position", target_position, "xyz")
        _put_vector(row, "target_quaternion", target_quaternion, "xyzw")
        _put_vector(row, "target_axis_angle", action_diagnostics.target_axis_angle, "xyz")
        _put_vector(row, "command_position", command_position, "xyz")
        _put_vector(row, "command_axis_angle", action_diagnostics.filtered_axis_angle, "xyz")
        _put_vector(row, "eef_position_before", action_diagnostics.eef_position_before, "xyz")
        _put_vector(row, "eef_quaternion_before", action_diagnostics.eef_quaternion_before_xyzw, "xyzw")
        _put_vector(row, "eef_position_after", eef_position_after, "xyz")
        _put_vector(row, "eef_quaternion_after", eef_quaternion_after, "xyzw")
        self._writer.writerow(row)
        self._trace_file.flush()
        self._recent_rows.append(row.copy())

        for metric in self._maxima:
            value = row[metric]
            if value != "":
                self._maxima[metric] = max(self._maxima[metric], float(value))
        if joint_positions is not None:
            self._max_joint_step_rotation_degrees = np.maximum(
                self._max_joint_step_rotation_degrees, joint_step_rotation
            )
            self._max_joint_cumulative_rotation_degrees = np.maximum(
                self._max_joint_cumulative_rotation_degrees, self._joint_cumulative_rotation_degrees
            )
        if joint_limit_margin is not None:
            self._minimum_joint_limit_margin_degrees = np.minimum(
                self._minimum_joint_limit_margin_degrees, joint_limit_margin
            )

        event_triggers = {
            "raw_tracker_rotation_jump",
            "canonical_rotation_jump",
            "coordinate_conversion_mismatch",
            "gate_held",
            "gate_resynchronized",
            "target_rotation_jump",
            "command_rotation_jump",
            "eef_rotation_jump",
            "joint_space_rotation_slew",
            "manual_visual_mark",
        }.intersection(triggers)
        manually_confirms_recent = manual_mark and self.events and elapsed - self._last_event_at <= 2.0
        if manually_confirms_recent:
            event = self.events[-1]
            event["manual"] = True
            if "manual_visual_mark" not in event["triggers"]:
                event["triggers"].append("manual_visual_mark")
            print(f"[wrist diagnostic] manual marker attached to event {event['event']}", flush=True)

        can_open_event = self.config.continue_after_event or self._event_started_at is None or manual_mark
        if (
            event_triggers
            and not manually_confirms_recent
            and can_open_event
            and (manual_mark or elapsed - self._last_event_at >= 1.0)
        ):
            classification, evidence = self._classify(row, triggers)
            event = {
                "event": len(self.events) + 1,
                "sample": self._sample_count,
                "step": step,
                "elapsed_s": round(elapsed, 6),
                "manual": manual_mark,
                "classification": classification,
                "explanation": _CLASSIFICATION_EXPLANATIONS[classification],
                "evidence": evidence,
                "triggers": triggers,
            }
            self.events.append(event)
            self._last_event_at = elapsed
            if self._event_started_at is None:
                self._event_started_at = now
            print(
                f"[wrist diagnostic] event {event['event']}: {classification}; {evidence}",
                flush=True,
            )

        if new_tracking_frame:
            self._previous_raw_quaternion = None if raw_quaternion is None else np.asarray(raw_quaternion).copy()
            self._previous_raw_landmarks = None if raw_landmarks is None else np.asarray(raw_landmarks).copy()
            self._previous_canonical_quaternion = frame.wrist_quaternion_xyzw.copy()
            self._previous_canonical_position = frame.wrist_position.copy()
            self._previous_source_timestamp_ns = frame.source_timestamp_ns
            self._previous_receipt_timestamp = frame.receipt_timestamp
            self._previous_frame_timestamp = frame.timestamp
        self._previous_target_quaternion = np.asarray(target_quaternion).copy()
        self._previous_target_position = np.asarray(target_position).copy()
        self._previous_command_quaternion = command_quaternion.copy()
        self._previous_command_position = np.asarray(command_position).copy()
        self._previous_eef_quaternion = eef_quaternion_after.copy()
        self._previous_eef_position = eef_position_after.copy()
        self._previous_arm_joint_positions = None if joint_positions is None else joint_positions.copy()
        self._sample_count += 1
        return row

    def _classify(self, row: dict[str, Any], triggers: list[str]) -> tuple[str, str]:
        if "manual_visual_mark" in triggers:
            window = [
                candidate
                for candidate in self._recent_rows
                if candidate["segment"] == row["segment"]
                and float(row["elapsed_s"]) - float(candidate["elapsed_s"]) <= 2.0
            ]
            window_triggers = {
                trigger
                for candidate in window
                for trigger in str(candidate["triggers"]).split("|")
                if trigger
            }
            triggers = list(set(triggers).union(window_triggers))
            row = self._window_peak_row(row, window)

        raw = 0.0 if row["raw_rotation_delta_deg"] == "" else float(row["raw_rotation_delta_deg"])
        canonical = (
            0.0 if row["canonical_rotation_delta_deg"] == "" else float(row["canonical_rotation_delta_deg"])
        )
        target = 0.0 if row["target_rotation_delta_deg"] == "" else float(row["target_rotation_delta_deg"])
        command = 0.0 if row["command_rotation_delta_deg"] == "" else float(row["command_rotation_delta_deg"])
        eef = 0.0 if row["eef_rotation_delta_deg"] == "" else float(row["eef_rotation_delta_deg"])
        skew = 0.0 if row["pair_skew_ms"] == "" else float(row["pair_skew_ms"])
        threshold = self.config.flip_threshold_degrees

        if "gate_resynchronized" in triggers:
            return (
                "persistent_tracker_flip_resynchronized",
                f"gate=resynchronized, raw={raw:.1f}°, canonical={canonical:.1f}°, target={target:.1f}°",
            )
        if "coordinate_conversion_mismatch" in triggers:
            error = float(row["conversion_delta_error_deg"])
            return (
                "coordinate_conversion_discontinuity",
                f"raw={raw:.1f}°, canonical={canonical:.1f}°, delta error={error:.2f}°",
            )
        sign_change = {
            "raw_quaternion_sign_change",
            "canonical_quaternion_sign_change",
        }.intersection(triggers)
        if sign_change and (target >= threshold or command >= threshold):
            return (
                "quaternion_representation_discontinuity",
                f"physical input delta={canonical:.2f}°, target={target:.1f}°, command={command:.1f}°",
            )
        if raw >= threshold and canonical >= threshold:
            classification = "tracker_jump_contained" if "gate_held" in triggers else "tracker_orientation_jump"
            return (
                classification,
                f"raw={raw:.1f}°, canonical={canonical:.1f}°, gate={row['gate_decision']}, pair skew={skew:.1f} ms",
            )
        if canonical < threshold and target >= threshold:
            return "adapter_transform_discontinuity", f"canonical={canonical:.1f}°, target={target:.1f}°"
        if target < threshold and command >= threshold:
            return "action_filter_discontinuity", f"target={target:.1f}°, command={command:.1f}°"
        if "joint_space_rotation_slew" in triggers:
            joint = str(row["franka_max_wrist_joint_window_rotation_joint"])
            rotation = float(row["franka_max_wrist_joint_window_rotation_deg"])
            margin = row["franka_min_joint_limit_margin_deg"]
            margin_joint = str(row["franka_min_joint_limit_margin_joint"])
            margin_evidence = (
                "limit margin unavailable"
                if margin == ""
                else f"nearest limit={float(margin):.1f}° at {margin_joint}"
            )
            return (
                "ik_joint_space_rollover",
                f"{joint} accumulated {rotation:.1f}° in "
                f"{self.config.joint_rotation_window_seconds:g} s; {margin_evidence}",
            )
        if command < threshold and eef >= threshold:
            return "ik_controller_or_physics_discontinuity", f"command={command:.1f}°, measured EEF={eef:.1f}°"
        if "gate_held" in triggers:
            return "tracker_jump_contained", f"gate held input: {row['gate_reason']}"
        if skew > self.config.pair_skew_threshold_ms:
            return "wrist_landmark_pairing_risk", f"pair skew={skew:.1f} ms"
        if "manual_visual_mark" in triggers:
            window = [
                candidate
                for candidate in self._recent_rows
                if candidate["segment"] == row["segment"]
                and float(row["elapsed_s"]) - float(candidate["elapsed_s"]) <= 2.0
            ]
            if len(window) >= 2:
                raw_span = _rotation_delta_degrees(
                    self._row_quaternion(window[0], "raw_quaternion_unity"),
                    self._row_quaternion(window[-1], "raw_quaternion_unity"),
                )
                canonical_span = _rotation_delta_degrees(
                    self._row_quaternion(window[0], "canonical_quaternion"),
                    self._row_quaternion(window[-1], "canonical_quaternion"),
                )
                if (
                    raw_span is not None
                    and canonical_span is not None
                    and raw_span >= threshold
                    and canonical_span >= threshold
                ):
                    return (
                        "tracker_orientation_slew",
                        f"2 s window raw span={raw_span:.1f}°, canonical span={canonical_span:.1f}°",
                    )
        return (
            "unclassified_manual_event",
            f"raw={raw:.1f}°, canonical={canonical:.1f}°, target={target:.1f}°, "
            f"command={command:.1f}°, EEF={eef:.1f}°",
        )

    @staticmethod
    def _window_peak_row(current: dict[str, Any], window: list[dict[str, Any]]) -> dict[str, Any]:
        peak = current.copy()
        metrics = (
            "raw_rotation_delta_deg",
            "canonical_rotation_delta_deg",
            "conversion_delta_error_deg",
            "pair_skew_ms",
            "target_rotation_delta_deg",
            "command_rotation_delta_deg",
            "eef_rotation_delta_deg",
        )
        for metric in metrics:
            values = [float(candidate[metric]) for candidate in window if candidate[metric] != ""]
            peak[metric] = max(values) if values else ""
        joint_rows = [candidate for candidate in window if candidate["franka_max_joint_window_rotation_deg"] != ""]
        if joint_rows:
            joint_peak = max(joint_rows, key=lambda candidate: float(candidate["franka_max_joint_window_rotation_deg"]))
            peak["franka_max_joint_window_rotation_deg"] = joint_peak["franka_max_joint_window_rotation_deg"]
            peak["franka_max_joint_window_rotation_joint"] = joint_peak["franka_max_joint_window_rotation_joint"]
        wrist_joint_rows = [
            candidate for candidate in window if candidate["franka_max_wrist_joint_window_rotation_deg"] != ""
        ]
        if wrist_joint_rows:
            wrist_joint_peak = max(
                wrist_joint_rows,
                key=lambda candidate: float(candidate["franka_max_wrist_joint_window_rotation_deg"]),
            )
            peak["franka_max_wrist_joint_window_rotation_deg"] = wrist_joint_peak[
                "franka_max_wrist_joint_window_rotation_deg"
            ]
            peak["franka_max_wrist_joint_window_rotation_joint"] = wrist_joint_peak[
                "franka_max_wrist_joint_window_rotation_joint"
            ]
        margin_rows = [candidate for candidate in window if candidate["franka_min_joint_limit_margin_deg"] != ""]
        if margin_rows:
            margin_peak = min(margin_rows, key=lambda candidate: float(candidate["franka_min_joint_limit_margin_deg"]))
            peak["franka_min_joint_limit_margin_deg"] = margin_peak["franka_min_joint_limit_margin_deg"]
            peak["franka_min_joint_limit_margin_joint"] = margin_peak["franka_min_joint_limit_margin_joint"]
        return peak

    @staticmethod
    def _joint_vector(value: Any | None, name: str) -> np.ndarray | None:
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (7,):
            raise ValueError(f"{name} must contain exactly 7 Franka arm joints")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain only finite values")
        return vector

    @staticmethod
    def _row_quaternion(row: dict[str, Any], prefix: str) -> np.ndarray | None:
        values = [row[f"{prefix}_{label}"] for label in "xyzw"]
        if any(value == "" for value in values):
            return None
        return np.asarray(values, dtype=np.float64)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._trace_file.flush()
        self._trace_file.close()
        self.events_path.write_text(json.dumps(self.events, indent=2) + "\n", encoding="utf-8")
        self._write_summary()
        self._write_session("complete")

    def _write_summary(self) -> None:
        lines = [
            "# Wrist flip diagnostic summary",
            "",
            f"- Samples: {self._sample_count}",
            f"- Unique tracking frames: {self._tracking_frame_count}",
            f"- Frames missing raw HTS diagnostics: {self._missing_source_diagnostic_frames}",
            f"- Detected or marked events: {len(self.events)}",
            f"- Tracking-to-world yaw: {self.config.tracking_yaw_degrees:g}°",
            f"- Trace: `{self.trace_path.name}`",
            "",
            "## Largest observed deltas",
            "",
            "| Stage | Maximum |",
            "|---|---:|",
            f"| Raw Unity wrist | {self._maxima['raw_rotation_delta_deg']:.2f}° |",
            f"| Canonical HandFrame wrist | {self._maxima['canonical_rotation_delta_deg']:.2f}° |",
            f"| Conversion delta error | {self._maxima['conversion_delta_error_deg']:.3f}° |",
            f"| Wrist/landmark receive skew | {self._maxima['pair_skew_ms']:.2f} ms |",
            f"| Pre-safety target | {self._maxima['target_rotation_delta_deg']:.2f}° |",
            f"| Post-safety command | {self._maxima['command_rotation_delta_deg']:.2f}° |",
            f"| Measured end effector | {self._maxima['eef_rotation_delta_deg']:.2f}° |",
            f"| Franka joint, single step | {float(np.max(self._max_joint_step_rotation_degrees)):.2f}° |",
            "",
            "## Franka joint motion and limits",
            "",
            (
                f"Cumulative rotation is the sum of absolute measured joint changes within one anchor/reset segment. "
                f"The J5-J7 rollover detector uses a {self.config.joint_rotation_window_seconds:g}-second window "
                f"and a {self.config.joint_rotation_threshold_degrees:g}° threshold."
            ),
            "",
            "| Joint | Maximum step | Maximum segment cumulative | Minimum limit margin |",
            "|---|---:|---:|---:|",
            *[
                (
                    f"| J{index + 1} | {self._max_joint_step_rotation_degrees[index]:.2f}° | "
                    f"{self._max_joint_cumulative_rotation_degrees[index]:.2f}° | "
                    + (
                        "n/a |"
                        if not math.isfinite(self._minimum_joint_limit_margin_degrees[index])
                        else f"{self._minimum_joint_limit_margin_degrees[index]:.2f}° |"
                    )
                )
                for index in range(7)
            ],
            "",
            "## Events",
            "",
        ]
        if not self.events:
            lines.extend(
                [
                    "No automatic threshold crossing or manual marker was recorded.",
                    "",
                    "If a flip was visible, run again and press **F** immediately when it occurs. A slow slew can "
                    "look like "
                    "a flip without exceeding the per-frame threshold.",
                ]
            )
        else:
            lines.extend(["| Event | Time | Classification | Evidence |", "|---:|---:|---|---|"])
            for event in self.events:
                evidence = str(event["evidence"]).replace("|", "/")
                lines.append(
                    f"| {event['event']} | {event['elapsed_s']:.3f}s | `{event['classification']}` | {evidence} |"
                )
            lines.append("")
            for event in self.events:
                lines.extend(
                    [
                        f"### Event {event['event']}: `{event['classification']}`",
                        "",
                        event["explanation"],
                        "",
                        f"Evidence: {event['evidence']}",
                        "",
                    ]
                )
        self.summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

"""Offline hand-eye calibration for a lighthouse VIVE wrist tracker.

The estimator consumes synchronized pose pairs and solves

``reference_T_wrist[i] = reference_T_lighthouse * lighthouse_T_tracker[i] * tracker_T_wrist``.

It does not open a VIVE, Quest, OpenXR, or OmniGibson device.  Capture tools
can therefore write the JSON input on one machine and run calibration later.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dex_teleop.tracking.vive import (
    RigidTransform,
    ViveCalibration,
    ViveTrackerMount,
    VIVE_TRACKER_POSE_FRAME,
    compose_transforms,
)
from dex_teleop.types import Handedness


VIVE_CALIBRATION_CAPTURE_SCHEMA_VERSION = 1
MINIMUM_CALIBRATION_POSES = 6

VIVE_CALIBRATION_INPUT_SCHEMA = """input.json schema (all quaternions are xyzw):
{
  "schema_version": 1,
  "tracker_pose_frame": "dex_teleop_lighthouse_rh_z_up",
  "reference_frame": "quest_tracking",
  "tracker_serial": "LHR-01234567",
  "handedness": "right",
  "samples": [
    {
      "reference_timestamp": 12.345,
      "tracker_timestamp": 12.350,
      "reference_T_wrist": {
        "translation": [0.1, 0.2, 0.3],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      },
      "lighthouse_T_tracker": {
        "translation": [0.4, 0.5, 0.6],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    }
  ]
}

At least six time-paired poses are required. Move and rotate the wrist about
multiple axes. The output is directly loadable by ViveCalibration/ViveWristSource.
Translations are meters. Timestamps are seconds in one shared clock domain.
`lighthouse_T_tracker` is in dex_teleop's converted lighthouse basis: for raw
libsurvive Pose() arrays use position [x, -z, y] and quaternion [x, -z, y, w].
"""


def _inverse_transform(transform: RigidTransform) -> RigidTransform:
    rotation = Rotation.from_quat(transform.quaternion_xyzw)
    inverse_rotation = rotation.inv()
    return RigidTransform(
        inverse_rotation.apply(-transform.translation),
        inverse_rotation.as_quat(),
    )


def _relative_transform(
    first: RigidTransform, second: RigidTransform
) -> RigidTransform:
    return compose_transforms(_inverse_transform(first), second)


@dataclass(frozen=True)
class ViveCalibrationPair:
    """One time-paired reference wrist and lighthouse tracker observation."""

    reference_timestamp: float
    tracker_timestamp: float
    reference_from_wrist: RigidTransform
    lighthouse_from_tracker: RigidTransform

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_timestamp) or not math.isfinite(
            self.tracker_timestamp
        ):
            raise ValueError("Calibration timestamps must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping, *, field: str) -> "ViveCalibrationPair":
        try:
            return cls(
                reference_timestamp=float(value["reference_timestamp"]),
                tracker_timestamp=float(value["tracker_timestamp"]),
                reference_from_wrist=RigidTransform.from_mapping(
                    value["reference_T_wrist"], field=f"{field}.reference_T_wrist"
                ),
                lighthouse_from_tracker=RigidTransform.from_mapping(
                    value["lighthouse_T_tracker"], field=f"{field}.lighthouse_T_tracker"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid calibration pair {field}: {error}") from error


@dataclass(frozen=True)
class ViveCalibrationCapture:
    """Validated calibration-capture document."""

    reference_frame: str
    tracker_serial: str
    handedness: Handedness
    pairs: tuple[ViveCalibrationPair, ...]
    tracker_pose_frame: str = VIVE_TRACKER_POSE_FRAME

    def __post_init__(self) -> None:
        if not self.reference_frame or not self.reference_frame.strip():
            raise ValueError("Calibration reference_frame must be non-empty")
        if not self.tracker_serial or not self.tracker_serial.strip():
            raise ValueError("Calibration tracker_serial must be non-empty")
        object.__setattr__(self, "handedness", Handedness(self.handedness))
        object.__setattr__(self, "pairs", tuple(self.pairs))
        if self.tracker_pose_frame != VIVE_TRACKER_POSE_FRAME:
            raise ValueError(
                "Calibration tracker_pose_frame must be "
                f"{VIVE_TRACKER_POSE_FRAME!r}; convert raw libsurvive Pose() arrays first"
            )
        if len(self.pairs) < MINIMUM_CALIBRATION_POSES:
            raise ValueError(
                f"VIVE calibration requires at least {MINIMUM_CALIBRATION_POSES} paired poses; "
                f"received {len(self.pairs)}"
            )

    @classmethod
    def load(cls, path: str | Path) -> "ViveCalibrationCapture":
        path = Path(path).expanduser()
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, Mapping):
            raise ValueError("VIVE calibration capture must be a JSON object")
        if document.get("schema_version") != VIVE_CALIBRATION_CAPTURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported VIVE calibration capture schema {document.get('schema_version')!r}; "
                f"expected {VIVE_CALIBRATION_CAPTURE_SCHEMA_VERSION}"
            )
        raw_pairs = document.get("samples")
        if not isinstance(raw_pairs, list):
            raise ValueError("VIVE calibration capture samples must be a list")
        try:
            return cls(
                reference_frame=str(document["reference_frame"]),
                tracker_serial=str(document["tracker_serial"]),
                handedness=Handedness(document["handedness"]),
                pairs=tuple(
                    ViveCalibrationPair.from_mapping(value, field=f"samples[{index}]")
                    for index, value in enumerate(raw_pairs)
                ),
                tracker_pose_frame=str(document["tracker_pose_frame"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid VIVE calibration capture: {error}") from error


@dataclass(frozen=True)
class ViveCalibrationReport:
    """Physical residuals and numerical diagnostics for an estimate."""

    sample_count: int
    translation_rmse_m: float
    translation_max_m: float
    rotation_rmse_deg: float
    rotation_max_deg: float
    timestamp_skew_rms_ms: float
    timestamp_skew_max_ms: float
    tracker_translation_span_m: float
    tracker_rotation_span_deg: float
    motion_condition_number: float
    optimizer_condition_number: float
    optimizer_cost: float
    optimizer_evaluations: int

    def as_mapping(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "translation_rmse_m": self.translation_rmse_m,
            "translation_max_m": self.translation_max_m,
            "rotation_rmse_deg": self.rotation_rmse_deg,
            "rotation_max_deg": self.rotation_max_deg,
            "timestamp_skew_rms_ms": self.timestamp_skew_rms_ms,
            "timestamp_skew_max_ms": self.timestamp_skew_max_ms,
            "tracker_translation_span_m": self.tracker_translation_span_m,
            "tracker_rotation_span_deg": self.tracker_rotation_span_deg,
            "motion_condition_number": self.motion_condition_number,
            "optimizer_condition_number": self.optimizer_condition_number,
            "optimizer_cost": self.optimizer_cost,
            "optimizer_evaluations": self.optimizer_evaluations,
        }


@dataclass(frozen=True)
class ViveCalibrationEstimate:
    calibration: ViveCalibration
    report: ViveCalibrationReport


@dataclass(frozen=True)
class _RelativeMotions:
    tracker: tuple[RigidTransform, ...]
    wrist: tuple[RigidTransform, ...]
    rotation_system: np.ndarray
    translation_system: np.ndarray
    condition_number: float
    rotation_span_radians: float


def _relative_motions(capture: ViveCalibrationCapture) -> _RelativeMotions:
    tracker_relative: list[RigidTransform] = []
    wrist_relative: list[RigidTransform] = []
    rotation_system: list[np.ndarray] = []
    translation_system: list[np.ndarray] = []
    rotation_axes: list[np.ndarray] = []
    rotation_angles: list[float] = []

    for first_index in range(len(capture.pairs)):
        first = capture.pairs[first_index]
        for second in capture.pairs[first_index + 1 :]:
            tracker_motion = _relative_transform(
                first.lighthouse_from_tracker, second.lighthouse_from_tracker
            )
            wrist_motion = _relative_transform(
                first.reference_from_wrist, second.reference_from_wrist
            )
            tracker_rotation = Rotation.from_quat(tracker_motion.quaternion_xyzw)
            wrist_rotation = Rotation.from_quat(wrist_motion.quaternion_xyzw)
            tracker_matrix = tracker_rotation.as_matrix()
            wrist_matrix = wrist_rotation.as_matrix()
            tracker_relative.append(tracker_motion)
            wrist_relative.append(wrist_motion)
            rotation_system.append(
                np.kron(np.eye(3), tracker_matrix) - np.kron(wrist_matrix.T, np.eye(3))
            )
            translation_system.append(tracker_matrix - np.eye(3))
            rotation_vector = tracker_rotation.as_rotvec()
            angle = float(np.linalg.norm(rotation_vector))
            rotation_angles.append(angle)
            if angle >= math.radians(5.0):
                rotation_axes.append(rotation_vector / angle)

    if len(rotation_axes) < 3:
        raise ValueError(
            "Calibration motion is degenerate: capture at least three poses separated by 5 degrees"
        )
    axis_singular_values = np.linalg.svd(np.asarray(rotation_axes), compute_uv=False)
    if axis_singular_values[1] < 0.25:
        raise ValueError(
            "Calibration motion is degenerate: rotate the wrist about at least two non-parallel axes"
        )
    rotation_span = max(rotation_angles, default=0.0)
    if rotation_span < math.radians(15.0):
        raise ValueError("Calibration rotation span must be at least 15 degrees")

    rotation_matrix = np.vstack(rotation_system)
    rotation_singular_values = np.linalg.svd(rotation_matrix, compute_uv=False)
    # A well-constrained AX=XB system has one zero singular value.  The
    # penultimate value measures the weakest observable direction.
    observable_rotation = float(rotation_singular_values[-2])
    if observable_rotation <= np.finfo(np.float64).eps:
        raise ValueError("Calibration rotation system is rank deficient")
    rotation_condition = float(rotation_singular_values[0] / observable_rotation)

    translation_matrix = np.vstack(translation_system)
    translation_singular_values = np.linalg.svd(translation_matrix, compute_uv=False)
    if translation_singular_values[-1] <= 1e-6:
        raise ValueError(
            "Calibration translation system is rank deficient; rotate the wrist about more axes"
        )
    translation_condition = float(
        translation_singular_values[0] / translation_singular_values[-1]
    )
    condition = max(rotation_condition, translation_condition)
    if not math.isfinite(condition) or condition > 1e4:
        raise ValueError(
            f"Calibration motion is poorly conditioned (condition number {condition:.3g})"
        )

    return _RelativeMotions(
        tracker=tuple(tracker_relative),
        wrist=tuple(wrist_relative),
        rotation_system=rotation_matrix,
        translation_system=translation_matrix,
        condition_number=condition,
        rotation_span_radians=rotation_span,
    )


def _initial_estimate(
    capture: ViveCalibrationCapture, relative: _RelativeMotions
) -> tuple[RigidTransform, RigidTransform]:
    _, _, vectors = np.linalg.svd(relative.rotation_system)
    tracker_from_wrist_matrix = vectors[-1].reshape((3, 3), order="F")
    # The homogeneous SVD vector has arbitrary sign.  A true rotation has a
    # positive determinant before projection.
    if np.linalg.det(tracker_from_wrist_matrix) < 0.0:
        tracker_from_wrist_matrix *= -1.0
    left, _, right = np.linalg.svd(tracker_from_wrist_matrix)
    tracker_from_wrist_matrix = left @ right
    if np.linalg.det(tracker_from_wrist_matrix) < 0.0:
        left[:, -1] *= -1.0
        tracker_from_wrist_matrix = left @ right
    tracker_from_wrist_rotation = Rotation.from_matrix(tracker_from_wrist_matrix)

    translation_rhs = []
    for tracker_motion, wrist_motion in zip(relative.tracker, relative.wrist):
        translation_rhs.append(
            tracker_from_wrist_rotation.apply(wrist_motion.translation)
            - tracker_motion.translation
        )
    tracker_from_wrist_translation, _, _, _ = np.linalg.lstsq(
        relative.translation_system, np.concatenate(translation_rhs), rcond=None
    )
    tracker_from_wrist = RigidTransform(
        tracker_from_wrist_translation, tracker_from_wrist_rotation.as_quat()
    )

    reference_from_lighthouse_candidates: list[RigidTransform] = []
    for pair in capture.pairs:
        lighthouse_from_wrist = compose_transforms(
            pair.lighthouse_from_tracker, tracker_from_wrist
        )
        reference_from_lighthouse_candidates.append(
            compose_transforms(
                pair.reference_from_wrist, _inverse_transform(lighthouse_from_wrist)
            )
        )
    candidate_rotations = Rotation.from_quat(
        np.asarray(
            [
                candidate.quaternion_xyzw
                for candidate in reference_from_lighthouse_candidates
            ]
        )
    )
    reference_from_lighthouse = RigidTransform(
        np.median(
            np.asarray(
                [
                    candidate.translation
                    for candidate in reference_from_lighthouse_candidates
                ]
            ),
            axis=0,
        ),
        candidate_rotations.mean().as_quat(),
    )
    return reference_from_lighthouse, tracker_from_wrist


def _pack(
    reference_from_lighthouse: RigidTransform, tracker_from_wrist: RigidTransform
) -> np.ndarray:
    return np.concatenate(
        (
            reference_from_lighthouse.translation,
            Rotation.from_quat(reference_from_lighthouse.quaternion_xyzw).as_rotvec(),
            tracker_from_wrist.translation,
            Rotation.from_quat(tracker_from_wrist.quaternion_xyzw).as_rotvec(),
        )
    )


def _unpack(parameters: np.ndarray) -> tuple[RigidTransform, RigidTransform]:
    return (
        RigidTransform(
            parameters[0:3], Rotation.from_rotvec(parameters[3:6]).as_quat()
        ),
        RigidTransform(
            parameters[6:9], Rotation.from_rotvec(parameters[9:12]).as_quat()
        ),
    )


def _pose_errors(
    capture: ViveCalibrationCapture,
    reference_from_lighthouse: RigidTransform,
    tracker_from_wrist: RigidTransform,
) -> tuple[np.ndarray, np.ndarray]:
    translations = []
    rotations = []
    for pair in capture.pairs:
        prediction = compose_transforms(
            compose_transforms(reference_from_lighthouse, pair.lighthouse_from_tracker),
            tracker_from_wrist,
        )
        translations.append(
            prediction.translation - pair.reference_from_wrist.translation
        )
        expected_rotation = Rotation.from_quat(
            pair.reference_from_wrist.quaternion_xyzw
        )
        predicted_rotation = Rotation.from_quat(prediction.quaternion_xyzw)
        rotations.append((expected_rotation.inv() * predicted_rotation).as_rotvec())
    return np.asarray(translations), np.asarray(rotations)


def estimate_vive_calibration(
    capture: ViveCalibrationCapture,
    *,
    max_time_skew_seconds: float = 0.020,
    translation_scale_m: float = 0.010,
    rotation_scale_degrees: float = 1.0,
    robust_loss: str = "soft_l1",
) -> ViveCalibrationEstimate:
    """Estimate both constant transforms and return a loadable calibration.

    ``translation_scale_m`` and ``rotation_scale_degrees`` specify the noise
    scale used to balance position and orientation residuals.  They do not
    constrain the resulting transform.
    """

    if max_time_skew_seconds < 0.0 or not math.isfinite(max_time_skew_seconds):
        raise ValueError("max_time_skew_seconds must be finite and non-negative")
    if translation_scale_m <= 0.0 or not math.isfinite(translation_scale_m):
        raise ValueError("translation_scale_m must be finite and positive")
    if rotation_scale_degrees <= 0.0 or not math.isfinite(rotation_scale_degrees):
        raise ValueError("rotation_scale_degrees must be finite and positive")
    supported_losses = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
    if robust_loss not in supported_losses:
        raise ValueError(f"robust_loss must be one of {sorted(supported_losses)}")

    reference_timestamps = np.asarray(
        [pair.reference_timestamp for pair in capture.pairs]
    )
    tracker_timestamps = np.asarray([pair.tracker_timestamp for pair in capture.pairs])
    if np.any(np.diff(reference_timestamps) <= 0.0) or np.any(
        np.diff(tracker_timestamps) <= 0.0
    ):
        raise ValueError(
            "Calibration timestamps must be strictly increasing in both streams"
        )
    timestamp_skews = np.abs(reference_timestamps - tracker_timestamps)
    worst_skew = float(np.max(timestamp_skews))
    if worst_skew > max_time_skew_seconds:
        raise ValueError(
            f"Calibration pair time skew {worst_skew * 1000.0:.3f} ms exceeds "
            f"the {max_time_skew_seconds * 1000.0:.3f} ms limit"
        )

    relative = _relative_motions(capture)
    initial_reference, initial_mount = _initial_estimate(capture, relative)
    rotation_scale_radians = math.radians(rotation_scale_degrees)

    def residual(parameters: np.ndarray) -> np.ndarray:
        reference_from_lighthouse, tracker_from_wrist = _unpack(parameters)
        translations, rotations = _pose_errors(
            capture, reference_from_lighthouse, tracker_from_wrist
        )
        return np.concatenate(
            (
                translations.reshape(-1) / translation_scale_m,
                rotations.reshape(-1) / rotation_scale_radians,
            )
        )

    solution = least_squares(
        residual,
        _pack(initial_reference, initial_mount),
        method="trf",
        loss=robust_loss,
        f_scale=1.0,
        max_nfev=2000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(f"VIVE calibration optimizer failed: {solution.message}")
    jacobian_singular_values = np.linalg.svd(solution.jac, compute_uv=False)
    if jacobian_singular_values[-1] <= np.finfo(np.float64).eps:
        raise ValueError("Optimized calibration is numerically rank deficient")
    optimizer_condition = float(
        jacobian_singular_values[0] / jacobian_singular_values[-1]
    )
    if not math.isfinite(optimizer_condition) or optimizer_condition > 1e8:
        raise ValueError(
            f"Optimized calibration is poorly conditioned (condition number {optimizer_condition:.3g})"
        )

    reference_from_lighthouse, tracker_from_wrist = _unpack(solution.x)
    translations, rotations = _pose_errors(
        capture, reference_from_lighthouse, tracker_from_wrist
    )
    translation_norms = np.linalg.norm(translations, axis=1)
    rotation_norms = np.linalg.norm(rotations, axis=1)
    tracker_positions = np.asarray(
        [pair.lighthouse_from_tracker.translation for pair in capture.pairs]
    )
    tracker_span = max(
        float(np.linalg.norm(first - second))
        for index, first in enumerate(tracker_positions)
        for second in tracker_positions[index + 1 :]
    )

    calibration = ViveCalibration(
        reference_frame=capture.reference_frame,
        reference_from_lighthouse=reference_from_lighthouse,
        mounts=(
            ViveTrackerMount(
                serial=capture.tracker_serial,
                handedness=capture.handedness,
                tracker_to_wrist=tracker_from_wrist,
            ),
        ),
    )
    report = ViveCalibrationReport(
        sample_count=len(capture.pairs),
        translation_rmse_m=float(np.sqrt(np.mean(translation_norms**2))),
        translation_max_m=float(np.max(translation_norms)),
        rotation_rmse_deg=math.degrees(float(np.sqrt(np.mean(rotation_norms**2)))),
        rotation_max_deg=math.degrees(float(np.max(rotation_norms))),
        timestamp_skew_rms_ms=float(np.sqrt(np.mean(timestamp_skews**2)) * 1000.0),
        timestamp_skew_max_ms=worst_skew * 1000.0,
        tracker_translation_span_m=tracker_span,
        tracker_rotation_span_deg=math.degrees(relative.rotation_span_radians),
        motion_condition_number=relative.condition_number,
        optimizer_condition_number=optimizer_condition,
        optimizer_cost=float(solution.cost),
        optimizer_evaluations=int(solution.nfev),
    )
    return ViveCalibrationEstimate(calibration=calibration, report=report)


def capture_as_mapping(capture: ViveCalibrationCapture) -> dict:
    """Serialize a capture using the documented offline interchange schema."""

    return {
        "schema_version": VIVE_CALIBRATION_CAPTURE_SCHEMA_VERSION,
        "tracker_pose_frame": capture.tracker_pose_frame,
        "reference_frame": capture.reference_frame,
        "tracker_serial": capture.tracker_serial,
        "handedness": capture.handedness.value,
        "samples": [
            {
                "reference_timestamp": pair.reference_timestamp,
                "tracker_timestamp": pair.tracker_timestamp,
                "reference_T_wrist": pair.reference_from_wrist.as_mapping(),
                "lighthouse_T_tracker": pair.lighthouse_from_tracker.as_mapping(),
            }
            for pair in capture.pairs
        ],
    }


def estimate_from_pairs(
    pairs: Sequence[ViveCalibrationPair],
    *,
    reference_frame: str,
    tracker_serial: str,
    handedness: Handedness,
    **kwargs,
) -> ViveCalibrationEstimate:
    """Convenience wrapper for programmatic capture pipelines."""

    return estimate_vive_calibration(
        ViveCalibrationCapture(
            reference_frame, tracker_serial, handedness, tuple(pairs)
        ),
        **kwargs,
    )


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate VIVE lighthouse and wrist-mount extrinsics from paired poses.",
        epilog=VIVE_CALIBRATION_INPUT_SCHEMA,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="JSON pose-pair capture")
    parser.add_argument(
        "--output", type=Path, required=True, help="ViveCalibration JSON to create"
    )
    parser.add_argument(
        "--max-time-skew-ms",
        type=float,
        default=20.0,
        help="Reject any pose pair farther apart than this (default: 20)",
    )
    parser.add_argument(
        "--translation-scale-mm",
        type=float,
        default=10.0,
        help="Expected position noise used for residual weighting (default: 10)",
    )
    parser.add_argument(
        "--rotation-scale-deg",
        type=float,
        default=1.0,
        help="Expected orientation noise used for residual weighting (default: 1)",
    )
    parser.add_argument(
        "--loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
        help="Least-squares loss (default: soft_l1)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Run the device-free VIVE calibration CLI."""

    args = _cli_parser().parse_args(arguments)
    output = args.output.expanduser()
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to replace existing calibration {output}; pass --overwrite"
        )
    capture = ViveCalibrationCapture.load(args.input)
    estimate = estimate_vive_calibration(
        capture,
        max_time_skew_seconds=args.max_time_skew_ms / 1000.0,
        translation_scale_m=args.translation_scale_mm / 1000.0,
        rotation_scale_degrees=args.rotation_scale_deg,
        robust_loss=args.loss,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(estimate.calibration.as_mapping(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(json.dumps(estimate.report.as_mapping(), indent=2))
    return 0

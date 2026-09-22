"""Hardware acceptance diagnostic for MANUS Integrated or Remote acquisition."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from dex_teleop.tracking.manus import ManusSource
from dex_teleop.tracking.manus_calibration import ManusCoreCalibration
from dex_teleop.types import Handedness


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSONL acceptance log"
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--mode", choices=("integrated", "remote"), default="remote")
    parser.add_argument(
        "--core-host", help="Exact Core IP/name or sorted discovery index"
    )
    parser.add_argument("--loopback-only", action="store_true")
    parser.add_argument(
        "--hand-motion",
        choices=("auto", "tracker", "tracker_rotation_only", "imu", "none"),
        help="Default: tracker for Remote, auto for Integrated",
    )
    parser.add_argument("--sdk-root")
    parser.add_argument("--bridge")
    parser.add_argument("--glove-calibration")
    parser.add_argument("--core-calibration")
    parser.add_argument("--settings-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--startup-timeout", type=float, default=35.0)
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--glove-timeout", type=int, default=15)
    parser.add_argument("--reconnect-timeout", type=int, default=15)
    parser.add_argument("--discovery-wait", type=int, default=1)
    parser.add_argument("--minimum-rate-hz", type=float, default=30.0)
    parser.add_argument("--minimum-motion-m", type=float, default=0.05)
    parser.add_argument("--require-tracker-motion", action="store_true")
    parser.add_argument("--require-trackable-tracker", action="store_true")
    parser.add_argument(
        "--expected-tracker-id",
        help="Exact Ultimate ID required by tracker acceptance policy",
    )
    parser.add_argument(
        "--expected-tracker-user-id",
        type=int,
        help="Exact non-negative MANUS Core user assignment",
    )
    parser.add_argument("--tracker-stale-timeout", type=float, default=0.25)
    return parser


def _sample_record(articulation, wrist) -> dict:
    return {
        "type": "raw_skeleton_pair",
        "source": articulation.source,
        "handedness": articulation.handedness.value,
        "sequence": articulation.source_frame_id,
        "capture_monotonic_ns": articulation.source_timestamp_ns,
        "articulation_receipt_monotonic_ns": round(
            articulation.receipt_timestamp * 1e9
        ),
        "wrist_receipt_monotonic_ns": round(wrist.receipt_timestamp * 1e9),
        "manus_publish_time": articulation.provenance.get("manus_publish_time"),
        "mode": articulation.provenance.get("mode"),
        "core_host_name": articulation.provenance.get("core_host_name"),
        "core_host_ip": articulation.provenance.get("core_host_ip"),
        "core_version": articulation.provenance.get("core_version"),
        "sdk_version": articulation.provenance.get("sdk_version"),
        "hand_motion": articulation.provenance.get("hand_motion"),
        "connection_generation": articulation.provenance.get("connection_generation"),
        "glove_id": articulation.provenance.get("glove_id"),
        "articulation_schema": articulation.schema,
        "articulation_coordinate_frame": articulation.coordinate_frame,
        "joint_count": len(articulation.joint_names),
        "joint_positions": {
            name: value.tolist() for name, value in articulation.joint_positions.items()
        },
        "joint_orientations_xyzw": {
            name: value.tolist()
            for name, value in articulation.joint_orientations_xyzw.items()
        },
        "wrist_position": wrist.position.tolist(),
        "wrist_quaternion_xyzw": wrist.quaternion_xyzw.tolist(),
        "wrist_reference_frame": wrist.reference_frame,
        "wrist_anatomical_frame": wrist.anatomical_frame,
        "tracker_health": wrist.provenance.get("tracker_health"),
        "validated_tracker_id": wrist.provenance.get("validated_tracker_id"),
        "validated_tracker_quality": wrist.provenance.get("validated_tracker_quality"),
        "same_callback": (
            articulation.source == wrist.source
            and articulation.handedness == wrist.handedness
            and articulation.timestamp == wrist.timestamp
            and articulation.source_timestamp_ns == wrist.source_timestamp_ns
            and articulation.source_frame_id == wrist.source_frame_id
        ),
    }


def _tracker_record(diagnostics) -> dict:
    return {
        "type": "tracker_stream",
        "sequence": diagnostics.sequence,
        "manus_publish_time": diagnostics.manus_publish_time,
        "capture_monotonic_ns": diagnostics.capture_monotonic_ns,
        "receipt_monotonic_ns": diagnostics.receipt_monotonic_ns,
        "connection_generation": diagnostics.connection_generation,
        "session_id": diagnostics.session_id,
        "trackers": [dict(tracker) for tracker in diagnostics.trackers],
    }


def _rotation_span_degrees(quaternions: list[np.ndarray]) -> float:
    if not quaternions:
        return 0.0
    first = quaternions[0]
    return max(
        math.degrees(
            2.0 * math.acos(float(np.clip(abs(np.dot(first, item)), 0.0, 1.0)))
        )
        for item in quaternions
    )


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if (
        not math.isfinite(args.duration)
        or args.duration <= 0.0
        or not math.isfinite(args.minimum_rate_hz)
        or args.minimum_rate_hz < 0.0
        or not math.isfinite(args.minimum_motion_m)
        or args.minimum_motion_m < 0.0
    ):
        raise SystemExit(
            "Duration/rate/motion thresholds must be finite and non-negative"
        )
    if args.mode == "integrated" and args.core_host is not None:
        raise SystemExit("--core-host requires --mode remote")
    if not math.isfinite(args.tracker_stale_timeout) or args.tracker_stale_timeout <= 0:
        raise SystemExit("--tracker-stale-timeout must be positive and finite")

    expected_tracker_id = args.expected_tracker_id
    expected_tracker_type = f"{args.hand}_hand"
    expected_tracker_system = "openvr"
    expected_tracker_user_id = args.expected_tracker_user_id
    if args.core_calibration is not None:
        try:
            core_calibration = ManusCoreCalibration.load(args.core_calibration)
            wrist_calibration = core_calibration.wrist(args.hand)
            calibrated_tracker_id = wrist_calibration.tracker_id
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(f"Invalid MANUS Core calibration: {error}") from error
        if (
            calibrated_tracker_id is not None
            and calibrated_tracker_id != expected_tracker_id
        ):
            raise SystemExit(
                "--expected-tracker-id must match tracker_id in the selected "
                "Core calibration wrist"
            )
    require_validated_tracker = (
        args.require_tracker_motion or args.require_trackable_tracker
    )
    if require_validated_tracker and (
        expected_tracker_id is None or expected_tracker_user_id is None
    ):
        raise SystemExit(
            "Tracker acceptance requires --expected-tracker-id and "
            "--expected-tracker-user-id"
        )
    if expected_tracker_user_id is not None and expected_tracker_user_id < 0:
        raise SystemExit("--expected-tracker-user-id must be non-negative")

    calibration_argument = {f"{args.hand}_calibration": args.glove_calibration}
    source = ManusSource(
        bridge_executable=args.bridge,
        sdk_root=args.sdk_root,
        mode=args.mode,
        core_host=args.core_host,
        loopback_only=args.loopback_only,
        hand_motion=(
            args.hand_motion
            if args.hand_motion is not None
            else ("tracker" if args.mode == "remote" else "auto")
        ),
        tracker_diagnostics=True,
        startup_timeout=args.startup_timeout,
        connect_timeout=args.connect_timeout,
        glove_timeout=args.glove_timeout,
        reconnect_timeout=args.reconnect_timeout,
        discovery_wait=args.discovery_wait,
        settings_dir=args.settings_dir,
        log_dir=args.log_dir,
        core_calibration=args.core_calibration,
        required_handedness=args.hand,
        max_pending_samples=max(256, math.ceil(args.duration * 240.0)),
        require_tracker_for_wrist=require_validated_tracker,
        tracker_stale_timeout=args.tracker_stale_timeout,
        expected_tracker_id=expected_tracker_id,
        expected_tracker_user_id=expected_tracker_user_id,
        **calibration_argument,
    )
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    side = Handedness(args.hand)
    pair_records: list[dict] = []
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    tracker_packets: list[dict] = []
    last_tracker_sequence = None

    started = time.monotonic()
    with output.open("w", encoding="utf-8") as stream:
        source.start()
        try:
            stream.write(
                json.dumps(
                    {"type": "session", "metadata": source.recording_metadata()},
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                source.check_health()
                batch = source.drain_hand_tracking(side)
                if len(batch.articulations) != len(batch.wrists):
                    raise RuntimeError(
                        "MANUS combined drain returned an unpaired batch"
                    )
                for articulation, wrist in zip(
                    batch.articulations, batch.wrists, strict=True
                ):
                    record = _sample_record(articulation, wrist)
                    if not record["same_callback"]:
                        raise RuntimeError(
                            "MANUS articulation/wrist callback identity mismatch"
                        )
                    pair_records.append(record)
                    positions.append(wrist.position)
                    quaternions.append(wrist.quaternion_xyzw)
                    stream.write(
                        json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
                    )
                tracker = source.tracker_diagnostics()
                if tracker is not None and tracker.sequence != last_tracker_sequence:
                    last_tracker_sequence = tracker.sequence
                    record = _tracker_record(tracker)
                    tracker_packets.append(record)
                    stream.write(
                        json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
                    )
                time.sleep(0.002)
        finally:
            source.close()

        elapsed = max(time.monotonic() - started, np.finfo(np.float64).eps)
        capture_span = (
            0.0
            if len(pair_records) < 2
            else (
                pair_records[-1]["capture_monotonic_ns"]
                - pair_records[0]["capture_monotonic_ns"]
            )
            / 1e9
        )
        rate_hz = 0.0 if capture_span <= 0.0 else (len(pair_records) - 1) / capture_span
        translation_span = (
            0.0
            if not positions
            else max(float(np.linalg.norm(item - positions[0])) for item in positions)
        )
        trackable_ids = sorted(
            {
                tracker["id"]
                for packet in tracker_packets
                for tracker in packet["trackers"]
                if tracker.get("quality") == "trackable"
            }
        )
        expected_tracker_observations = [
            tracker
            for packet in tracker_packets
            for tracker in packet["trackers"]
            if tracker.get("id") == expected_tracker_id
        ]
        expected_tracker_valid = any(
            tracker.get("quality") == "trackable"
            and tracker.get("pose_valid") is True
            and tracker.get("is_hmd") is False
            and tracker.get("type") == expected_tracker_type
            and tracker.get("tracking_system") == expected_tracker_system
            and (
                expected_tracker_user_id is None
                or tracker.get("user_id") == expected_tracker_user_id
            )
            for tracker in expected_tracker_observations
        )
        failures = []
        if not pair_records:
            failures.append("no raw-skeleton pairs")
        if rate_hz < args.minimum_rate_hz:
            failures.append(
                f"raw-skeleton rate {rate_hz:.3f} Hz below {args.minimum_rate_hz:.3f} Hz"
            )
        if args.require_tracker_motion and translation_span < args.minimum_motion_m:
            failures.append(
                f"global wrist motion {translation_span:.4f} m below "
                f"{args.minimum_motion_m:.4f} m"
            )
        if args.require_trackable_tracker and not expected_tracker_valid:
            failures.append(
                f"expected tracker {expected_tracker_id!r} did not report the "
                "required Ultimate identity, role, pose, system, and quality"
            )
        summary = {
            "type": "summary",
            "passed": not failures,
            "failures": failures,
            "elapsed_seconds": elapsed,
            "raw_skeleton_pairs": len(pair_records),
            "raw_skeleton_rate_hz": rate_hz,
            "all_pairs_same_callback": all(
                record["same_callback"] for record in pair_records
            ),
            "wrist_translation_span_m": translation_span,
            "wrist_rotation_span_deg": _rotation_span_degrees(quaternions),
            "tracker_packet_count": len(tracker_packets),
            "trackable_tracker_ids": trackable_ids,
            "expected_tracker_id": expected_tracker_id,
            "expected_tracker_valid": expected_tracker_valid,
            "final_metadata": source.recording_metadata(),
        }
        stream.write(json.dumps(summary, allow_nan=False, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, allow_nan=False, sort_keys=True))
    print(f"Wrote MANUS diagnostic log: {output}")
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

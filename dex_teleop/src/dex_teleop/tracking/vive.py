"""Optional lighthouse VIVE Tracker wrist-pose source.

This provider talks to base-station VIVE trackers through ``pysurvive`` and is
deliberately independent of SteamVR and the active OpenXR runtime.  Tracker
roles and both extrinsic transforms must be explicit in a calibration file;
the implementation never assigns the first two discovered devices to hands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Mapping

import numpy as np

from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.tracking.transforms import RigidTransform, compose_transforms
from dex_teleop.types import Handedness, OPENXR_ANATOMICAL_WRIST_FRAME, WristPoseSample


VIVE_CALIBRATION_SCHEMA_VERSION = 1
VIVE_TRACKER_POSE_FRAME = "dex_teleop_lighthouse_rh_z_up"
LIBSURVIVE_TO_DEX_TELEOP_BASIS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

@dataclass(frozen=True)
class ViveTrackerMount:
    serial: str
    handedness: Handedness
    tracker_to_wrist: RigidTransform

    def __post_init__(self) -> None:
        if not isinstance(self.serial, str) or not self.serial.strip():
            raise ValueError("VIVE tracker serial must be non-empty")
        object.__setattr__(self, "handedness", Handedness(self.handedness))


@dataclass(frozen=True)
class ViveCalibration:
    """Persistent lighthouse-to-reference and per-mount wrist calibration."""

    reference_frame: str
    reference_from_lighthouse: RigidTransform
    mounts: tuple[ViveTrackerMount, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference_frame, str)
            or not self.reference_frame.strip()
        ):
            raise ValueError("VIVE calibration reference_frame must be non-empty")
        if not self.mounts:
            raise ValueError("VIVE calibration must contain at least one tracker mount")
        serials = [mount.serial for mount in self.mounts]
        sides = [mount.handedness for mount in self.mounts]
        if len(set(serials)) != len(serials):
            raise ValueError("VIVE calibration contains duplicate tracker serials")
        if len(set(sides)) != len(sides):
            raise ValueError(
                "VIVE calibration contains multiple trackers for the same hand"
            )

    @classmethod
    def load(cls, path: str | Path) -> "ViveCalibration":
        path = Path(path).expanduser()
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, Mapping):
            raise ValueError("VIVE calibration must be a JSON object")
        if document.get("schema_version") != VIVE_CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported VIVE calibration schema {document.get('schema_version')!r}; "
                f"expected {VIVE_CALIBRATION_SCHEMA_VERSION}"
            )
        if document.get("lighthouse_frame") != VIVE_TRACKER_POSE_FRAME:
            raise ValueError(
                "VIVE calibration lighthouse_frame must be "
                f"{VIVE_TRACKER_POSE_FRAME!r}; raw libsurvive Pose() arrays are not calibration-frame poses"
            )
        raw_mounts = document.get("trackers")
        if not isinstance(raw_mounts, list):
            raise ValueError("VIVE calibration trackers must be a list")
        try:
            mounts = tuple(
                ViveTrackerMount(
                    serial=raw["serial"],
                    handedness=Handedness(raw["handedness"]),
                    tracker_to_wrist=RigidTransform.from_mapping(
                        raw["tracker_to_wrist"],
                        field=f"trackers[{index}].tracker_to_wrist",
                    ),
                )
                for index, raw in enumerate(raw_mounts)
            )
            return cls(
                reference_frame=document["reference_frame"],
                reference_from_lighthouse=RigidTransform.from_mapping(
                    document["reference_from_lighthouse"],
                    field="reference_from_lighthouse",
                ),
                mounts=mounts,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid VIVE calibration: {error}") from error

    def as_mapping(self) -> dict:
        return {
            "schema_version": VIVE_CALIBRATION_SCHEMA_VERSION,
            "lighthouse_frame": VIVE_TRACKER_POSE_FRAME,
            "reference_frame": self.reference_frame,
            "reference_from_lighthouse": self.reference_from_lighthouse.as_mapping(),
            "trackers": [
                {
                    "serial": mount.serial,
                    "handedness": mount.handedness.value,
                    "tracker_to_wrist": mount.tracker_to_wrist.as_mapping(),
                }
                for mount in self.mounts
            ],
        }


def _survive_pose_to_lighthouse_transform(pose) -> RigidTransform:
    """Convert a raw libsurvive pose into dex_teleop's lighthouse basis.

    The calibration interchange schema uses this *converted* basis, not the
    native arrays returned by ``SimpleObject.Pose()``.
    """

    position = LIBSURVIVE_TO_DEX_TELEOP_BASIS @ np.asarray(pose.Pos, dtype=np.float64)
    quaternion_wxyz = pose.Rot
    return RigidTransform(
        position,
        [
            float(quaternion_wxyz[1]),
            float(-quaternion_wxyz[3]),
            float(quaternion_wxyz[2]),
            float(quaternion_wxyz[0]),
        ],
    )


def _decode_nonempty_identifier(value: object, *, field: str) -> str:
    if value is None:
        raise SourceUnavailableError(f"libsurvive returned an empty {field}")
    decoded = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if not decoded or not decoded.strip():
        raise SourceUnavailableError(f"libsurvive returned an empty {field}")
    return decoded


def _pose_timestamp_to_monotonic(
    pose_timestamp: object,
    *,
    receipt_monotonic: float,
    receipt_wall_time: float,
) -> tuple[float, float]:
    """Map libsurvive's Unix-epoch pose time into the host monotonic domain."""

    try:
        timestamp = float(pose_timestamp)
    except (TypeError, ValueError) as error:
        raise SourceUnavailableError(
            f"libsurvive returned an invalid pose timestamp {pose_timestamp!r}"
        ) from error
    if not math.isfinite(timestamp) or timestamp <= 0.0:
        raise SourceUnavailableError(
            f"libsurvive returned an invalid pose timestamp {pose_timestamp!r}"
        )
    capture_monotonic = receipt_monotonic + (timestamp - receipt_wall_time)
    if not math.isfinite(capture_monotonic):
        raise SourceUnavailableError(
            "Could not map the libsurvive pose timestamp to CLOCK_MONOTONIC"
        )
    return capture_monotonic, timestamp


class ViveWristSource:
    """Threaded, serial-bound VIVE lighthouse wrist source."""

    def __init__(
        self,
        calibration: ViveCalibration | str | Path,
        *,
        poll_interval: float = 0.002,
        max_pending_samples: int = 4096,
        context_factory: Callable[[], object] | None = None,
        context_close: Callable[[object], None] | None = None,
        serial_reader: Callable[[object], object] | None = None,
    ) -> None:
        if poll_interval <= 0.0:
            raise ValueError("poll_interval must be positive")
        if max_pending_samples <= 0:
            raise ValueError("max_pending_samples must be positive")
        self.calibration = (
            calibration
            if isinstance(calibration, ViveCalibration)
            else ViveCalibration.load(calibration)
        )
        self.poll_interval = float(poll_interval)
        self.max_pending_samples = int(max_pending_samples)
        self._context_factory = context_factory
        self._context_close = context_close
        self._serial_reader = serial_reader
        self._mounts = {mount.serial: mount for mount in self.calibration.mounts}
        self._latest: dict[Handedness, WristPoseSample] = {}
        self._pending = {side: deque() for side in Handedness}
        self._sequences = {side: 0 for side in Handedness}
        self._last_pose_timestamp_seconds: dict[str, float] = {}
        self._lock = threading.Lock()
        self._context_lock = threading.Lock()
        self._context: object | None = None
        self._live_context_close: Callable[[object], None] | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ViveWristSource has already been started")
        self._thread = threading.Thread(
            target=self._run, name="dex-teleop-vive", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self.close()
            raise SourceUnavailableError(
                "VIVE lighthouse source did not initialize within 5 seconds"
            )
        try:
            self.check_health()
        except BaseException:
            self.close()
            raise

    def read_wrist(self, handedness: Handedness) -> WristPoseSample | None:
        self.check_health()
        with self._lock:
            return self._latest.get(Handedness(handedness))

    def drain_wrists(self, handedness: Handedness) -> tuple[WristPoseSample, ...]:
        """Consume every acquired wrist sample in capture order."""

        self.check_health()
        side = Handedness(handedness)
        with self._lock:
            samples = tuple(self._pending[side])
            self._pending[side].clear()
        return samples

    def check_health(self) -> None:
        if self._error is not None:
            raise SourceUnavailableError(
                f"VIVE lighthouse source failed: {self._error}"
            ) from self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            # NextUpdated is non-blocking in supported libsurvive releases. Give
            # the normal loop a chance to stop before closing its context from
            # this thread; the latter is retained as an interrupt for bindings
            # that block unexpectedly.
            self._thread.join(timeout=max(0.05, 2.0 * self.poll_interval))
        if self._thread is not None and self._thread.is_alive():
            self._close_context_once()
            self._thread.join(timeout=2.0)
        else:
            self._close_context_once()

    def _default_context(
        self,
    ) -> tuple[object, Callable[[object], None], Callable[[object], object]]:
        try:
            import pysurvive
            from pysurvive.pysurvive_generated import (
                survive_simple_close,
                survive_simple_serial_number,
            )
        except ImportError as error:
            raise SourceUnavailableError(
                "VIVE lighthouse tracking requires the optional pysurvive package"
            ) from error

        context = pysurvive.SimpleContext([sys.argv[0]])
        if context is None:
            raise SourceUnavailableError("pysurvive failed to create a SimpleContext")

        def close_context(value) -> None:
            survive_simple_close(value.ptr)

        def read_serial(value) -> object:
            return survive_simple_serial_number(value.ptr)

        return context, close_context, read_serial

    def _install_context(
        self,
        context: object,
        close_context: Callable[[object], None] | None,
    ) -> None:
        with self._context_lock:
            self._context = context
            self._live_context_close = close_context

    def _close_context_once(self) -> None:
        with self._context_lock:
            context = self._context
            close_context = self._live_context_close
            self._context = None
            self._live_context_close = None
        if context is not None and close_context is not None:
            close_context(context)

    def _run(self) -> None:
        context = None
        close_context = self._context_close
        serial_reader = self._serial_reader
        try:
            if self._context_factory is None:
                context, close_context, serial_reader = self._default_context()
            else:
                context = self._context_factory()
                if context is None:
                    raise SourceUnavailableError("VIVE context factory returned None")
            if serial_reader is None:
                raise SourceUnavailableError(
                    "A custom VIVE context requires serial_reader so calibration binds the hardware serial"
                )
            self._install_context(context, close_context)
            self._ready.set()

            while not self._stop.is_set():
                updated = context.NextUpdated()
                if self._stop.is_set():
                    break
                if not updated:
                    self._stop.wait(self.poll_interval)
                    continue
                serial = _decode_nonempty_identifier(
                    serial_reader(updated), field="hardware serial"
                )
                codename = _decode_nonempty_identifier(
                    updated.Name(), field="object codename"
                )
                mount = self._mounts.get(serial)
                if mount is None:
                    continue
                pose, pose_timestamp = updated.Pose()
                receipt_wall_time = time.time()
                receipt_timestamp = time.monotonic()
                capture_timestamp, raw_pose_timestamp = _pose_timestamp_to_monotonic(
                    pose_timestamp,
                    receipt_monotonic=receipt_timestamp,
                    receipt_wall_time=receipt_wall_time,
                )
                lighthouse_from_tracker = _survive_pose_to_lighthouse_transform(pose)
                reference_from_tracker = compose_transforms(
                    self.calibration.reference_from_lighthouse,
                    lighthouse_from_tracker,
                )
                reference_from_wrist = compose_transforms(
                    reference_from_tracker,
                    mount.tracker_to_wrist,
                )
                with self._lock:
                    previous_timestamp = self._last_pose_timestamp_seconds.get(serial)
                    if (
                        previous_timestamp is not None
                        and raw_pose_timestamp == previous_timestamp
                    ):
                        continue
                    if (
                        previous_timestamp is not None
                        and raw_pose_timestamp < previous_timestamp
                    ):
                        raise SourceUnavailableError(
                            f"libsurvive pose time moved backwards for tracker {serial}: "
                            f"{raw_pose_timestamp} after {previous_timestamp} seconds since Unix epoch"
                        )
                    sequence = self._sequences[mount.handedness]
                    self._sequences[mount.handedness] += 1
                    sample = WristPoseSample(
                        timestamp=capture_timestamp,
                        receipt_timestamp=receipt_timestamp,
                        # libsurvive exposes seconds, not an integer nanosecond
                        # counter. Preserve that native value in provenance.
                        source_timestamp_ns=None,
                        source_frame_id=sequence,
                        handedness=mount.handedness,
                        position=reference_from_wrist.translation,
                        quaternion_xyzw=reference_from_wrist.quaternion_xyzw,
                        source=f"vive:{serial}",
                        reference_frame=self.calibration.reference_frame,
                        anatomical_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
                        provenance={
                            "hardware_serial": serial,
                            "libsurvive_codename": codename,
                            "raw_pose_timestamp": raw_pose_timestamp,
                            "raw_pose_timestamp_units": "seconds_since_unix_epoch",
                            "lighthouse_frame": VIVE_TRACKER_POSE_FRAME,
                        },
                    )
                    queue = self._pending[mount.handedness]
                    if len(queue) >= self.max_pending_samples:
                        raise SourceUnavailableError(
                            f"VIVE {mount.handedness.value}-wrist acquisition overflowed its "
                            f"{self.max_pending_samples}-sample queue; the consumer is not draining fast enough"
                        )
                    self._last_pose_timestamp_seconds[serial] = raw_pose_timestamp
                    queue.append(sample)
                    self._latest[mount.handedness] = sample
        except BaseException as error:
            self._error = error
            self._ready.set()
        finally:
            try:
                self._close_context_once()
            except BaseException as error:
                if self._error is None:
                    self._error = error

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

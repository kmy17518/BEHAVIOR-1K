"""Articulation-only tracking: forward one articulation source and pin the wrist."""

from __future__ import annotations

import threading
from typing import Mapping

import numpy as np

from dex_teleop.tracking.multimodal import HandTrackingSampleBatch
from dex_teleop.types import HandArticulationSample, Handedness, WristPoseSample


FIXED_WRIST_SOURCE_NAME = "fixed_wrist"
FIXED_WRIST_REFERENCE_FRAME = "fixed_wrist"


class FixedWristSource:
    """Wrap an articulation source so the runtime sees a rigidly fixed wrist.

    Register one instance under two keys of ``MultiSourceTrackingWorker``: the
    articulation key (``quest``, ``manus``) and a wrist key.  The worker then
    drains this object once per cycle through :meth:`drain_hand_tracking` and,
    because both roles resolve to the same object, treats every synthesized
    wrist as co-emitted with its articulation sample.  Fusion therefore never
    searches or interpolates, and the wrist carried by the wrapped source (Quest
    pose, MANUS IMU or tracker) is discarded rather than used.

    Finger retargeting is unaffected: both retargeters normalize the landmarks
    against a wrist frame estimated from the landmarks themselves, so a constant
    wrist only fixes where the ``HandFrame`` landmarks sit in the world.
    """

    def __init__(
        self,
        articulation_source,
        *,
        position=(0.0, 0.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_name: str = FIXED_WRIST_SOURCE_NAME,
    ) -> None:
        for method in ("start", "check_health", "close", "read_articulation"):
            if not callable(getattr(articulation_source, method, None)):
                raise TypeError(f"Articulation source must provide {method}()")
        if not source_name or not source_name.strip() or "/" in source_name:
            raise ValueError("source_name must be non-empty and may not contain '/'")
        pose_position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        pose_quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4).copy()
        if not np.isfinite(pose_position).all() or not np.isfinite(pose_quaternion).all():
            raise ValueError("Fixed wrist pose must be finite")
        norm = float(np.linalg.norm(pose_quaternion))
        if norm <= 0.0:
            raise ValueError("Fixed wrist quaternion must have non-zero norm")
        self.articulation_source = articulation_source
        self.source_name = source_name
        self.position = pose_position
        self.quaternion_xyzw = pose_quaternion / norm
        self.position.setflags(write=False)
        self.quaternion_xyzw.setflags(write=False)
        self._lock = threading.Lock()
        self._latest_wrist: dict[Handedness, WristPoseSample] = {}
        self._last_read_identity: dict[Handedness, tuple] = {}

    # Lifecycle delegates to the single physical receiver.
    def start(self) -> None:
        self.articulation_source.start()

    def check_health(self) -> None:
        self.articulation_source.check_health()

    def close(self) -> None:
        self.articulation_source.close()

    def read_articulation(self, handedness: Handedness) -> HandArticulationSample | None:
        return self.articulation_source.read_articulation(handedness)

    def read_wrist(self, handedness: Handedness) -> WristPoseSample | None:
        """Return the fixed wrist paired with the most recently drained articulation.

        Before the first drain this peeks at the wrapped source's latest
        articulation so latest-value callers still receive a matching pair.
        """

        side = Handedness(handedness)
        with self._lock:
            latest = self._latest_wrist.get(side)
        if latest is not None:
            return latest
        sample = self.articulation_source.read_articulation(side)
        return None if sample is None else self.wrist_for(sample)

    def wrist_for(self, articulation: HandArticulationSample) -> WristPoseSample:
        """Synthesize the co-emitted fixed wrist for one articulation sample."""

        provenance: Mapping[str, object] = {
            "articulation_source": articulation.source,
            "fixed_wrist": True,
        }
        return WristPoseSample(
            timestamp=articulation.timestamp,
            receipt_timestamp=articulation.receipt_timestamp,
            source_timestamp_ns=articulation.source_timestamp_ns,
            source_frame_id=articulation.source_frame_id,
            handedness=articulation.handedness,
            position=self.position,
            quaternion_xyzw=self.quaternion_xyzw,
            source=self.source_name,
            reference_frame=FIXED_WRIST_REFERENCE_FRAME,
            anatomical_frame=articulation.coordinate_frame,
            confidence=None,
            provenance=provenance,
        )

    def drain_hand_tracking(self, handedness: Handedness) -> HandTrackingSampleBatch:
        """Drain the wrapped articulation samples and pair each with a fixed wrist."""

        side = Handedness(handedness)
        articulations = self._drain_articulations(side)
        wrists = tuple(self.wrist_for(sample) for sample in articulations)
        if wrists:
            with self._lock:
                self._latest_wrist[side] = wrists[-1]
        return HandTrackingSampleBatch(articulations=articulations, wrists=wrists)

    def _drain_articulations(self, side: Handedness) -> tuple[HandArticulationSample, ...]:
        source = self.articulation_source
        drain_combined = getattr(source, "drain_hand_tracking", None)
        if callable(drain_combined):
            batch = drain_combined(side)
            if not isinstance(batch, HandTrackingSampleBatch):
                raise TypeError(
                    f"{type(source).__name__}.drain_hand_tracking returned "
                    f"{type(batch).__name__}, not HandTrackingSampleBatch"
                )
            # The wrapped receiver's own wrist samples are intentionally dropped.
            return tuple(batch.articulations)
        drain = getattr(source, "drain_articulations", None)
        if callable(drain):
            return tuple(drain(side))
        sample = source.read_articulation(side)
        if sample is None:
            return ()
        identity = (sample.source_frame_id, sample.source_timestamp_ns, sample.timestamp)
        with self._lock:
            if self._last_read_identity.get(side) == identity:
                return ()
            self._last_read_identity[side] = identity
        return (sample,)

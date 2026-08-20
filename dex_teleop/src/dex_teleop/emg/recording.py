"""Append action timing and native-rate EMG to a completed OG recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


EMG_GROUP = "emg"
ACTION_TIMING_GROUP = "action_timing"
SYNC_GROUP = "synchronization"
EMG_SCHEMA_VERSION = 1
ACTION_TIMING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActionTimingSample:
    """Wall-time boundaries and simulator time for one recorded action."""

    action_apply_monotonic_ns: int
    step_return_monotonic_ns: int
    sim_time_before_s: float
    sim_time_after_s: float

    def __post_init__(self) -> None:
        if self.action_apply_monotonic_ns < 0 or self.step_return_monotonic_ns < 0:
            raise ValueError("Action monotonic timestamps must be non-negative")
        if self.step_return_monotonic_ns < self.action_apply_monotonic_ns:
            raise ValueError("Action step return precedes action application")
        if not np.isfinite(self.sim_time_before_s) or not np.isfinite(self.sim_time_after_s):
            raise ValueError("Action simulator timestamps must be finite")
        if self.sim_time_after_s < self.sim_time_before_s:
            raise ValueError("Simulator time moved backwards during an action")


def _episode_groups(group: h5py.Group) -> list[tuple[int, h5py.Group]]:
    episodes = []
    for name, episode in group.items():
        if not name.startswith("demo_") or not isinstance(episode, h5py.Group):
            continue
        try:
            episode_id = int(name.removeprefix("demo_"))
        except ValueError:
            continue
        episodes.append((episode_id, episode))
    return sorted(episodes)


def write_action_timing_episodes(
    input_path: str | Path,
    episodes: list[list[ActionTimingSample]],
) -> None:
    """Store one timing row for every action recorded by the OG wrapper."""

    input_path = Path(input_path)
    with h5py.File(input_path, "r+") as recording:
        trajectory_episodes = _episode_groups(recording["data"])
        if len(episodes) != len(trajectory_episodes):
            raise RuntimeError(
                f"Action-timing episode count {len(episodes)} does not match trajectory count "
                f"{len(trajectory_episodes)}"
            )

        if ACTION_TIMING_GROUP in recording:
            del recording[ACTION_TIMING_GROUP]
        timing = recording.create_group(ACTION_TIMING_GROUP)
        timing.attrs["schema_version"] = ACTION_TIMING_SCHEMA_VERSION
        timing.attrs["clock"] = "time.monotonic_ns"
        timing.attrs["action_dataset"] = "/data/demo_N/action"
        timing.attrs["row_alignment"] = "one timing row for each action row with the same index"
        timing.attrs["action_semantics"] = "timestamp immediately before env.step(action)"
        timing.attrs["step_return_semantics"] = "timestamp immediately after env.step(action) returns"

        for samples, (episode_id, trajectory) in zip(episodes, trajectory_episodes, strict=True):
            expected_steps = int(trajectory.attrs["num_samples"])
            if len(samples) != expected_steps:
                raise RuntimeError(
                    f"Action timing demo_{episode_id} has {len(samples)} steps; trajectory has {expected_steps}"
                )
            episode = timing.create_group(f"demo_{episode_id}")
            episode.create_dataset(
                "action_apply_monotonic_ns",
                data=np.asarray([sample.action_apply_monotonic_ns for sample in samples], dtype=np.int64),
            )
            episode.create_dataset(
                "step_return_monotonic_ns",
                data=np.asarray([sample.step_return_monotonic_ns for sample in samples], dtype=np.int64),
            )
            episode.create_dataset(
                "sim_time_before_s",
                data=np.asarray([sample.sim_time_before_s for sample in samples], dtype=np.float64),
            )
            episode.create_dataset(
                "sim_time_after_s",
                data=np.asarray([sample.sim_time_after_s for sample in samples], dtype=np.float64),
            )
        recording.flush()


def merge_emg_recording(recording_path: str | Path, emg_path: str | Path) -> None:
    """Copy a closed sidecar recording and derive per-episode EMG row ranges."""

    recording_path = Path(recording_path)
    emg_path = Path(emg_path)
    with h5py.File(emg_path, "r") as emg_recording, h5py.File(recording_path, "r+") as recording:
        if EMG_GROUP not in emg_recording:
            raise RuntimeError(f"EMG sidecar recording has no '{EMG_GROUP}' group: {emg_path}")
        if ACTION_TIMING_GROUP not in recording:
            raise RuntimeError("Action timing must be written before EMG synchronization")
        if EMG_GROUP in recording:
            del recording[EMG_GROUP]
        emg_recording.copy(EMG_GROUP, recording)
        emg = recording[EMG_GROUP]
        emg.attrs["schema_version"] = EMG_SCHEMA_VERSION

        estimated_time = np.asarray(emg["samples/estimated_monotonic_ns"], dtype=np.int64)
        if len(estimated_time) > 1 and np.any(np.diff(estimated_time) <= 0):
            raise RuntimeError("Estimated EMG timestamps are not strictly increasing")
        if SYNC_GROUP in recording:
            del recording[SYNC_GROUP]
        synchronization = recording.create_group(SYNC_GROUP)
        synchronization.attrs["schema_version"] = 1
        synchronization.attrs["mapping"] = (
            "EMG half-open row ranges aligned to action application times in the desktop monotonic clock"
        )

        timing_episodes = _episode_groups(recording[ACTION_TIMING_GROUP])
        trajectory_episodes = _episode_groups(recording["data"])
        if [item[0] for item in timing_episodes] != [item[0] for item in trajectory_episodes]:
            raise RuntimeError("Action timing episode IDs do not match trajectory episode IDs")

        for episode_id, timing in timing_episodes:
            apply_times = np.asarray(timing["action_apply_monotonic_ns"], dtype=np.int64)
            return_times = np.asarray(timing["step_return_monotonic_ns"], dtype=np.int64)
            if len(apply_times) > 1 and np.any(np.diff(apply_times) <= 0):
                raise RuntimeError(f"Action application timestamps move backwards in demo_{episode_id}")
            if len(apply_times) == 0 or len(estimated_time) == 0:
                episode_row_start = episode_row_end = 0
                action_row_start = np.zeros(len(apply_times), dtype=np.int64)
                action_row_end = np.zeros(len(apply_times), dtype=np.int64)
            else:
                episode_row_start = int(np.searchsorted(estimated_time, apply_times[0], side="left"))
                episode_row_end = int(np.searchsorted(estimated_time, return_times[-1], side="right"))
                action_end_times = np.concatenate((apply_times[1:], return_times[-1:]))
                action_row_start = np.searchsorted(estimated_time, apply_times, side="left").astype(np.int64)
                action_row_end = np.searchsorted(estimated_time, action_end_times, side="left").astype(np.int64)
                action_row_end[-1] = episode_row_end
            episode = synchronization.create_group(f"demo_{episode_id}")
            episode.create_dataset("episode_emg_row_start", data=np.int64(episode_row_start))
            episode.create_dataset("episode_emg_row_end", data=np.int64(episode_row_end))
            episode.create_dataset("action_emg_row_start", data=action_row_start)
            episode.create_dataset("action_emg_row_end", data=action_row_end)
            episode.attrs["range_semantics"] = "half-open [row_start, row_end) into /emg/samples"
            episode.attrs["action_interval_semantics"] = (
                "action apply through next action apply; the final action ends when env.step returns"
            )
        recording.flush()

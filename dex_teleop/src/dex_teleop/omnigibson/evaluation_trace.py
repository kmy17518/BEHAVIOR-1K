"""Per-step BDDL and ARAT evaluation traces stored alongside HDF5 trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import h5py


EVALUATION_SCHEMA_VERSION = 1
EVALUATION_GROUP = "evaluation"
EVALUATION_STEPS_DATASET = "steps"


def build_bddl_trace(goal_conditions, goal_status: dict) -> dict:
    """Expand BehaviorTask goal indexes into readable per-condition status."""

    satisfied = {int(index) for index in goal_status["satisfied"]}
    unsatisfied = {int(index) for index in goal_status["unsatisfied"]}
    conditions = []
    for index, text in enumerate(goal_conditions):
        conditions.append(
            {
                "index": index,
                "text": str(text),
                "satisfied": index in satisfied,
            }
        )
    return {
        "satisfied_count": len(satisfied),
        "total_count": len(conditions),
        "all_satisfied": not unsatisfied,
        "satisfied_indices": sorted(satisfied),
        "unsatisfied_indices": sorted(unsatisfied),
        "conditions": conditions,
    }


def build_step_evaluation(goal_conditions, goal_status: dict, arat_trace: dict | None) -> dict:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "bddl": build_bddl_trace(goal_conditions, goal_status),
        "arat": {"enabled": False} if arat_trace is None else {"enabled": True, **arat_trace},
    }


def _episode_groups(group) -> list[tuple[int, h5py.Group]]:
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


def write_evaluation_episodes(input_path: str | Path, episodes: list[list[dict]]) -> None:
    """Append action-aligned evaluation JSON to an otherwise completed recording."""

    input_path = Path(input_path)
    with h5py.File(input_path, "r+") as recording:
        trajectory_episodes = _episode_groups(recording["data"])
        if len(episodes) != len(trajectory_episodes):
            raise RuntimeError(
                f"Evaluation episode count {len(episodes)} does not match trajectory count {len(trajectory_episodes)}"
            )

        if EVALUATION_GROUP in recording:
            del recording[EVALUATION_GROUP]
        evaluation = recording.create_group(EVALUATION_GROUP)
        evaluation.attrs["schema_version"] = EVALUATION_SCHEMA_VERSION
        string_dtype = h5py.string_dtype(encoding="utf-8")

        for trace_steps, (episode_id, trajectory) in zip(episodes, trajectory_episodes):
            expected_steps = int(trajectory.attrs["num_samples"])
            if len(trace_steps) != expected_steps:
                raise RuntimeError(
                    f"Evaluation demo_{episode_id} has {len(trace_steps)} steps; trajectory has {expected_steps}"
                )
            episode = evaluation.create_group(f"demo_{episode_id}")
            encoded = [json.dumps(step, separators=(",", ":"), ensure_ascii=False) for step in trace_steps]
            episode.create_dataset(EVALUATION_STEPS_DATASET, data=encoded, dtype=string_dtype)
        recording.flush()


def read_evaluation_episode(input_path: str | Path, episode_id: int, expected_steps: int | None = None) -> list[dict]:
    """Load one recorded evaluation trace without launching OmniGibson."""

    input_path = Path(input_path)
    with h5py.File(input_path, "r") as recording:
        try:
            dataset = recording[EVALUATION_GROUP][f"demo_{episode_id}"][EVALUATION_STEPS_DATASET]
        except KeyError as error:
            raise ValueError(
                "This recording has no per-step evaluation trace. Record it again with the current launch_og.py."
            ) from error
        steps = []
        for encoded in dataset:
            if isinstance(encoded, bytes):
                encoded = encoded.decode("utf-8")
            steps.append(json.loads(encoded))

    if expected_steps is not None and len(steps) != expected_steps:
        raise ValueError(
            f"Evaluation trace has {len(steps)} steps, but demo_{episode_id} has {expected_steps} trajectory steps"
        )
    return steps

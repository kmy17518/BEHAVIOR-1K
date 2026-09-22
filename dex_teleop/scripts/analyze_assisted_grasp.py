#!/usr/bin/env python3
"""Summarize assisted-grasp weld failures recorded by launch_og.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "dex_teleop" / "src"))

from dex_teleop.omnigibson.assisted_grasp_trace import (  # noqa: E402
    ASSISTED_GRASP_GROUP,
    read_assisted_grasp_config,
    read_assisted_grasp_episode,
    summarize_assisted_grasp_episode,
)


def _episode_ids(recording_path: Path) -> list[int]:
    with h5py.File(recording_path, "r") as recording:
        try:
            names = recording[ASSISTED_GRASP_GROUP].keys()
        except KeyError as error:
            raise ValueError(
                "This recording has no assisted-grasp trace. Record it with "
                "--assisted-grasp --assisted-grasp-debug."
            ) from error
        episode_ids = []
        for name in names:
            if name.startswith("demo_"):
                try:
                    episode_ids.append(int(name.removeprefix("demo_")))
                except ValueError:
                    pass
    return sorted(episode_ids)


def _format_optional(value, unit: str, precision: int = 3) -> str:
    return "unavailable" if value is None else f"{float(value):.{precision}f} {unit}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--episode", type=int, help="Analyze only this demo_N episode")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary")
    args = parser.parse_args(argv)

    recording_path = args.recording.expanduser().resolve()
    config = read_assisted_grasp_config(recording_path)
    episode_ids = [args.episode] if args.episode is not None else _episode_ids(recording_path)
    summaries = {
        episode_id: summarize_assisted_grasp_episode(read_assisted_grasp_episode(recording_path, episode_id))
        for episode_id in episode_ids
    }
    if args.json:
        print(json.dumps({"recording": str(recording_path), "config": config, "episodes": summaries}, indent=2))
        return

    print(f"Recording: {recording_path}")
    print(
        "Weld config: "
        f"break force={config.get('weld_break_force')} N, "
        f"break torque={config.get('weld_break_torque')} Nm, "
        f"squeeze bias={config.get('frozen_squeeze_bias_rad')} rad, "
        f"drift release={config.get('weld_drift_release_m')} m"
    )
    for episode_id, summary in summaries.items():
        weld_count = sum(event["kind"] in {"welded", "adopted"} for event in summary["events"])
        release_count = sum(event["kind"] == "released" for event in summary["events"])
        print(
            f"\ndemo_{episode_id}: {summary['steps']} action steps, {weld_count} weld(s), "
            f"{release_count} intentional release(s), {len(summary['breaks'])} broken weld(s)"
        )
        for failure in summary["breaks"]:
            time_s = failure["episode_time_s"]
            time_text = "unknown time" if time_s is None else f"t={float(time_s):.3f}s"
            print(
                f"  step {failure['step']} ({time_text}): {failure['object']} -> "
                f"{failure['classification']}, drift={_format_optional(failure['drift_m'], 'm')}, "
                f"max finger force={_format_optional(failure['max_finger_contact_force_n'], 'N')}, "
                f"deepest separation={_format_optional(failure['min_contact_separation_m'], 'm', 5)}"
            )
            if failure["joint_break_events"]:
                paths = ", ".join(event["joint_path"] for event in failure["joint_break_events"])
                print(f"    PhysX JOINT_BREAK: {paths}")
            if failure["external_contacts"]:
                print(f"    Held-object external contacts: {', '.join(failure['external_contacts'])}")
        if not summary["breaks"]:
            print("  No supervisor-detected broken welds in this episode.")


if __name__ == "__main__":
    main()

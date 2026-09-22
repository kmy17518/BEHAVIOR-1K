#!/usr/bin/env python
"""Offline ARAT re-scoring of a recorded teleoperation episode.

Replays an HDF5 recording made by the dex_teleop launcher's ``--recording-path`` option
(full serialized sim state per step) and runs the same ARAT scorer the launcher uses
online, so scores are reproducible and auditable after the fact.

The launcher records only engaged ``env.step`` calls (disengaged frames step the raw
simulator and are not captured), so every recorded step counts as engaged task time.

Usage:
    OMNIGIBSON_HEADLESS=1 conda run -n behavior_dex python \
        scripts/arat/score_recorded_episode.py --recording /path/to/recording.h5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "dex_teleop" / "src", REPO_ROOT / "bddl3", REPO_ROOT / "OmniGibson"):
    sys.path.insert(0, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="HDF5 file recorded via the launcher's --recording-path")
    parser.add_argument("--episode", type=int, default=0, help="Episode (demo) index inside the recording")
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "dex_teleop" / "outputs" / "arat_results"),
        help="Base directory for the re-scored item result JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recording = Path(args.recording).expanduser().resolve()
    if not recording.is_file():
        raise SystemExit(f"Recording not found: {recording}")

    data_root = REPO_ROOT / "datasets"
    configured = os.environ.get("OMNIGIBSON_DATA_PATH")
    if configured is not None and Path(configured).expanduser().resolve() != data_root.resolve():
        raise SystemExit(f"OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat; unset it or set it to {data_root}")
    os.environ["OMNIGIBSON_DATA_PATH"] = str(data_root)

    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    gm.RENDER_VIEWER_CAMERA = False

    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper

    from dex_teleop.arat import AratTaskCatalog
    from dex_teleop.arat.eval import load_rubrics
    from dex_teleop.arat.eval.live import LiveAratEvaluator
    from dex_teleop.arat.eval.report import format_item_summary, make_results_dir, write_item_result

    results_dir = make_results_dir(Path(args.results_dir))
    playback = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=str(recording),
        output_path=str(results_dir / "playback_obs.h5"),
        robot_obs_modalities=(),
        n_render_iterations=1,
        include_task=True,
        include_robot_control=True,
        include_contacts=True,
    )
    env = playback.env
    robot = env.robots[0]

    activity = env.task.activity_name
    catalog = AratTaskCatalog()
    if activity not in catalog.tasks:
        raise SystemExit(f"Recorded activity {activity!r} is not an ARAT task")
    task = catalog.tasks[activity]
    rubric = load_rubrics()[activity]
    evaluator = LiveAratEvaluator(env, robot, task, rubric)

    print(f"Re-scoring {activity} from {recording} (episode {args.episode})")
    playback.playback_episode(
        episode_id=args.episode,
        record_data=False,
        post_state_update_callback=lambda: evaluator.step(engaged=True),
    )

    result = evaluator.finalize()
    print(format_item_summary(result))
    result_path = write_item_result(result, results_dir)
    print(f"Wrote {result_path}")
    og.shutdown()


if __name__ == "__main__":
    main()

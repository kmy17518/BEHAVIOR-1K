#!/usr/bin/env python3
"""Render the ARAT task overview PNGs from the left-shoulder camera.

Run this script in ``behavior_dex`` with ``OMNIGIBSON_HEADLESS=1``. Each
output is a horizontal concatenation of 640 x 480 RGB frames in the declared
task order.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "datasets"
configured_data_root = os.environ.get("OMNIGIBSON_DATA_PATH")
if configured_data_root is not None and Path(configured_data_root).expanduser().resolve() != DATA_ROOT.resolve():
    raise SystemExit(
        "OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat; unset it or set it to "
        f"{DATA_ROOT}"
    )
os.environ["OMNIGIBSON_DATA_PATH"] = str(DATA_ROOT)
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

for source_root in (
    REPOSITORY_ROOT / "dex_teleop" / "src",
    REPOSITORY_ROOT / "OmniGibson",
    REPOSITORY_ROOT / "bddl3",
):
    sys.path.insert(0, str(source_root))

from dex_teleop.arat import AratTaskCatalog  # noqa: E402
from dex_teleop.arat.scene import validate_runtime_assets  # noqa: E402
from dex_teleop.omnigibson.launcher import (  # noqa: E402
    CAMERA_IMAGE_HEIGHT,
    CAMERA_IMAGE_WIDTH,
    RESET_JOINT_POSITIONS,
    _external_camera_configs,
    _hide_skybox_from_camera,
    _reset_arat_box,
    _show_robot_end_effectors,
    _validate_loaded_apparatus,
    build_environment_config,
)


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "dex_teleop" / "outputs"
LEFT_CAMERA_NAME = "arat_left_shoulder_camera"
OUTPUT_GROUPS = OrderedDict(
    {
        "grasp": (
            "arat_grasp_block_5cm",
            "arat_grasp_cricket_ball",
            "arat_grasp_sharpening_stone",
        ),
        "grip": (
            "arat_grip_pour_water",
            "arat_grip_alloy_tube_2_5cm",
            "arat_grip_alloy_tube_1cm",
            "arat_grip_washer_over_bolt",
        ),
        "pinch": (
            "arat_pinch_ball_bearing_index",
            "arat_pinch_marble_index",
        ),
        "gross_movement": ("arat_gross_movement_hand_mouth",),
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(OUTPUT_GROUPS),
        default=tuple(OUTPUT_GROUPS),
        help="Output groups to render; defaults to all four",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="Physics steps before capture, matching launch_og.py's idle loop",
    )
    parser.add_argument(
        "--render-frames",
        type=int,
        default=30,
        help="Render frames before capture so RTX materials and lighting converge",
    )
    return parser


def _left_camera_config() -> dict:
    camera = _external_camera_configs(include_wrist=False)[0]
    if camera["name"] != LEFT_CAMERA_NAME:
        raise AssertionError(f"Expected {LEFT_CAMERA_NAME}, got {camera['name']}")
    camera["modalities"] = ["rgb"]
    camera["sensor_kwargs"]["image_height"] = CAMERA_IMAGE_HEIGHT
    camera["sensor_kwargs"]["image_width"] = CAMERA_IMAGE_WIDTH
    return camera


def _rgb_to_image(rgb) -> Image.Image:
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[:2] != (CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH):
        raise ValueError(f"Unexpected left-camera RGB shape: {rgb.shape}")
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB or RGBA camera output, got shape {rgb.shape}")
    if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def render_task(task, settle_steps: int, render_frames: int) -> Image.Image:
    import omnigibson as og
    import torch as th

    if og.sim is not None and og.sim.scenes:
        og.clear()

    config = build_environment_config(task)
    config["env"]["external_sensors"] = [_left_camera_config()]
    env = og.Environment(configs=config)
    try:
        env.reset()
        _validate_loaded_apparatus(env, task)
        _reset_arat_box(env)
        _hide_skybox_from_camera()
        robot = env.robots[0]
        robot.set_joint_positions(th.tensor(RESET_JOINT_POSITIONS, dtype=th.float32))
        robot.keep_still()
        _show_robot_end_effectors(robot)
        # Match launch_og.py's disengaged loop before taking a still. Advancing
        # physics is important here: render-only frames do not synchronize an
        # articulated box pose through PhysX / Fabric after _reset_arat_box().
        reset_positions = th.tensor(RESET_JOINT_POSITIONS, dtype=th.float32)
        for _ in range(settle_steps):
            robot.set_joint_positions(reset_positions)
            robot.keep_still()
            og.sim.step()
        for _ in range(render_frames):
            og.sim.render()
        observations, _ = env.external_sensors[LEFT_CAMERA_NAME].get_obs()
        return _rgb_to_image(observations["rgb"])
    finally:
        og.clear()


def concatenate_frames(frames: list[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (CAMERA_IMAGE_WIDTH * len(frames), CAMERA_IMAGE_HEIGHT))
    for index, frame in enumerate(frames):
        output.paste(frame, (index * CAMERA_IMAGE_WIDTH, 0))
    return output


def main() -> None:
    import omnigibson as og
    from omnigibson.macros import gm

    args = _parser().parse_args()
    if args.settle_steps < 1:
        raise SystemExit("--settle-steps must be positive")
    if args.render_frames < 1:
        raise SystemExit("--render-frames must be positive")

    catalog = AratTaskCatalog()
    selected_activities = tuple(
        activity for group in args.groups for activity in OUTPUT_GROUPS[group]
    )
    selected_tasks = tuple(catalog.tasks[activity] for activity in selected_activities)
    validate_runtime_assets(selected_tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gm.USE_GPU_DYNAMICS = True
    try:
        for group in args.groups:
            activities = OUTPUT_GROUPS[group]
            frames = []
            for index, activity in enumerate(activities, start=1):
                print(f"ARAT_IMAGE_RENDERING group={group} panel={index}/{len(activities)} task={activity}", flush=True)
                frames.append(render_task(catalog.tasks[activity], args.settle_steps, args.render_frames))
            output_path = args.output_dir / f"{group}.png"
            temporary_path = output_path.with_name(f".{output_path.name}.tmp.png")
            concatenate_frames(frames).save(temporary_path)
            temporary_path.replace(output_path)
            print(f"ARAT_IMAGE_WRITTEN group={group} path={output_path}", flush=True)
    finally:
        if og.sim is not None:
            og.shutdown()


if __name__ == "__main__":
    main()

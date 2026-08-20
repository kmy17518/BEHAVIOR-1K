#!/usr/bin/env python3
"""Inspect and render the Franka + Sharpa dual wrist-camera mount in OmniGibson."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "datasets"
EXTRINSICS_PATH = (
    DATA_ROOT
    / "omnigibson-robot-assets"
    / "models"
    / "franka"
    / "franka_dexhand"
    / "franka_sharpa_right"
    / "source"
    / "wrist_camera_extrinsics.yaml"
)
CAMERA_IDS = ("thumb_wrist", "pinky_wrist")
SENSOR_NAMES = {
    "thumb_wrist": "arat_wrist_camera_thumb",
    "pinky_wrist": "arat_wrist_camera_pinky",
}
FRAME_COLORS = (
    (1.0, 0.1, 0.1, 1.0),
    (0.1, 1.0, 0.1, 1.0),
    (0.1, 0.4, 1.0, 1.0),
)
FRUSTUM_COLORS = {
    "thumb_wrist": (1.0, 0.75, 0.1, 1.0),
    "pinky_wrist": (1.0, 0.1, 0.8, 1.0),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="arat_grasp_block_10cm")
    parser.add_argument("--reset-pose", choices=("compact", "extended"), default="extended")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "wrist_camera_check")
    parser.add_argument("--headless", action="store_true", help="Render and exit without opening Kit windows")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="GUI simulation steps after capture; zero runs until Ctrl+C (ignored in headless mode)",
    )
    return parser


def _configure_import_paths(*, headless: bool) -> None:
    configured_data_root = os.environ.get("OMNIGIBSON_DATA_PATH")
    if configured_data_root is not None and Path(configured_data_root).expanduser().resolve() != DATA_ROOT.resolve():
        raise SystemExit(f"OMNIGIBSON_DATA_PATH must be unset or point to {DATA_ROOT}")
    os.environ["OMNIGIBSON_DATA_PATH"] = str(DATA_ROOT)
    if headless:
        os.environ["OMNIGIBSON_HEADLESS"] = "1"
    for path in (REPOSITORY_ROOT / "dex_teleop" / "src", REPOSITORY_ROOT / "OmniGibson", REPOSITORY_ROOT / "bddl3"):
        sys.path.insert(0, str(path))


def _camera_only_config(task_name: str, reset_pose: str) -> tuple[dict, object, object]:
    from dex_teleop.arat import AratTaskCatalog
    from dex_teleop.arat.camera_rig import load_camera_rig
    from dex_teleop.omnigibson.launcher import build_environment_config

    catalog = AratTaskCatalog()
    try:
        task = catalog.tasks[task_name]
    except KeyError as error:
        raise SystemExit(f"Unknown ARAT task {task_name!r}") from error
    rig = load_camera_rig(task.camera_rig)
    config = build_environment_config(task, reset_pose=reset_pose)
    wanted = set(SENSOR_NAMES.values())
    sensors = []
    for sensor in config["env"]["external_sensors"]:
        if sensor["name"] not in wanted:
            continue
        sensor["modalities"] = ["rgb"]
        sensor["sensor_kwargs"]["viewport_name"] = None
        sensors.append(sensor)
    if {sensor["name"] for sensor in sensors} != wanted:
        raise RuntimeError(f"Camera rig {task.camera_rig!r} does not expose both wrist cameras")
    config["env"]["external_sensors"] = sensors
    config["task"] = {"type": "DummyTask"}
    return config, task, rig


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _save_rgb(sensor, output_path: Path) -> np.ndarray:
    observation, _ = sensor.get_obs()
    rgb = _as_numpy(observation["rgb"])[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb * (255.0 if float(rgb.max()) <= 1.0 else 1.0), 0, 255)
    rgb = rgb.astype(np.uint8)
    Image.fromarray(rgb).save(output_path)
    return rgb


def _world_pose(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from omnigibson.utils.usd_utils import get_world_pose

    position, orientation = get_world_pose(prim_path)
    return _as_numpy(position), _as_numpy(orientation)


def _frame_segments(position: np.ndarray, quaternion_xyzw: np.ndarray, length: float = 0.045):
    rotation = Rotation.from_quat(quaternion_xyzw)
    starts = np.repeat(position[None], 3, axis=0)
    ends = starts + rotation.apply(np.eye(3) * length)
    return list(zip(starts, ends, FRAME_COLORS))


def _pinhole_frustum_segments(
    position: np.ndarray,
    optical_quaternion_xyzw: np.ndarray,
    *,
    horizontal_fov: float,
    aspect: float,
    far: float = 0.22,
    color,
):
    half_width = math.tan(horizontal_fov / 2.0) * far
    half_height = half_width / aspect
    corners_local = np.array(
        [
            [-half_width, -half_height, far],
            [half_width, -half_height, far],
            [half_width, half_height, far],
            [-half_width, half_height, far],
        ]
    )
    corners = Rotation.from_quat(optical_quaternion_xyzw).apply(corners_local) + position
    segments = [(position, corner, color) for corner in corners]
    segments.extend((corners[index], corners[(index + 1) % 4], color) for index in range(4))
    return segments


def _fisheye_frustum_segments(position, optical_quaternion_xyzw, parameters: dict, *, far: float, color):
    width, height = parameters["image_size"]
    cx, cy = parameters["principal_point"]
    fx, fy = parameters["focal_length_pixels"]
    samples_per_edge = 12
    alphas = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
    perimeter = [(alpha * width, 0.0) for alpha in alphas]
    perimeter.extend((width, alpha * height) for alpha in alphas)
    perimeter.extend(((1.0 - alpha) * width, height) for alpha in alphas)
    perimeter.extend((0.0, (1.0 - alpha) * height) for alpha in alphas)
    directions = []
    for pixel_x, pixel_y in perimeter:
        distorted = np.array([(pixel_x - cx) / fx, (pixel_y - cy) / fy])
        theta = np.linalg.norm(distorted)
        if theta == 0.0:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            radial = distorted / theta
            direction = np.array([radial[0] * math.sin(theta), radial[1] * math.sin(theta), math.cos(theta)])
        directions.append(direction)
    rotation = Rotation.from_quat(optical_quaternion_xyzw)
    boundary = rotation.apply(np.asarray(directions) * far) + position
    principal_end = rotation.apply([0.0, 0.0, far]) + position
    segments = [(position, principal_end, color)]
    segments.extend((position, point, color) for point in boundary[::samples_per_edge])
    segments.extend((boundary[index], boundary[(index + 1) % len(boundary)], color) for index in range(len(boundary)))
    return segments


def _draw_frames_and_frustums(robot, rig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import omnigibson.lazy as lazy
    from dex_teleop.arat.camera_rig import opencv_fisheye_parameters

    draw = lazy.isaacsim.util.debug_draw._debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()
    palm_path = robot.links["right_hand_C_MC"].prim_path
    frame_poses = {}
    segments = []
    for camera_id, side in (("thumb_wrist", "thumb"), ("pinky_wrist", "pinky")):
        body_path = f"{palm_path}/right_wrist_camera_mount/right_wrist_camera_{side}"
        optical_path = f"{body_path}/right_wrist_camera_{side}_optical"
        body_pose = _world_pose(body_path)
        optical_pose = _world_pose(optical_path)
        frame_poses[camera_id] = optical_pose
        segments.extend(_frame_segments(*body_pose))
        segments.extend(_frame_segments(*optical_pose))
        calibration = rig.calibration(camera_id)
        fisheye = opencv_fisheye_parameters(rig, camera_id)
        if fisheye is not None:
            segments.extend(
                _fisheye_frustum_segments(
                    *optical_pose,
                    parameters=fisheye,
                    far=0.22,
                    color=FRUSTUM_COLORS[camera_id],
                )
            )
        else:
            horizontal_fov = 2.0 * math.atan(
                calibration["horizontal_aperture"] / (2.0 * calibration["focal_length"])
            )
            segments.extend(
                _pinhole_frustum_segments(
                    *optical_pose,
                    horizontal_fov=horizontal_fov,
                    aspect=calibration["image_width"] / calibration["image_height"],
                    color=FRUSTUM_COLORS[camera_id],
                )
            )
    draw.draw_lines(
        [tuple(start) for start, _, _ in segments],
        [tuple(end) for _, end, _ in segments],
        [color for _, _, color in segments],
        [3.0] * len(segments),
    )
    return frame_poses


def _visible_fingertips(robot, optical_pose, calibration: dict, fisheye: dict | None = None) -> list[str]:
    position, quaternion = optical_pose
    world_to_optical = Rotation.from_quat(quaternion).inv()
    horizontal_fov = 2.0 * math.atan(calibration["horizontal_aperture"] / (2.0 * calibration["focal_length"]))
    tan_horizontal = math.tan(horizontal_fov / 2.0)
    tan_vertical = tan_horizontal / (calibration["image_width"] / calibration["image_height"])
    visible = []
    for digit in ("thumb", "index", "middle", "ring", "pinky"):
        tip_position, _ = robot.links[f"right_{digit}_fingertip"].get_position_orientation()
        optical = world_to_optical.apply(_as_numpy(tip_position) - position)
        if fisheye is not None:
            direction = optical / np.linalg.norm(optical)
            theta = math.acos(float(np.clip(direction[2], -1.0, 1.0)))
            radial_norm = np.linalg.norm(direction[:2])
            radial = np.zeros(2) if radial_norm == 0.0 else direction[:2] / radial_norm
            fx, fy = fisheye["focal_length_pixels"]
            cx, cy = fisheye["principal_point"]
            pixel = np.array([cx + fx * theta * radial[0], cy + fy * theta * radial[1]])
            width, height = fisheye["image_size"]
            in_view = 0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height
        else:
            in_view = (
                optical[2] > 0
                and abs(optical[0] / optical[2]) <= tan_horizontal
                and abs(optical[1] / optical[2]) <= tan_vertical
            )
        if in_view:
            visible.append(digit)
    return visible


def _validate_sensor_pose(sensor, optical_pose) -> float:
    _, sensor_quaternion = sensor.get_position_orientation()
    _, optical_quaternion = optical_pose
    expected_usd = Rotation.from_quat(optical_quaternion) * Rotation.from_euler("x", math.pi)
    error = expected_usd.inv() * Rotation.from_quat(_as_numpy(sensor_quaternion))
    return math.degrees(error.magnitude())


def main() -> None:
    args = _parser().parse_args()
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    _configure_import_paths(headless=args.headless)
    config, task, rig = _camera_only_config(args.task, args.reset_pose)

    from dex_teleop.arat.camera_rig import apply_camera_lens_models, opencv_fisheye_parameters

    import omnigibson as og
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    gm.USE_PBR_MATERIALS = True
    gm.GUI_VIEWPORT_ONLY = False
    gm.RENDER_VIEWER_CAMERA = not args.headless

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extrinsics = yaml.safe_load(EXTRINSICS_PATH.read_text(encoding="utf-8"))
    env = None
    try:
        env = og.Environment(configs=config)
        env.reset()
        apply_camera_lens_models(rig, env.external_sensors, CAMERA_IDS)
        robot = env.robots[0]
        for _ in range(8):
            og.sim.render()

        frame_poses = _draw_frames_and_frustums(robot, rig)
        images = {}
        for camera_id in CAMERA_IDS:
            sensor = env.external_sensors[SENSOR_NAMES[camera_id]]
            images[camera_id] = _save_rgb(sensor, args.output_dir / f"{camera_id}.png")
            visible = _visible_fingertips(
                robot,
                frame_poses[camera_id],
                rig.calibration(camera_id),
                opencv_fisheye_parameters(rig, camera_id),
            )
            angular_error = _validate_sensor_pose(sensor, frame_poses[camera_id])
            if len(visible) < 4:
                raise RuntimeError(f"{camera_id} sees too few open-pose fingertips: {visible}")
            if angular_error > 0.05:
                raise RuntimeError(f"{camera_id} sensor and optical frame differ by {angular_error:.6f} degrees")
            print(f"{camera_id}: visible open-pose fingertips={visible}; sensor/frame error={angular_error:.6f} deg")

        difference = float(np.mean(np.abs(images["thumb_wrist"].astype(float) - images["pinky_wrist"].astype(float))))
        if difference < 1.0:
            raise RuntimeError(f"Wrist images are nearly identical (mean absolute pixel difference {difference:.3f})")
        print(f"Rendered {args.output_dir / 'thumb_wrist.png'}")
        print(f"Rendered {args.output_dir / 'pinky_wrist.png'}")
        print(f"Mean absolute thumb/pinky image difference: {difference:.3f}")
        print(
            "Frames: X=red, Y=green, Z=blue; thumb frustum=yellow, pinky frustum=magenta. "
            f"Both principal rays follow +Z of {extrinsics['hand_frame']['link']}."
        )

        if not args.headless:
            print(f"Inspecting {task.activity}/{args.reset_pose}; press Ctrl+C to exit.")
            steps = 0
            while args.steps == 0 or steps < args.steps:
                og.sim.step()
                steps += 1
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            og.shutdown()


if __name__ == "__main__":
    main()

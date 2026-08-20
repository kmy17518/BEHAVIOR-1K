"""Versioned camera extrinsics, calibration, and viewport layouts for ARAT."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import yaml


CAMERA_RIGS_PATH = Path(__file__).with_name("camera_rigs.yaml")
CAMERA_RIG_FILENAMES = ("camera_rigs.yaml", "arat_default.yaml")
SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CameraRig:
    """A validated camera rig loaded from a packaged camera-rig YAML file."""

    name: str
    sensor_defaults: dict
    cameras: dict[str, dict]
    layouts: dict[str, dict]

    def camera(self, camera_id: str) -> dict:
        try:
            return self.cameras[camera_id]
        except KeyError as error:
            raise ValueError(f"Camera rig {self.name!r} has no camera {camera_id!r}") from error

    def layout(self, layout_name: str) -> dict:
        try:
            return self.layouts[layout_name]
        except KeyError as error:
            raise ValueError(f"Camera rig {self.name!r} has no layout {layout_name!r}") from error

    def calibration(self, camera_id: str) -> dict:
        calibration = deepcopy(self.sensor_defaults["calibration"])
        calibration.update(self.camera(camera_id).get("calibration", {}))
        return calibration


def _require_vector(value, length: int, description: str) -> None:
    if not isinstance(value, list) or len(value) != length or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{description} must contain {length} numeric values")


def _validate_rig(rig: CameraRig) -> None:
    defaults = rig.sensor_defaults
    calibration = defaults.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"Camera rig {rig.name!r} must define sensor_defaults.calibration")
    for field in ("image_width", "image_height"):
        if not isinstance(calibration.get(field), int) or calibration[field] <= 0:
            raise ValueError(f"Camera rig {rig.name!r} calibration {field} must be a positive integer")

    sensor_names = set()
    for camera_id, camera in rig.cameras.items():
        for field in ("sensor_name", "prim_name", "parent", "pose"):
            if field not in camera:
                raise ValueError(f"Camera {camera_id!r} in rig {rig.name!r} is missing {field!r}")
        sensor_name = camera["sensor_name"]
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(f"Camera {camera_id!r} has an invalid sensor_name")
        if sensor_name in sensor_names:
            raise ValueError(f"Camera rig {rig.name!r} repeats sensor_name {sensor_name!r}")
        sensor_names.add(sensor_name)

        parent = camera["parent"]
        if not isinstance(parent, dict) or parent.get("frame") not in {"scene", "robot_link"}:
            raise ValueError(f"Camera {camera_id!r} parent.frame must be 'scene' or 'robot_link'")
        if parent["frame"] == "robot_link" and not isinstance(parent.get("link"), str):
            raise ValueError(f"Robot-linked camera {camera_id!r} must name its parent link")
        pose = camera["pose"]
        if not isinstance(pose, dict):
            raise ValueError(f"Camera {camera_id!r} pose must be a mapping")
        _require_vector(pose.get("position"), 3, f"Camera {camera_id!r} position")
        _require_vector(pose.get("orientation_xyzw"), 4, f"Camera {camera_id!r} orientation_xyzw")

        merged_calibration = rig.calibration(camera_id)
        for field in ("image_width", "image_height"):
            if not isinstance(merged_calibration.get(field), int) or merged_calibration[field] <= 0:
                raise ValueError(f"Camera {camera_id!r} calibration {field} must be a positive integer")

        lens_model = camera.get("lens_model")
        if lens_model is not None:
            if not isinstance(lens_model, dict) or lens_model.get("type") != "opencv_fisheye":
                raise ValueError(f"Camera {camera_id!r} lens_model.type must be 'opencv_fisheye'")
            horizontal_fov = lens_model.get("horizontal_fov_degrees")
            if not isinstance(horizontal_fov, (int, float)) or not 0.0 < horizontal_fov < 360.0:
                raise ValueError(f"Camera {camera_id!r} fisheye horizontal FOV must be between 0 and 360 degrees")
            coefficients = lens_model.get("distortion_coefficients")
            _require_vector(coefficients, 4, f"Camera {camera_id!r} fisheye distortion_coefficients")
            principal_point = lens_model.get("principal_point")
            if principal_point is not None:
                _require_vector(principal_point, 2, f"Camera {camera_id!r} fisheye principal_point")

    for layout_name, layout in rig.layouts.items():
        workspace = layout.get("workspace", {})
        if not isinstance(workspace, dict):
            raise ValueError(f"Camera layout {layout_name!r} workspace must be a mapping")
        for field in ("viewport_only", "fill_viewports"):
            if field in workspace and not isinstance(workspace[field], bool):
                raise ValueError(f"Camera layout {layout_name!r} workspace {field} must be true or false")

        viewports = layout.get("viewports")
        if not isinstance(viewports, dict) or not viewports:
            raise ValueError(f"Camera layout {layout_name!r} must define viewports")
        main_viewports = [viewport_id for viewport_id, viewport in viewports.items() if viewport.get("main")]
        if main_viewports != ["main"]:
            raise ValueError(f"Camera layout {layout_name!r} must define exactly one main viewport named 'main'")
        for viewport_id, viewport in viewports.items():
            is_empty = viewport.get("empty", False)
            if not isinstance(is_empty, bool):
                raise ValueError(f"Viewport {viewport_id!r} empty must be true or false")
            camera_id = viewport.get("camera")
            if is_empty:
                if viewport_id == "main" or camera_id is not None:
                    raise ValueError("Only auxiliary viewports may be empty, without a camera")
            elif camera_id not in rig.cameras:
                raise ValueError(f"Viewport {viewport_id!r} references unknown camera {camera_id!r}")
            if viewport_id == "main":
                continue
            dock = viewport.get("dock")
            if not isinstance(viewport.get("window_name"), str) or not isinstance(dock, dict):
                raise ValueError(f"Viewport {viewport_id!r} must define window_name and dock")
            if dock.get("position") not in {"left", "right", "top", "bottom"}:
                raise ValueError(f"Viewport {viewport_id!r} has an invalid dock position")
            parent = dock.get("parent")
            if parent != "DockSpace" and parent not in viewports:
                raise ValueError(f"Viewport {viewport_id!r} has unknown dock parent {parent!r}")
            if not isinstance(dock.get("ratio"), (int, float)) or not 0 < dock["ratio"] < 1:
                raise ValueError(f"Viewport {viewport_id!r} dock ratio must be between zero and one")

        panels = layout.get("panels", {})
        if not isinstance(panels, dict):
            raise ValueError(f"Camera layout {layout_name!r} panels must be a mapping")
        for panel_id, panel in panels.items():
            dock = panel.get("dock") if isinstance(panel, dict) else None
            if not isinstance(dock, dict) or not isinstance(dock.get("parent"), str):
                raise ValueError(f"Panel {panel_id!r} must define a dock parent")
            if dock.get("position") not in {"left", "right", "top", "bottom", "same"}:
                raise ValueError(f"Panel {panel_id!r} has an invalid dock position")
            if not isinstance(dock.get("ratio", 0.5), (int, float)) or not 0 < dock.get("ratio", 0.5) < 1:
                raise ValueError(f"Panel {panel_id!r} dock ratio must be between zero and one")

        toggle_keys = set()
        for toggle in layout.get("toggles", []):
            key = toggle.get("key")
            viewport_id = toggle.get("viewport")
            camera_ids = toggle.get("cameras")
            if not isinstance(key, str) or len(key) != 1 or key.upper() in toggle_keys:
                raise ValueError(f"Camera layout {layout_name!r} has an invalid or duplicate toggle key")
            toggle_keys.add(key.upper())
            if viewport_id not in viewports:
                raise ValueError(f"Camera toggle {key!r} references unknown viewport {viewport_id!r}")
            if viewports[viewport_id].get("empty", False):
                raise ValueError(f"Camera toggle {key!r} cannot target an empty viewport")
            if not isinstance(camera_ids, list) or len(camera_ids) < 2:
                raise ValueError(f"Camera toggle {key!r} must cycle at least two cameras")
            if any(camera_id not in rig.cameras for camera_id in camera_ids):
                raise ValueError(f"Camera toggle {key!r} references an unknown camera")
            if camera_ids[0] != viewports[viewport_id]["camera"]:
                raise ValueError(f"Camera toggle {key!r} must start with the viewport's initial camera")


def camera_rig_paths() -> tuple[Path, ...]:
    """Return all packaged camera-rig YAML files in deterministic order."""

    return tuple(CAMERA_RIGS_PATH.with_name(filename) for filename in CAMERA_RIG_FILENAMES)


def _load_camera_rig_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"Camera rigs YAML {path} must use schema_version {SUPPORTED_SCHEMA_VERSION}")
    camera_rigs = raw.get("camera_rigs")
    if not isinstance(camera_rigs, dict):
        raise ValueError(f"Camera rigs YAML {path} must define camera_rigs")
    return camera_rigs


def camera_rig_names() -> tuple[str, ...]:
    """Return all camera-rig names available through packaged YAML files."""

    names = []
    for path in camera_rig_paths():
        names.extend(_load_camera_rig_file(path))
    if len(names) != len(set(names)):
        raise ValueError("Camera rig names must be unique across camera-rig YAML files")
    return tuple(names)


def load_camera_rig(name: str, path: str | Path | None = None) -> CameraRig:
    """Load and validate one named camera rig from one file or the packaged registry."""

    paths = (Path(path),) if path is not None else camera_rig_paths()
    matches = []
    known_names = []
    for candidate in paths:
        camera_rigs = _load_camera_rig_file(candidate)
        known_names.extend(camera_rigs)
        if name in camera_rigs:
            matches.append(camera_rigs[name])
    if not matches:
        raise ValueError(f"Unknown camera rig {name!r}; choose from {tuple(known_names)}")
    if len(matches) > 1:
        raise ValueError(f"Camera rig {name!r} is defined in more than one YAML file")
    data = matches[0]
    rig = CameraRig(
        name=name,
        sensor_defaults=deepcopy(data["sensor_defaults"]),
        cameras=deepcopy(data["cameras"]),
        layouts=deepcopy(data["layouts"]),
    )
    _validate_rig(rig)
    return rig


def layout_camera_ids(rig: CameraRig, layout_name: str) -> tuple[str, ...]:
    """Return camera IDs used by the layout's viewports or toggle cycles."""

    layout = rig.layout(layout_name)
    referenced = {
        viewport["camera"]
        for viewport in layout["viewports"].values()
        if not viewport.get("empty", False)
    }
    for toggle in layout.get("toggles", []):
        referenced.update(toggle["cameras"])
    return tuple(camera_id for camera_id in rig.cameras if camera_id in referenced)


def opencv_fisheye_parameters(rig: CameraRig, camera_id: str) -> dict | None:
    """Resolve an ideal OpenCV-fisheye calibration from a camera-rig entry."""

    camera = rig.camera(camera_id)
    lens_model = camera.get("lens_model")
    if lens_model is None:
        return None
    calibration = rig.calibration(camera_id)
    width = calibration["image_width"]
    height = calibration["image_height"]
    horizontal_fov = math.radians(lens_model["horizontal_fov_degrees"])
    focal_length_pixels = width / horizontal_fov
    principal_point = lens_model.get("principal_point", [width / 2.0, height / 2.0])
    return {
        "image_size": [width, height],
        "principal_point": list(principal_point),
        "focal_length_pixels": [focal_length_pixels, focal_length_pixels],
        "distortion_coefficients": list(lens_model["distortion_coefficients"]),
        "horizontal_fov_degrees": float(lens_model["horizontal_fov_degrees"]),
    }


def apply_camera_lens_models(rig: CameraRig, sensors: dict, camera_ids: Iterable[str]) -> None:
    """Apply configured Isaac Sim lens-distortion schemas to loaded OmniGibson sensors."""

    configured = []
    for camera_id in camera_ids:
        parameters = opencv_fisheye_parameters(rig, camera_id)
        if parameters is None:
            continue
        camera = rig.camera(camera_id)
        try:
            sensor = sensors[camera["sensor_name"]]
        except KeyError as error:
            raise RuntimeError(f"Cannot configure missing camera sensor {camera['sensor_name']!r}") from error
        configured.append((sensor, parameters))
    if not configured:
        return

    import omnigibson as og
    import omnigibson.lazy as lazy

    with og.sim.editing_usd():
        for sensor, parameters in configured:
            prim = sensor.prim
            prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI")
            prim.GetAttribute("omni:lensdistortion:model").Set("opencvFisheye")
            width, height = parameters["image_size"]
            cx, cy = parameters["principal_point"]
            fx, fy = parameters["focal_length_pixels"]
            values = {
                "imageSize": lazy.pxr.Gf.Vec2i(width, height),
                "cx": cx,
                "cy": cy,
                "fx": fx,
                "fy": fy,
            }
            values.update(
                {
                    f"k{index + 1}": coefficient
                    for index, coefficient in enumerate(parameters["distortion_coefficients"])
                }
            )
            for attribute, value in values.items():
                prim.GetAttribute(f"omni:lensdistortion:opencvFisheye:{attribute}").Set(value)


def build_camera_sensor_configs(
    rig: CameraRig,
    layout_name: str,
    *,
    robot_prim_path: str,
    camera_ids: Iterable[str] | None = None,
    viewport_name: str | None = "Viewport",
) -> list[dict]:
    """Convert YAML camera definitions to OmniGibson external-sensor configs."""

    selected_ids = layout_camera_ids(rig, layout_name) if camera_ids is None else tuple(camera_ids)
    configs = []
    for camera_id in selected_ids:
        camera = rig.camera(camera_id)
        parent = camera["parent"]
        if parent["frame"] == "scene":
            relative_prim_path = f"/{camera['prim_name']}"
            pose_frame = "scene"
        else:
            if not robot_prim_path:
                raise ValueError(f"Camera {camera_id!r} requires a robot prim path")
            relative_prim_path = f"{robot_prim_path}/{parent['link']}/{camera['prim_name']}"
            pose_frame = "parent"
        pose = camera["pose"]
        configs.append(
            {
                "sensor_type": "VisionSensor",
                "name": camera["sensor_name"],
                "relative_prim_path": relative_prim_path,
                "modalities": deepcopy(rig.sensor_defaults.get("modalities", [])),
                "sensor_kwargs": {"viewport_name": viewport_name, **rig.calibration(camera_id)},
                "position": deepcopy(pose["position"]),
                "orientation": deepcopy(pose["orientation_xyzw"]),
                "pose_frame": pose_frame,
                "include_in_obs": bool(rig.sensor_defaults.get("include_in_obs", False)),
            }
        )
    return configs

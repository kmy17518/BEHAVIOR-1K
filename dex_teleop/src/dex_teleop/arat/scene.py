"""Resolve the version-1 ARAT assets and per-activity scene templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dex_teleop.arat.catalog import AratTask


DATASET_NAME = "arat-assets-v1"
ROBOT_NAME = "franka_sharpa_right"
ROBOT_DATASET_NAME = "omnigibson-robot-assets"
ROBOT_MODEL = "franka"
ROBOT_END_EFFECTOR = "sharpa_right"
SCENE_MODEL = "arat_base"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DATASET_ROOT = _REPOSITORY_ROOT / "datasets" / DATASET_NAME
_ROBOT_DATASET_ROOT = _REPOSITORY_ROOT / "datasets" / ROBOT_DATASET_NAME
_SCENE_ROOT = (
    _REPOSITORY_ROOT
    / "datasets"
    / "arat-task-instances"
    / "scenes"
    / SCENE_MODEL
    / "json"
)


def get_task_scene_path(task: AratTask) -> Path:
    """Return the saved OmniGibson scene template for ``task``."""

    return _SCENE_ROOT / f"{SCENE_MODEL}_task_{task.activity}_0_0_template.json"


def get_task_tro_path(task: AratTask) -> Path:
    """Return the task-relevant-object state file for ``task``."""

    stem = f"{SCENE_MODEL}_task_{task.activity}"
    return _SCENE_ROOT / f"{stem}_instances" / f"{stem}_0_0_template-tro_state.json"


def get_task_tro_data(task: AratTask) -> dict:
    """Load and validate the task's pose-bearing TRO state."""

    tro_path = get_task_tro_path(task)
    try:
        tro = json.loads(tro_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing ARAT TRO state: {tro_path}") from error

    try:
        poses = tro["robot_poses"]["robot"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed ARAT robot poses in TRO state: {tro_path}") from error
    if not isinstance(poses, list) or len(poses) != 1:
        raise ValueError(f"{tro_path} must define exactly one generic robot pose")
    pose = poses[0]
    if not isinstance(pose, dict) or len(pose.get("position", ())) != 3 or len(pose.get("orientation", ())) != 4:
        raise ValueError(f"Malformed ARAT robot pose in TRO state: {tro_path}")
    return tro


def get_task_scene_data(task: AratTask, *, include_task_metadata: bool = False) -> dict:
    """Load a task scene, optionally embedding its BDDL map and TRO robot poses."""

    scene_path = get_task_scene_path(task)
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing ARAT scene template: {scene_path}") from error
    if include_task_metadata:
        scene.setdefault("metadata", {}).setdefault("task", {}).update(build_task_metadata(task))
    return scene


def get_task_scene_object_names(task: AratTask) -> frozenset[str]:
    """Return the object names encoded in a task's saved scene."""

    scene_path = get_task_scene_path(task)
    scene = get_task_scene_data(task)
    try:
        return frozenset(scene["objects_info"]["init_info"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed ARAT scene template: {scene_path}") from error


def build_task_metadata(task: AratTask) -> dict:
    """Build the explicit BDDL-instance-to-saved-object map for one task."""

    object_names = get_task_scene_object_names(task)
    inst_to_name = {"agent.n.01_1": ROBOT_NAME, **task.instances}
    if task.subscale != "gross_movement":
        inst_to_name["breakfast_table.n.01_1"] = "table"
    if task.activity == "arat_grip_pour_water":
        inst_to_name["water.n.06_1"] = "water"
    missing = set(inst_to_name.values()).difference({ROBOT_NAME, "water", *object_names})
    if missing:
        raise ValueError(f"{task.activity} maps BDDL instances to unknown scene objects: {sorted(missing)}")
    return {
        "activity": task.activity,
        "asset_version": 1,
        "inst_to_name": inst_to_name,
        "robot_poses": get_task_tro_data(task)["robot_poses"],
    }


def validate_runtime_assets(tasks: Iterable[AratTask]) -> None:
    """Fail early when a selected scene or a required version-1 asset is unavailable."""

    missing = []
    required = (
        _DATASET_ROOT / "scenes" / SCENE_MODEL / "json" / f"{SCENE_MODEL}_best.json",
        _DATASET_ROOT / "objects" / "breakfast_table" / "nvoqyl" / "usd" / "nvoqyl.usd",
        _DATASET_ROOT / "objects" / "arat_box" / "aratbx" / "usd" / "aratbx.usd",
        _DATASET_ROOT / "objects" / "mannequin" / "nphsfp" / "usd" / "nphsfp.usd",
        _ROBOT_DATASET_ROOT / "models" / ROBOT_MODEL / f"{ROBOT_MODEL}.yaml",
        _ROBOT_DATASET_ROOT
        / "models"
        / ROBOT_MODEL
        / "franka_dexhand"
        / f"franka_{ROBOT_END_EFFECTOR}"
        / "usd"
        / f"franka_{ROBOT_END_EFFECTOR}.usda",
    )
    missing.extend(str(path) for path in required if not path.is_file())

    for task in tasks:
        scene_path = get_task_scene_path(task)
        tro_path = get_task_tro_path(task)
        missing_task_assets = [path for path in (scene_path, tro_path) if not path.is_file()]
        if missing_task_assets:
            missing.extend(str(path) for path in missing_task_assets)
            continue
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        init_info = scene.get("init_info", {})
        if init_info.get("class_name") != "Scene":
            raise ValueError(f"{scene_path} is not a plain Scene template")
        scene_args = init_info.get("args", {})
        expected_scene_args = {
            "use_floor_plane": True,
            "floor_plane_visible": True,
            "floor_plane_color": [0.5, 0.5, 0.5],
            "use_skybox": True,
        }
        for key, expected in expected_scene_args.items():
            if scene_args.get(key) != expected:
                raise ValueError(f"{scene_path} has {key}={scene_args.get(key)!r}, expected {expected!r}")
        init_objects = scene.get("objects_info", {}).get("init_info", {}).values()
        for obj_info in init_objects:
            is_foreign_dataset_object = (
                obj_info.get("class_name") == "DatasetObject"
                and obj_info.get("args", {}).get("dataset_name") != DATASET_NAME
            )
            if is_foreign_dataset_object:
                raise ValueError(f"{scene_path} contains an object outside {DATASET_NAME}")
        names = get_task_scene_object_names(task)
        if task.subscale == "gross_movement":
            if names != {"mannequin"}:
                raise ValueError(f"{scene_path} must contain only mannequin/nphsfp")
        elif {"table", "arat_box"}.difference(names):
            raise ValueError(f"{scene_path} is missing the resized breakfast table or articulated ARAT box")
        get_task_tro_data(task)
        build_task_metadata(task)

    if missing:
        raise FileNotFoundError("Missing required ARAT runtime assets:\n" + "\n".join(missing))

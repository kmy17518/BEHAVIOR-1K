"""Validate version-1 meshes, collision geometry, URDF dimensions, and masses."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPECS_PATH = ROOT.parent / "arat_components_specs.yaml"
MANIFEST_PATH = ROOT / "validation" / "asset_manifest.json"
REPORT_PATH = ROOT / "validation" / "export_validation.json"
PREVIEW_PATH = ROOT / "previews" / "arat_components_contact_sheet.png"
EXACT_MASSES_KG = {
    "wooden_block_10": 0.492,
    "wooden_block_7_5": 0.196,
    "wooden_block_5_0": 0.055,
    "wooden_block_2_5": 0.0065,
    "cricket_ball": 0.159,
    "sharpening_stone": 0.0603,
    "marble": 0.0054,
    "ball_bearing": 0.0011,
    "small_alloy_tube": 0.0142,
    "large_alloy_tube": 0.0385,
    "washer": 0.016,
    "cup_blue": 0.1254,
    "cup_red": 0.1254,
}


def geometry(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if not loaded.geometry:
        raise AssertionError(f"No geometry in {path}")
    return loaded.to_geometry()


def close(actual, expected, label: str, tolerance: float = 2e-4) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main() -> None:
    specs = yaml.safe_load(SPECS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["asset_count"] != 27 or len(manifest["assets"]) != 27:
        raise AssertionError("Version 1 must contain exactly 27 apparatus models")
    if not (ROOT / "arat_components.blend").is_file() or not PREVIEW_PATH.is_file():
        raise AssertionError("Blender source or preview is missing")

    reports = {}
    collision_count = 0
    model_names = set()
    for asset in manifest["assets"]:
        model = asset["model"]
        if model in model_names:
            raise AssertionError(f"Duplicate model {model}")
        model_names.add(model)
        model_dir = ROOT / "models" / model
        visual_path = model_dir / "meshes" / "visual.obj"
        urdf_path = model_dir / f"{model}.urdf"
        for path in (visual_path, urdf_path):
            if not path.is_file():
                raise AssertionError(f"Missing required export {path}")

        visual = geometry(visual_path)
        extent = np.asarray(asset["bbox_metres"], dtype=float)
        close(visual.bounds[0], np.asarray([-extent[0] / 2, -extent[1] / 2, 0.0]), f"{model} minimum")
        close(visual.bounds[1], np.asarray([extent[0] / 2, extent[1] / 2, extent[2]]), f"{model} maximum")

        tree = ET.parse(urdf_path)
        base_link = tree.find("./link[@name='base_link']")
        if base_link is None:
            raise AssertionError(f"{model} has no base_link")
        mass = float(base_link.find("inertial/mass").attrib["value"])
        if abs(mass - float(asset["mass_kg"])) > 1e-9:
            raise AssertionError(f"{model} URDF / manifest mass mismatch")
        if model in EXACT_MASSES_KG and abs(mass - EXACT_MASSES_KG[model]) > 1e-9:
            raise AssertionError(f"{model} mass must be {EXACT_MASSES_KG[model]} kg, got {mass}")

        colliders = base_link.findall("collision")
        if len(colliders) != asset["collision_mesh_count"]:
            raise AssertionError(f"{model} collision count mismatch")
        for collider in colliders:
            path = model_dir / collider.find("geometry/mesh").attrib["filename"]
            mesh = geometry(path)
            if not mesh.is_watertight or not mesh.is_convex:
                raise AssertionError(f"Collider must be watertight and convex: {path}")
        collision_count += len(colliders)

        if model.startswith("cup_"):
            meta = tree.find("./link[@name='meta__base_link_fillable_0_0_link']")
            if meta is None or not (model_dir / "meshes" / "fillable_volume.obj").is_file():
                raise AssertionError(f"{model} lacks its fillable volume")

        reports[model] = {
            "category": asset["category"],
            "bbox_metres": asset["bbox_metres"],
            "mass_kg": mass,
            "collision_mesh_count": len(colliders),
        }

    required = {
        "tin_lid",
        "tube_starting_fixture",
        "tube_target_fixture",
        "washer_target_fixture",
        "arat_standard_table",
        "arat_standard_shelf",
        "plank_starting_point",
        "plank_target_point",
        "plank_washer_target_point",
        "bolt_large_starting_point",
        "bolt_large_target_point",
        "bolt_small_starting_point",
        "bolt_small_target_point",
        "bolt_washer_target_point",
    }
    if not required <= model_names:
        raise AssertionError(f"Missing required version-1 models: {sorted(required - model_names)}")
    if collision_count != 149:
        raise AssertionError(f"Expected 149 convex collision meshes, got {collision_count}")

    report = {
        "status": "passed",
        "asset_version": 1,
        "authority": specs["project"]["authority"],
        "asset_count": len(reports),
        "collision_mesh_count": collision_count,
        "assets": reports,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ARAT_V1_EXPORT_VALIDATION_PASSED assets={len(reports)} collisions={collision_count}")


if __name__ == "__main__":
    main()

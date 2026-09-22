"""Validate the version-1 ARAT specification and resolve it for Blender.

This is deliberately strict about every dimension and mass stated by the
standardized manual. Values the manual leaves unspecified are documented in
``arat_components_specs.yaml`` instead of being presented as requirements.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPECS_PATH = ROOT.parent / "arat_components_specs.yaml"
RESOLVED_PATH = Path(__file__).resolve().parent / "resolved_specs.json"
TEXTURE_PATH = ROOT / "textures" / "wood_basecolor.png"


def require(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> None:
    specs = yaml.safe_load(SPECS_PATH.read_text(encoding="utf-8"))
    require(specs["project"]["version"], 1, "asset version")
    require(specs["project"]["authority"], "ARAT_specification.md", "authority")

    blocks = {item["model"]: (item["edge"], item["mass_grams"]) for item in specs["assets"]["wooden_blocks"]["models"]}
    require(
        blocks,
        {
            "wooden_block_10": (10.0, 492.0),
            "wooden_block_7_5": (7.5, 196.0),
            "wooden_block_5_0": (5.0, 55.0),
            "wooden_block_2_5": (2.5, 6.5),
        },
        "wooden blocks",
    )
    for key, expected in {
        "cricket_ball": (7.1, 159.0),
        "marble": (1.6, 5.4),
        "ball_bearing": (6.0, 1.1),
    }.items():
        item = specs["assets"][key]
        require((item["diameter"], item["mass_grams"]), expected, key)

    stone = specs["assets"]["sharpening_stone"]
    require((stone["dimensions_xyz"], stone["mass_grams"]), ([10.0, 2.5, 1.0], 60.3), "sharpening stone")
    tin = specs["assets"]["tin_lid"]
    require((tin["outside_diameter"], tin["rim_height"]), (9.0, 1.0), "tin lid")
    require(tin["inside_diameter"], tin["outside_diameter"] - 2 * tin["wall_thickness"], "tin inside diameter")
    require(tin["inside_depth"], tin["rim_height"] - tin["floor_thickness"], "tin inside depth")

    tubes = {item["model"]: item for item in specs["assets"]["alloy_tubes"]["models"]}
    tube_wall_assumptions = {
        "large_alloy_tube": "large_tube_wall_thickness_cm",
        "small_alloy_tube": "small_tube_wall_thickness_cm",
    }
    for model, expected in {
        "large_alloy_tube": (2.5, 11.5, 38.5, 2.0),
        "small_alloy_tube": (1.0, 16.0, 14.2, 0.8),
    }.items():
        item = tubes[model]
        require(
            (item["outside_diameter"], item["height"], item["mass_grams"], item["peg_diameter"]),
            expected,
            model,
        )
        require(
            item["inside_diameter"],
            item["outside_diameter"] - 2 * item["wall_thickness"],
            f"{model} inside diameter",
        )
        require(
            item["wall_thickness"],
            specs["assumptions"][tube_wall_assumptions[model]],
            f"{model} wall thickness assumption",
        )
        if not isinstance(item["collision_axial_segments"], int) or item["collision_axial_segments"] < 1:
            raise ValueError(f"{model} collision_axial_segments must be a positive integer")
        if item["inside_diameter"] <= item["peg_diameter"]:
            raise ValueError(f"{model} must clear its standardized peg")

    fixtures = specs["assets"]
    require(fixtures["tube_starting_fixture"]["plank_dimensions_xyz"], [6.0, 8.5, 1.5], "starting plank")
    require(fixtures["tube_target_fixture"]["plank_dimensions_xyz"], [8.5, 34.0, 3.5], "target plank")
    require(fixtures["washer_target_fixture"]["plank_dimensions_xyz"], [8.5, 8.5, 1.5], "washer plank")
    start_pegs = {peg["id"]: (peg["diameter"], peg["height"]) for peg in fixtures["tube_starting_fixture"]["pegs"]}
    target_pegs = {peg["id"]: (peg["diameter"], peg["height"]) for peg in fixtures["tube_target_fixture"]["pegs"]}
    require(start_pegs, {"large": (2.0, 13.5), "small": (0.8, 6.0)}, "starting pegs")
    require(target_pegs, {"large": (2.0, 8.0), "small": (0.8, 6.0)}, "target pegs")
    require(
        [(peg["diameter"], peg["height"]) for peg in fixtures["washer_target_fixture"]["pegs"]],
        [(0.8, 8.5)],
        "washer peg",
    )
    components = fixtures["fixture_components"]
    require(components["plank_category"], "arat_plank", "independent fixture plank category")
    require(components["bolt_category"], "arat_bolt", "independent fixture bolt category")
    require(
        {item["model"]: item["dimensions_xyz"] for item in components["planks"]},
        {
            "plank_starting_point": [6.0, 8.5, 1.5],
            "plank_target_point": [8.5, 34.0, 3.5],
            "plank_washer_target_point": [8.5, 8.5, 1.5],
        },
        "independent fixture planks",
    )
    require(
        {item["model"]: (item["diameter"], item["height"]) for item in components["bolts"]},
        {
            "bolt_large_starting_point": (2.0, 13.5),
            "bolt_large_target_point": (2.0, 8.0),
            "bolt_small_starting_point": (0.8, 6.0),
            "bolt_small_target_point": (0.8, 6.0),
            "bolt_washer_target_point": (0.8, 8.5),
        },
        "independent fixture bolts",
    )

    washer = specs["assets"]["washer"]
    require(
        (washer["outside_diameter"], washer["inside_diameter"], washer["mass_grams"]),
        (3.5, 1.5, 16.0),
        "washer",
    )
    cups = specs["assets"]["cups"]
    dimensions = cups["dimensions"]
    if not 7.0 <= dimensions["upper_outside_diameter"] <= 8.0:
        raise ValueError("cup upper diameter is outside 7-8 cm")
    if not 6.0 <= dimensions["lower_outside_diameter"] <= 7.0:
        raise ValueError("cup lower diameter is outside 6-7 cm")
    if not 12.0 <= dimensions["height"] <= 15.0:
        raise ValueError("cup height is outside 12-15 cm")
    require(cups["mass_grams"], 125.4, "empty cup mass")

    table = specs["assets"]["table"]
    require(table["overall_dimensions_xyz"], [49.0, 76.0, 75.0], "table depth/width/height")
    shelf = specs["assets"]["shelf"]
    require(shelf["overall_height"], 37.0, "shelf height above table")
    require(shelf["construction"], "purpose_built_open_stand", "shelf construction")
    require(shelf["support_frame_dimensions_xyz"], [23.0, 46.0, 34.0], "shelf support frame")
    require(shelf["top_plank_dimensions_xyz"], [23.0, 46.0, 3.0], "shelf top plank")

    model_count = 4 + 1 + 1 + 1 + 1 + 1 + 2 + 3 + 8 + 1 + 2 + 1 + 1
    require(model_count, 27, "model count")
    if not TEXTURE_PATH.is_file():
        raise FileNotFoundError(TEXTURE_PATH)
    RESOLVED_PATH.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")
    print(f"ARAT_V1_SPECS_PREPARED models={model_count} json={RESOLVED_PATH}")


if __name__ == "__main__":
    main()

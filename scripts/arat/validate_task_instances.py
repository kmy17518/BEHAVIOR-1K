#!/usr/bin/env python3
"""Load-test the ARAT base scene and all 19 instances in ``behavior_dex``."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import ContainedParticles, Filled, OnTop
from omnigibson.utils.sampling_utils import raytest_batch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPOSITORY_ROOT / "datasets" / "arat-assets-v1"
INSTANCE_ROOT = REPOSITORY_ROOT / "datasets" / "arat-task-instances"
BASE_SCENE = ASSET_ROOT / "scenes" / "arat_base" / "json" / "arat_base_best.json"
INSTANCE_JSON_DIR = INSTANCE_ROOT / "scenes" / "arat_base" / "json"
MANIFEST_PATH = INSTANCE_ROOT / "layout_manifest.json"
REPORT_PATH = INSTANCE_ROOT / "validation" / "behavior_dex_load_validation.json"
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
    "plastic_tumbler_1": 0.1254,
    "plastic_tumbler_2": 0.1254,
}
BLOCK_SIDE_LENGTHS = {
    "wooden_block_10": 0.10,
    "wooden_block_7_5": 0.075,
    "wooden_block_5_0": 0.05,
    "wooden_block_2_5": 0.025,
}
WATER_PROXY_PARTICLES = 41
WATER_REQUIRED_ML = 118.29411825
ROBOT_POSITION = [-0.5685013461, -0.1084822643, 0.0156103678]
RESET_JOINT_POSITIONS = [-0.0006, -1.30, 0.0006, -2.87, 0.001, 1.999, 0.749] + [0.0] * 22
ROBOT_SHOULDER_HEIGHT_FROM_ROOT = 1.194
HUMAN_SHOULDER_TO_TABLE = 1.033 - 0.75
REFERENCE_TABLE_TOP_Z = ROBOT_POSITION[2] + ROBOT_SHOULDER_HEIGHT_FROM_ROOT - HUMAN_SHOULDER_TO_TABLE
REFERENCE_TABLE_PROXIMAL_DISTANCE = 0.235
COMMON_TABLE_HEIGHT_INCREASE = 0.10
COMMON_TABLE_DISTANCE_INCREASE = 0.20
TABLE_TOP_Z = REFERENCE_TABLE_TOP_Z + COMMON_TABLE_HEIGHT_INCREASE
TABLE_FRONT_X = ROBOT_POSITION[0] + REFERENCE_TABLE_PROXIMAL_DISTANCE + COMMON_TABLE_DISTANCE_INCREASE
TABLE_BACK_X = TABLE_FRONT_X + 0.75
TABLE_CENTER_Y = ROBOT_POSITION[1]
LAYOUT_TRANSLATION_X = TABLE_FRONT_X - (-0.245)
LAYOUT_TRANSLATION_Y = TABLE_CENTER_Y
LAYOUT_TRANSLATION_Z = TABLE_TOP_Z - 0.75
BOX_ROOT = [
    0.1352 + LAYOUT_TRANSLATION_X,
    0.27305 + LAYOUT_TRANSLATION_Y,
    0.7771 + LAYOUT_TRANSLATION_Z,
]
BOX_COVER_THICKNESS = 0.017462
BOX_COVER_JOINT_DROP = 0.0271 - BOX_COVER_THICKNESS
BOX_COVER_SLOPE_X = 0.0
BOX_RIGHT_Y = -0.27305 + LAYOUT_TRANSLATION_Y
BOX_CENTER_Y = TABLE_CENTER_Y
BOX_LID_FRONT_X = -0.195 + LAYOUT_TRANSLATION_X
BOX_LID_BACK_X = 0.1352 + LAYOUT_TRANSLATION_X
BOX_LID_TOP_Z = 0.7771 + LAYOUT_TRANSLATION_Z - BOX_COVER_JOINT_DROP
BOX_LID_COLLISION_SURFACE_OFFSET_Z = 0.0
BOX_BASE_TOP_Z = 1.12 + LAYOUT_TRANSLATION_Z
TASK_CENTER_Y = -0.17305 + LAYOUT_TRANSLATION_Y
TIN_LID_RADIUS = 0.045
TIN_LID_HEIGHT = 0.01
TIN_LID_1_PITCH = -math.atan(BOX_COVER_SLOPE_X)
TIN_LID_1_ORIENTATION = [0.0, math.sin(TIN_LID_1_PITCH / 2.0), 0.0, math.cos(TIN_LID_1_PITCH / 2.0)]
TIN_LID_1_FRONT_OFFSET_X = (
    TIN_LID_RADIUS * math.cos(TIN_LID_1_PITCH) - TIN_LID_HEIGHT * math.sin(TIN_LID_1_PITCH)
)
TIN_LID_1_ROOT_X = TABLE_FRONT_X + 0.05 + TIN_LID_1_FRONT_OFFSET_X
TIN_LID_1_AABB_CENTER_X = TIN_LID_1_ROOT_X + TIN_LID_HEIGHT * math.sin(TIN_LID_1_PITCH) / 2.0
TIN_LID_1_ROOT_Z = (
    BOX_LID_TOP_Z
    + BOX_LID_COLLISION_SURFACE_OFFSET_Z
    + BOX_COVER_SLOPE_X * (TIN_LID_1_ROOT_X - BOX_ROOT[0])
)
TIN_FIXED_SUPPORT_CLEARANCE = 0.0015


def horizontal_cover_support_z(center_x: float, half_extent_x: float) -> float:
    highest_contact_x = center_x + half_extent_x
    return BOX_LID_TOP_Z + BOX_COVER_SLOPE_X * (highest_contact_x - BOX_ROOT[0])


TIN_LID_1_CENTER = [
    TIN_LID_1_ROOT_X,
    TASK_CENTER_Y,
    TIN_LID_1_ROOT_Z + TIN_FIXED_SUPPORT_CLEARANCE,
]
TIN_LID_2_CENTER = [
    0.1802 + LAYOUT_TRANSLATION_X,
    TASK_CENTER_Y,
    1.12 + LAYOUT_TRANSLATION_Z + TIN_FIXED_SUPPORT_CLEARANCE,
]
STONE_ROOT = [
    -0.14609261996306222 + LAYOUT_TRANSLATION_X,
    -0.16815926199630624 + LAYOUT_TRANSLATION_Y,
    0.7901 + LAYOUT_TRANSLATION_Z - BOX_COVER_JOINT_DROP,
]
STONE_CENTER = [
    -0.14505306150908986 + LAYOUT_TRANSLATION_X,
    TASK_CENTER_Y,
    0.7901 + LAYOUT_TRANSLATION_Z - BOX_COVER_JOINT_DROP,
]
STONE_ORIENTATION = [0.7032331763, 0.0739127852, 0.0739127852, 0.7032331763]
CUP_1_CENTER = [
    -0.1075 + LAYOUT_TRANSLATION_X,
    0.1175 + LAYOUT_TRANSLATION_Y,
    0.7776 + LAYOUT_TRANSLATION_Z - BOX_COVER_JOINT_DROP,
]
CUP_2_CENTER = [
    -0.1075 + LAYOUT_TRANSLATION_X,
    -0.1175 + LAYOUT_TRANSLATION_Y,
    0.7776 + LAYOUT_TRANSLATION_Z - BOX_COVER_JOINT_DROP,
]
START_CENTER = [-0.165 + LAYOUT_TRANSLATION_X, TASK_CENTER_Y]
TARGET_PLANK_FORWARD_SHIFT = 0.020
TARGET_PLANK_CENTER = [
    0.0927 + LAYOUT_TRANSLATION_X - TARGET_PLANK_FORWARD_SHIFT,
    -0.10305 + LAYOUT_TRANSLATION_Y,
]
TARGET_BOLT_CENTER = [0.0927 + LAYOUT_TRANSLATION_X - TARGET_PLANK_FORWARD_SHIFT, TASK_CENTER_Y]
CONTACT_CLEARANCE = 0.0005
FIXED_SUPPORT_CLEARANCE = 0.0
MANNEQUIN_TARGET_HEIGHT = 1.1454 + LAYOUT_TRANSLATION_Z
MANNEQUIN_TARGET_CENTER_XY = [
    0.03598145395517349 + LAYOUT_TRANSLATION_X,
    2.9802322387695312e-08 + LAYOUT_TRANSLATION_Y,
]
MANNEQUIN_ORIENTATION = [-2**-0.5, 0.0, 2**-0.5, 0.0]
MANNEQUIN_SCALE = 0.6519119810200881 * MANNEQUIN_TARGET_HEIGHT / 1.1454
MANNEQUIN_NATIVE_COLLISION_AABB_CENTER = [
    0.07910473742101737,
    1.4772632368931227e-05,
    0.016168891737376692,
]
MANNEQUIN_NATIVE_COLLISION_AABB_EXTENT_X = 1.75698565657241
MANNEQUIN_ROOT = [
    MANNEQUIN_TARGET_CENTER_XY[0] + MANNEQUIN_SCALE * MANNEQUIN_NATIVE_COLLISION_AABB_CENTER[2],
    MANNEQUIN_TARGET_CENTER_XY[1] + MANNEQUIN_SCALE * MANNEQUIN_NATIVE_COLLISION_AABB_CENTER[1],
    MANNEQUIN_SCALE
    * (MANNEQUIN_NATIVE_COLLISION_AABB_CENTER[0] + MANNEQUIN_NATIVE_COLLISION_AABB_EXTENT_X / 2.0),
]

GROSS_MOVEMENT_ACTIVITIES = {
    "arat_gross_movement_hand_behind_head",
    "arat_gross_movement_hand_top_head",
    "arat_gross_movement_hand_mouth",
}

TRANSFER_ACTIVITIES = {
    "arat_grasp_cricket_ball": "cricket_ball",
    "arat_pinch_ball_bearing_ring": "ball_bearing",
    "arat_pinch_marble_index": "marble",
    "arat_pinch_ball_bearing_middle": "ball_bearing",
    "arat_pinch_ball_bearing_index": "ball_bearing",
    "arat_pinch_marble_ring": "marble",
    "arat_pinch_marble_middle": "marble",
}


def close(actual, expected, label: str, tolerance: float = 1e-4) -> None:
    if not th.allclose(
        th.as_tensor(actual, dtype=th.float64),
        th.as_tensor(expected, dtype=th.float64),
        rtol=0.0,
        atol=tolerance,
    ):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def static_validation() -> tuple[dict, list[Path]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if len(manifest["tasks"]) != 19:
        raise AssertionError("Expected 19 task layouts")
    base_data = json.loads(BASE_SCENE.read_text(encoding="utf-8"))
    expected_scene_args = {
        "use_floor_plane": True,
        "floor_plane_visible": True,
        "floor_plane_color": [0.5, 0.5, 0.5],
        "use_skybox": True,
    }
    if base_data["init_info"]["class_name"] != "Scene":
        raise AssertionError("The ARAT base is not a plain Scene")
    for key, expected in expected_scene_args.items():
        if base_data["init_info"]["args"].get(key) != expected:
            raise AssertionError(f"The ARAT base has the wrong {key}")
    base_names = set(base_data["objects_info"]["init_info"])
    if base_names != {"table", "arat_task_light_near", "arat_task_light_far"}:
        raise AssertionError(f"Unexpected base-scene objects: {base_names}")

    paths = []
    for task in manifest["tasks"]:
        activity = task["activity"]
        path = INSTANCE_JSON_DIR / f"arat_base_task_{task['activity']}_0_0_template.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["init_info"]["class_name"] != "Scene":
            raise AssertionError(f"{path.name} is not a plain Scene")
        for key, expected in expected_scene_args.items():
            if data["init_info"]["args"].get(key) != expected:
                raise AssertionError(f"{path.name} has the wrong {key}")
        for obj_info in data["objects_info"]["init_info"].values():
            if obj_info["class_name"] == "DatasetObject" and obj_info["args"]["dataset_name"] != "arat-assets-v1":
                raise AssertionError(f"{path.name} contains an object outside version 1")
        saved_names = set(data["objects_info"]["init_info"])
        expected_names = {obj["name"] for obj in task["objects"]}
        if activity not in GROSS_MOVEMENT_ACTIVITIES:
            expected_names.update({"arat_task_light_near", "arat_task_light_far"})
        if saved_names != expected_names:
            raise AssertionError(f"{path.name} object mismatch: expected {expected_names}, got {saved_names}")
        if activity in GROSS_MOVEMENT_ACTIVITIES:
            if saved_names != {"mannequin"}:
                raise AssertionError(f"{path.name} gross-movement scene must contain only mannequin/nphsfp")
        elif "arat_box" not in saved_names:
            raise AssertionError(f"{path.name} is missing arat_box/aratbx")

        task_objects = {obj["name"]: obj for obj in task["objects"]}
        if activity.startswith("arat_grasp_block_"):
            moved_name = next(name for name in task_objects if name in BLOCK_SIDE_LENGTHS)
            requested_names = {"table", "arat_box", moved_name}
            movable_names = {moved_name}
        elif activity in TRANSFER_ACTIVITIES:
            moved_name = TRANSFER_ACTIVITIES[activity]
            requested_names = {"table", "arat_box", "tin_lid_1", "tin_lid_2", moved_name}
            movable_names = {moved_name}
        elif activity == "arat_grasp_sharpening_stone":
            requested_names = {"table", "arat_box", "sharpening_stone"}
            movable_names = {"sharpening_stone"}
        elif activity == "arat_grip_pour_water":
            requested_names = {"table", "arat_box", "plastic_tumbler_1", "plastic_tumbler_2"}
            movable_names = {"plastic_tumbler_1", "plastic_tumbler_2"}
        elif activity.startswith("arat_grip_alloy_tube_"):
            moved_name = next(name for name in task_objects if name.endswith("alloy_tube"))
            requested_names = {
                "table",
                "arat_box",
                "plank_starting_point",
                "bolt_starting_point",
                "plank_target_point",
                "bolt_target_point",
                moved_name,
            }
            movable_names = {moved_name}
        elif activity == "arat_grip_washer_over_bolt":
            requested_names = {
                "table",
                "arat_box",
                "tin_lid",
                "washer",
                "plank_target_point",
                "bolt_target_point",
            }
            movable_names = {"washer"}
        elif activity in GROSS_MOVEMENT_ACTIVITIES:
            requested_names = {"mannequin"}
            movable_names = set()
        else:
            raise AssertionError(f"Unknown ARAT activity {activity}")
        if set(task_objects) != requested_names:
            raise AssertionError(f"{activity} requested object mismatch: {set(task_objects)} != {requested_names}")
        if (
            activity.startswith("arat_grip_alloy_tube_")
            and task_objects["plank_starting_point"]["model"] != "plank_starting_point"
        ):
            raise AssertionError(f"{activity} does not use the 6 cm tube starting plank")
        if (
            activity == "arat_grip_washer_over_bolt"
            and task_objects["plank_target_point"]["model"] != "plank_washer_target_point"
        ):
            raise AssertionError("Washer task does not use its independent 8.5 cm-deep target plank")
        actual_movable = {name for name, obj in task_objects.items() if not obj["fixed_base"]}
        if actual_movable != movable_names:
            raise AssertionError(f"{activity} movable object mismatch: {actual_movable} != {movable_names}")

        for obj in task["objects"]:
            saved_position = data["state"]["registry"]["object_registry"][obj["name"]]["root_link"]["pos"]
            close(saved_position, obj["position_xyz_m"], f"{path.name}:{obj['name']} saved start", tolerance=5e-5)
        paths.append(path)

    table_constraints = manifest["table"]
    close(
        [table_constraints["depth_m"], table_constraints["width_m"], table_constraints["height_m"]],
        [0.75, 0.76, TABLE_TOP_Z],
        "table dimensions",
    )
    close(
        [table_constraints["proximal_edge_world_x_m"], table_constraints["distal_edge_world_x_m"]],
        [TABLE_FRONT_X, TABLE_BACK_X],
        "table X edges",
    )
    close(table_constraints["center_world_y_m"], TABLE_CENTER_Y, "table lateral center")
    close(
        table_constraints["proximal_edge_distance_from_robot_m"],
        REFERENCE_TABLE_PROXIMAL_DISTANCE + COMMON_TABLE_DISTANCE_INCREASE,
        "robot-to-table proximal-edge distance",
    )
    close(
        table_constraints["height_increase_from_reference_m"],
        COMMON_TABLE_HEIGHT_INCREASE,
        "table height increase from reference",
    )
    close(
        table_constraints["distance_increase_from_reference_m"],
        COMMON_TABLE_DISTANCE_INCREASE,
        "table distance increase from reference",
    )
    ergonomics = manifest["ergonomic_alignment"]
    close(ergonomics["robot_position_xyz_m"], ROBOT_POSITION, "ergonomic robot position")
    close(ergonomics["shoulder_to_table_offset_m"], HUMAN_SHOULDER_TO_TABLE, "shoulder-table offset")
    close(
        ergonomics["realized_robot_shoulder_to_table_offset_m"],
        HUMAN_SHOULDER_TO_TABLE - COMMON_TABLE_HEIGHT_INCREASE,
        "realized robot shoulder-table offset",
    )
    if ergonomics["robot_pose_changed"]:
        raise AssertionError("The ergonomic layout must not change the robot pose")
    box_constraints = manifest["target_box"]
    if (box_constraints["category"], box_constraints["model"]) != ("arat_box", "aratbx"):
        raise AssertionError("Wrong ARAT box identity")
    close(box_constraints["root_xyz_m"], BOX_ROOT, "box root")
    close(box_constraints["cover_proximal_edge_world_x_m"], BOX_LID_FRONT_X, "box lid front edge")
    close(box_constraints["cover_distal_edge_world_x_m"], BOX_LID_BACK_X, "box lid back edge")
    close(box_constraints["cover_proximal_edge_from_table_proximal_edge_m"], 0.05, "box A-to-B")
    close(box_constraints["center_world_y_m"], BOX_CENTER_Y, "box horizontal center")
    close(box_constraints["rightmost_edge_world_y_m"], BOX_RIGHT_Y, "box right edge")
    close(box_constraints["shell_surface_height_above_table_m"], 0.37, "box base height")
    close(box_constraints["back_support_margin_m"], 0.2380375, "box rear support margin")
    close(box_constraints["cover_joint_zero_angle_deg"], 0.0, "open cover joint angle")
    close(box_constraints["cover_joint_drop_m"], BOX_COVER_JOINT_DROP, "open cover joint drop")
    close(box_constraints["cover_bottom_clearance_above_table_m"], 0.0, "open cover table clearance")
    close(box_constraints["cover_surface_slope_deg"], 0.0, "open cover surface slope")

    block_placement = manifest["block_placement"]
    close(block_placement["front_edge_from_table_proximal_edge_m"], 0.05, "block front-edge constraint")
    close(block_placement["box_rightmost_edge_world_y_m"], BOX_RIGHT_Y, "box right-edge datum")
    close(block_placement["centroid_from_box_rightmost_edge_m"], 0.10, "block lateral constraint")
    close(block_placement["centroid_world_y_m"], TASK_CENTER_Y, "block centroid world Y")
    for task in manifest["tasks"][:4]:
        block = next(obj for obj in task["objects"] if obj["name"] in BLOCK_SIDE_LENGTHS)
        side_length = BLOCK_SIDE_LENGTHS[block["name"]]
        expected_x = TABLE_FRONT_X + 0.05 + side_length / 2.0
        close(
            block["position_xyz_m"],
            [expected_x, TASK_CENTER_Y, BOX_LID_TOP_Z + CONTACT_CLEARANCE],
            f"{block['name']} authored position",
        )

    common_adjustment = manifest["common_layout_adjustment"]
    close(
        common_adjustment["height_increase_m"],
        COMMON_TABLE_HEIGHT_INCREASE,
        "common table-height increase",
    )
    close(
        common_adjustment["distance_increase_from_robot_m"],
        COMMON_TABLE_DISTANCE_INCREASE,
        "common robot-to-table distance increase",
    )
    close(
        common_adjustment["robot_to_table_proximal_edge_distance_m"],
        REFERENCE_TABLE_PROXIMAL_DISTANCE + COMMON_TABLE_DISTANCE_INCREASE,
        "common robot-to-table proximal-edge distance",
    )
    if (
        not common_adjustment["table_feet_remain_on_floor"]
        or common_adjustment["robot_pose_changed"]
        or not common_adjustment["applies_to_all_table_tasks"]
    ):
        raise AssertionError("The common layout adjustment must preserve floor contact and the robot pose")

    cricket_placement = manifest["tin_transfer_placement"]
    close(
        cricket_placement["tin_lid_1_front_edge_from_table_proximal_edge_m"],
        0.05,
        "cricket tin_lid_1 front-edge constraint",
    )
    close(cricket_placement["tin_lid_1_center_xyz_m"], TIN_LID_1_CENTER, "tin_lid_1 position")
    close(
        cricket_placement["tin_lid_1_orientation_xyzw"],
        TIN_LID_1_ORIENTATION,
        "tin_lid_1 cover-aligned orientation",
    )
    close(cricket_placement["tin_lid_2_front_edge_world_x_m"], BOX_ROOT[0], "tin_lid_2 front edge")
    close(cricket_placement["tin_lid_2_front_edge_from_box_top_front_edge_m"], 0.0, "tin_lid_2 edge contact")
    close(cricket_placement["tin_lid_2_center_xyz_m"], TIN_LID_2_CENTER, "tin_lid_2 position")
    close(
        cricket_placement["fixed_support_clearance_m"],
        TIN_FIXED_SUPPORT_CLEARANCE,
        "tin fixed-support clearance",
    )
    close(cricket_placement["box_rightmost_edge_world_y_m"], BOX_RIGHT_Y, "cricket box right-edge datum")
    close(
        cricket_placement["lid_centroids_from_box_rightmost_edge_m"],
        0.10,
        "cricket lid lateral constraint",
    )
    cricket_task = next(task for task in manifest["tasks"] if task["activity"] == "arat_grasp_cricket_ball")
    cricket_objects = {obj["name"]: obj for obj in cricket_task["objects"]}
    expected_cricket_names = {"table", "arat_box", "tin_lid_1", "tin_lid_2", "cricket_ball"}
    if set(cricket_objects) != expected_cricket_names:
        raise AssertionError(f"Cricket-ball layout object mismatch: {set(cricket_objects)}")
    close(
        cricket_objects["cricket_ball"]["position_xyz_m"],
        [
            TIN_LID_1_AABB_CENTER_X,
            TASK_CENTER_Y,
            TIN_LID_1_ROOT_Z + TIN_FIXED_SUPPORT_CLEARANCE + 0.0008 + CONTACT_CLEARANCE,
        ],
        "cricket ball position",
    )

    stone_placement = manifest["sharpening_stone_placement"]
    close(stone_placement["front_edge_from_table_proximal_edge_m"], 0.05, "stone front-edge constraint")
    close(stone_placement["box_rightmost_edge_world_y_m"], BOX_RIGHT_Y, "stone box right-edge datum")
    close(stone_placement["centroid_from_box_rightmost_edge_m"], 0.10, "stone lateral constraint")
    close(stone_placement["centroid_world_xyz_m"], STONE_CENTER, "stone centroid")
    close(stone_placement["root_world_xyz_m"], STONE_ROOT, "stone root")
    close(stone_placement["orientation_xyzw"], STONE_ORIENTATION, "stone orientation")
    if stone_placement["supported_by"] != "arat_box lid":
        raise AssertionError("Sharpening stone must be supported by the ARAT box lid")
    stone_task = next(task for task in manifest["tasks"] if task["activity"] == "arat_grasp_sharpening_stone")
    stone = next(obj for obj in stone_task["objects"] if obj["name"] == "sharpening_stone")
    close(stone["position_xyz_m"], stone_placement["root_world_xyz_m"], "authored stone root")
    close(stone["orientation_xyzw"], stone_placement["orientation_xyzw"], "authored stone orientation")

    pouring = manifest["pouring_placement"]
    close(pouring["box_horizontal_midline_world_y_m"], BOX_CENTER_Y, "pour box midline")
    close(pouring["tumbler_1_center_xyz_m"], CUP_1_CENTER, "tumbler 1 center")
    close(pouring["tumbler_2_center_xyz_m"], CUP_2_CENTER, "tumbler 2 center")
    close(pouring["inner_edges_from_midline_m"], 0.08, "tumbler inner-edge offsets")
    close(pouring["front_edges_from_table_proximal_edge_m"], 0.10, "tumbler front offsets")

    tube = manifest["tube_placement"]
    close(tube["plank_starting_dimensions_depth_width_height_m"], [0.06, 0.085, 0.015], "start plank dimensions")
    close(tube["plank_target_dimensions_depth_width_height_m"], [0.085, 0.34, 0.035], "target plank dimensions")
    close(tube["start_bolt_center_xy_m"], START_CENTER, "start bolt center")
    close(tube["start_plank_front_edge_from_table_proximal_edge_m"], 0.05, "start plank front offset")
    close(tube["start_plank_front_overhang_beyond_lid_m"], 0.0, "start plank lid overhang")
    close(tube["target_bolt_center_xy_m"], TARGET_BOLT_CENTER, "target bolt center")
    close(
        tube["target_plank_forward_shift_from_lid_back_edge_m"],
        TARGET_PLANK_FORWARD_SHIFT,
        "target plank forward shift",
    )
    close(tube["large_bolts_diameter_start_height_target_height_m"], [0.02, 0.135, 0.08], "large bolts")
    close(tube["small_bolts_diameter_start_height_target_height_m"], [0.008, 0.06, 0.06], "small bolts")
    if (
        tube["target_plank_back_edge_aligned_with_lid_back_edge"]
        or not tube["target_plank_clears_black_hinge"]
        or not tube["target_plank_right_edge_aligned_with_box_rightmost_edge"]
        or not tube["tube_initially_pegged_on_start_bolt"]
    ):
        raise AssertionError("Tube fixture alignment metadata is incomplete")

    washer = manifest["washer_placement"]
    close(washer["tin_lid_front_edge_from_table_proximal_edge_m"], 0.05, "washer lid A-to-B")
    close(washer["tin_lid_centroid_from_box_rightmost_edge_m"], 0.10, "washer lid X-to-C")
    close(washer["plank_target_dimensions_depth_width_height_m"], [0.085, 0.085, 0.015], "washer plank")
    close(washer["bolt_target_diameter_height_m"], [0.008, 0.085], "washer bolt")
    close(washer["bolt_target_center_xy_m"], TARGET_BOLT_CENTER, "washer bolt center")
    close(
        washer["plank_target_forward_shift_from_lid_back_edge_m"],
        TARGET_PLANK_FORWARD_SHIFT,
        "washer target plank forward shift",
    )
    if (
        washer["plank_target_back_edge_aligned_with_lid_back_edge"]
        or not washer["plank_target_clears_black_hinge"]
    ):
        raise AssertionError("Washer target plank does not clear the black hinge")

    gross = manifest["gross_movement_placement"]
    if set(gross["activities"]) != GROSS_MOVEMENT_ACTIVITIES:
        raise AssertionError("Gross-movement activity list is incomplete")
    if gross["only_scene_object"] != "mannequin/nphsfp" or not gross["fixed_base"]:
        raise AssertionError("Gross-movement scenes must contain only fixed mannequin/nphsfp")
    close(gross["target_height_m"], MANNEQUIN_TARGET_HEIGHT, "mannequin target height")
    close(gross["target_world_xy_aabb_center_m"], MANNEQUIN_TARGET_CENTER_XY, "mannequin target XY center")
    close(gross["root_xyz_m"], MANNEQUIN_ROOT, "mannequin root")
    close(gross["orientation_xyzw"], MANNEQUIN_ORIENTATION, "mannequin orientation")
    close(gross["uniform_scale"], MANNEQUIN_SCALE, "mannequin scale")
    if not gross["head_at_top"]:
        raise AssertionError("Mannequin is not declared head-up")
    return manifest, paths


def aabb_center(obj):
    return (obj.aabb[0] + obj.aabb[1]) / 2.0


def tin_box_collision_overlaps(lid, box) -> list[str]:
    """Return box rigid-body paths that overlap any compound tin collider."""

    box_link_paths = set(box.link_prim_paths)
    overlaps = []

    def overlap_callback(hit):
        if hit.rigid_body in box_link_paths:
            overlaps.append(hit.rigid_body)
            return False
        return True

    for collision_mesh in lid.root_link.collision_meshes.values():
        prim = collision_mesh.prim
        encoded_path = lazy.pxr.PhysicsSchemaTools.encodeSdfPath(prim.GetPrimPath())
        if prim.GetTypeName() == "Mesh":
            og.sim.psqi.overlap_mesh(*encoded_path, reportFn=overlap_callback)
        else:
            og.sim.psqi.overlap_shape(*encoded_path, reportFn=overlap_callback)
        if overlaps:
            break
    return overlaps


def validate_tin_collision_clearance(lid, box, path_name: str, sloped: bool) -> None:
    """Verify the tin's collision underside clears the box over its footprint."""

    lower, upper = lid.aabb
    center = aabb_center(lid)
    root_position, _ = lid.get_position_orientation()
    starts = []
    ends = []
    underside_zs = []
    for ix in range(-8, 9):
        for iy in range(-8, 9):
            dx = TIN_LID_RADIUS * ix / 8.0
            dy = TIN_LID_RADIUS * iy / 8.0
            if dx * dx + dy * dy > (TIN_LID_RADIUS * 0.98) ** 2:
                continue
            x = center[0] + dx
            y = center[1] + dy
            underside_z = (
                root_position[2] + BOX_COVER_SLOPE_X * (x - root_position[0])
                if sloped
                else root_position[2]
            )
            underside_zs.append(float(underside_z))
            starts.append([x, y, underside_z + 0.02])
            ends.append([x, y, underside_z - 0.05])
    results = raytest_batch(
        th.as_tensor(starts, dtype=lower.dtype, device=lower.device),
        th.as_tensor(ends, dtype=lower.dtype, device=lower.device),
        only_closest=False,
        ignore_bodies=lid.link_prim_paths,
        ignore_collisions=lid.link_prim_paths,
    )
    box_links = set(box.link_prim_paths)
    clearances = []
    for underside_z, hits in zip(underside_zs, results):
        all_box_hits = [
            float(hit["position"][2])
            for hit in hits
            if hit["rigidBody"] in box_links
        ]
        intrusions = [hit_z - underside_z for hit_z in all_box_hits if hit_z > underside_z + 1e-5]
        if intrusions:
            raise AssertionError(
                f"{path_name}:{lid.name} box collision protrudes into the tin: "
                f"max={max(intrusions):.6f} m"
            )
        support_hits = [hit_z for hit_z in all_box_hits if hit_z <= underside_z + 1e-5]
        if not support_hits:
            raise AssertionError(f"{path_name}:{lid.name} has no box support below part of its footprint")
        clearances.append(underside_z - max(support_hits))
    minimum = min(clearances)
    maximum = max(clearances)
    if minimum < 0.0005:
        raise AssertionError(
            f"{path_name}:{lid.name} collision penetrates or lacks visible clearance: min={minimum:.6f} m"
        )
    if maximum > 0.0020:
        raise AssertionError(
            f"{path_name}:{lid.name} is too far above its support for OnTop: max={maximum:.6f} m"
        )

    # Ray sampling quantifies the separation, while PhysX overlap queries
    # exhaustively test every collider in the compound tin asset (floor plus
    # sixteen rim wedges) against the articulated box collision shapes.
    overlaps = tin_box_collision_overlaps(lid, box)
    if overlaps:
        raise AssertionError(
            f"{path_name}:{lid.name} has exact PhysX collider overlap with {overlaps[0]}"
        )


def validate_tin_transfer(scene, table, box, path_name: str, moved_name: str, support_relations: list) -> None:
    lid_1 = scene.object_registry("name", "tin_lid_1")
    lid_2 = scene.object_registry("name", "tin_lid_2")
    moved = scene.object_registry("name", moved_name)
    if lid_1 is None or lid_2 is None or moved is None:
        raise AssertionError(f"{path_name}: incomplete two-tin transfer layout")
    close(lid_1.get_position_orientation()[0], TIN_LID_1_CENTER, f"{path_name}:tin_lid_1 root", tolerance=5e-4)
    close(lid_2.get_position_orientation()[0], TIN_LID_2_CENTER, f"{path_name}:tin_lid_2 root", tolerance=5e-4)
    close(lid_1.aabb[0][0] - table.aabb[0][0], 0.05, f"{path_name}:tin_lid_1 A-to-B", tolerance=5e-4)
    _, lid_1_orientation = lid_1.get_position_orientation()
    expected_lid_1_orientation = th.tensor(
        TIN_LID_1_ORIENTATION,
        dtype=lid_1_orientation.dtype,
        device=lid_1_orientation.device,
    )
    if 1.0 - abs(float(th.dot(lid_1_orientation, expected_lid_1_orientation))) > 1e-5:
        raise AssertionError(f"{path_name}:tin_lid_1 is not aligned to the sloped box cover")
    validate_tin_collision_clearance(lid_1, box, path_name, sloped=True)
    validate_tin_collision_clearance(lid_2, box, path_name, sloped=False)
    for lid in (lid_1, lid_2):
        close(
            aabb_center(lid)[1] - box.links["base_link"].aabb[0][1],
            0.10,
            f"{path_name}:{lid.name} X-to-C",
            tolerance=5e-4,
        )
        if not lid.fixed_base:
            raise AssertionError(f"{path_name}:{lid.name} is not fixed")
    close(
        lid_2.aabb[0][0],
        box.links["base_link"].aabb[0][0],
        f"{path_name}:tin_lid_2 front-edge contact",
        tolerance=5e-4,
    )
    close(
        lid_2.aabb[0][2] - box.links["base_link"].aabb[1][2],
        TIN_FIXED_SUPPORT_CLEARANCE,
        f"{path_name}:tin_lid_2 on base top",
        tolerance=5e-4,
    )
    support_relations.extend(((lid_1, box), (lid_2, box), (moved, lid_1)))
    lid_center = aabb_center(lid_1)
    moved_center = aabb_center(moved)
    close(moved_center[:2], lid_center[:2], f"{path_name}:{moved_name} centered in tin_lid_1", tolerance=5e-4)
    if moved.fixed_base:
        raise AssertionError(f"{path_name}:{moved_name} should be movable")
    inner_radius = 0.0442
    if th.any(moved.aabb[0][:2] < lid_center[:2] - inner_radius) or th.any(
        moved.aabb[1][:2] > lid_center[:2] + inner_radius
    ):
        raise AssertionError(f"{path_name}:{moved_name} is outside tin_lid_1's inner footprint")


def runtime_validation(paths: list[Path]) -> list[dict]:
    gm.USE_GPU_DYNAMICS = True
    og.launch()
    all_paths = [BASE_SCENE, *paths]
    results = []
    for index, path in enumerate(all_paths):
        # A full clear is required when consecutive templates reuse an object
        # name with a different model (the large and small tube bolts do this).
        # Simulator.restore otherwise retains the first object instance.
        if og.sim.scenes:
            og.clear()
        og.sim.restore(scene_files=[str(path)])
        scene = og.sim.scenes[0]
        support_relations = []
        expected = json.loads(path.read_text(encoding="utf-8"))
        expected_names = set(expected["objects_info"]["init_info"])
        loaded_names = set(scene.object_registry.get_dict("name"))
        if loaded_names != expected_names:
            raise AssertionError(f"{path.name} loaded {loaded_names}, expected {expected_names}")
        if scene.__class__.__name__ != "Scene":
            raise AssertionError(f"{path.name} loaded as {scene.__class__.__name__}")
        if og.sim.floor_plane is None or og.sim.skybox is None:
            raise AssertionError(f"{path.name} is missing the floor or skybox")
        if any("chair" in name for name in loaded_names):
            raise AssertionError(f"Chair found in {path.name}")

        activity = (
            path.name.removeprefix("arat_base_task_").removesuffix("_0_0_template.json") if index > 0 else None
        )
        is_gross = activity in GROSS_MOVEMENT_ACTIVITIES
        table = scene.object_registry("name", "table")
        if is_gross:
            if loaded_names != {"mannequin"}:
                raise AssertionError(f"{path.name} must contain only mannequin/nphsfp")
        else:
            if table is None or table.category != "breakfast_table" or table.model != "nvoqyl":
                raise AssertionError(f"{path.name} did not load breakfast_table/nvoqyl")
            close(
                table.aabb_extent,
                [0.75, 0.76, TABLE_TOP_Z],
                f"{path.name}:table AABB",
                tolerance=5e-4,
            )
            close(table.aabb[0][2], 0.0, f"{path.name}:table feet at floor", tolerance=5e-4)
            close(
                table.aabb[1][2],
                TABLE_TOP_Z,
                f"{path.name}:table surface",
                tolerance=5e-4,
            )
            close(
                table.aabb[0][0],
                TABLE_FRONT_X,
                f"{path.name}:table proximal edge",
                tolerance=5e-4,
            )
        if index == 0:
            if loaded_names != {"table", "arat_task_light_near", "arat_task_light_far"}:
                raise AssertionError("Base scene must contain only the desk and the two prior-setup lights")
            box = None
            cover = None
        elif is_gross:
            box = None
            cover = None
        elif not is_gross:
            box = scene.object_registry("name", "arat_box")
            if box is None or box.category != "arat_box" or box.model != "aratbx":
                raise AssertionError(f"{path.name} is missing arat_box/aratbx")
            if set(box.joints) != {"front_cover_joint", "top_handle_joint"}:
                raise AssertionError(f"{path.name} loaded unexpected box joints: {set(box.joints)}")
            if not box.fixed_base:
                raise AssertionError(f"{path.name}:ARAT box is not fixed")
            box.joints["front_cover_joint"].friction = 20.0
            for joint in box.joints.values():
                joint.set_pos(0.0)
                joint.set_vel(0.0)
            og.sim.render()
            close(box.get_position_orientation()[0], BOX_ROOT, f"{path.name}:box root", tolerance=5e-4)
            cover = box.links["front_cover_link"]
            base = box.links["base_link"]
            close(
                box.joints["front_cover_joint"].get_state()[0],
                [0.0],
                f"{path.name}:box cover joint zero angle",
                tolerance=1e-5,
            )
            close(
                cover.aabb[0][2] - table.aabb[1][2],
                0.0,
                f"{path.name}:box cover underside flush with table",
                tolerance=5e-4,
            )
            _, box_orientation = box.get_position_orientation()
            _, cover_orientation = cover.get_position_orientation()
            if 1.0 - abs(float(th.dot(box_orientation, cover_orientation))) > 1e-5:
                raise AssertionError(f"{path.name}:box cover is not horizontal at joint position 0")
            close(cover.aabb[0][0] - table.aabb[0][0], 0.05, f"{path.name}:box A-to-B", tolerance=5e-4)
            close(
                (base.aabb[0][1] + base.aabb[1][1]) / 2.0,
                BOX_CENTER_Y,
                f"{path.name}:box horizontal center",
                tolerance=5e-4,
            )
            close(base.aabb[0][1], BOX_RIGHT_Y, f"{path.name}:box right edge", tolerance=5e-4)
            support_relations.append((box, table))
            close(base.aabb[1][2] - table.aabb[1][2], 0.37, f"{path.name}:box base height", tolerance=5e-4)
            close(
                table.aabb[1][0] - base.aabb[1][0],
                0.2380375,
                f"{path.name}:box rear support margin",
                tolerance=5e-4,
            )

        for block_name in BLOCK_SIDE_LENGTHS:
            block = scene.object_registry("name", block_name)
            if block is not None:
                support_relations.append((block, box))
                close(block.aabb[0][0] - table.aabb[0][0], 0.05, f"{path.name}:{block_name} A-to-B", tolerance=5e-4)
                close(
                    aabb_center(block)[1] - box.links["base_link"].aabb[0][1],
                    0.10,
                    f"{path.name}:{block_name} X-to-C",
                    tolerance=5e-4,
                )
                close(
                    block.aabb[0][2] - cover.aabb[1][2],
                    CONTACT_CLEARANCE,
                    f"{path.name}:{block_name} supported by lid",
                    tolerance=5e-4,
                )

        if activity in TRANSFER_ACTIVITIES:
            validate_tin_transfer(
                scene,
                table,
                box,
                path.name,
                TRANSFER_ACTIVITIES[activity],
                support_relations,
            )

        stone = scene.object_registry("name", "sharpening_stone")
        if stone is not None:
            support_relations.append((stone, box))
            close(stone.aabb[0][0] - table.aabb[0][0], 0.05, f"{path.name}:stone A-to-B", tolerance=5e-4)
            close(aabb_center(stone)[1] - box.links["base_link"].aabb[0][1], 0.10, f"{path.name}:stone X-to-C", tolerance=5e-4)
            _, stone_orientation = stone.get_position_orientation()
            expected_orientation = th.tensor(STONE_ORIENTATION, dtype=stone_orientation.dtype, device=stone_orientation.device)
            if 1.0 - abs(float(th.dot(stone_orientation, expected_orientation))) > 1e-5:
                raise AssertionError(f"{path.name}:sharpening stone orientation changed")
            close(stone.aabb[0][2] - cover.aabb[1][2], CONTACT_CLEARANCE, f"{path.name}:stone on lid", tolerance=5e-4)
            if th.any(stone.aabb[0][:2] < cover.aabb[0][:2]) or th.any(stone.aabb[1][:2] > cover.aabb[1][:2]):
                raise AssertionError(f"{path.name}:sharpening stone is not fully supported by the lid")

        tumbler_1 = scene.object_registry("name", "plastic_tumbler_1")
        tumbler_2 = scene.object_registry("name", "plastic_tumbler_2")
        if tumbler_1 is not None or tumbler_2 is not None:
            if tumbler_1 is None or tumbler_2 is None:
                raise AssertionError(f"{path.name}:pouring layout has only one tumbler")
            for tumbler in (tumbler_1, tumbler_2):
                support_relations.append((tumbler, box))
                close(tumbler.aabb[0][0] - table.aabb[0][0], 0.10, f"{path.name}:{tumbler.name} front offset", tolerance=5e-4)
                close(tumbler.aabb[0][2] - cover.aabb[1][2], CONTACT_CLEARANCE, f"{path.name}:{tumbler.name} on lid", tolerance=5e-4)
            close(tumbler_1.aabb[0][1] - BOX_CENTER_Y, 0.08, f"{path.name}:tumbler 1 inner edge", tolerance=5e-4)
            close(BOX_CENTER_Y - tumbler_2.aabb[1][1], 0.08, f"{path.name}:tumbler 2 inner edge", tolerance=5e-4)

        start_plank = scene.object_registry("name", "plank_starting_point")
        if start_plank is not None:
            start_bolt = scene.object_registry("name", "bolt_starting_point")
            target_plank = scene.object_registry("name", "plank_target_point")
            target_bolt = scene.object_registry("name", "bolt_target_point")
            tube_name = "large_alloy_tube" if scene.object_registry("name", "large_alloy_tube") is not None else "small_alloy_tube"
            tube = scene.object_registry("name", tube_name)
            for fixture in (start_plank, start_bolt, target_plank, target_bolt):
                if fixture is None or not fixture.fixed_base:
                    raise AssertionError(f"{path.name}:tube fixture members must all exist and be fixed")
            if start_plank.category != "arat_plank" or target_plank.category != "arat_plank":
                raise AssertionError(f"{path.name}:tube planks do not use the arat_plank category")
            if start_bolt.category != "arat_bolt" or target_bolt.category != "arat_bolt":
                raise AssertionError(f"{path.name}:tube bolts do not use the arat_bolt category")
            close(start_plank.aabb_extent, [0.06, 0.085, 0.015], f"{path.name}:start plank dimensions", tolerance=5e-4)
            close(target_plank.aabb_extent, [0.085, 0.34, 0.035], f"{path.name}:target plank dimensions", tolerance=5e-4)
            bolt_extent = [0.02, 0.02, 0.135] if tube_name == "large_alloy_tube" else [0.008, 0.008, 0.06]
            target_extent = [0.02, 0.02, 0.08] if tube_name == "large_alloy_tube" else [0.008, 0.008, 0.06]
            close(start_bolt.aabb_extent, bolt_extent, f"{path.name}:start bolt dimensions", tolerance=5e-4)
            close(target_bolt.aabb_extent, target_extent, f"{path.name}:target bolt dimensions", tolerance=5e-4)
            close(aabb_center(start_bolt)[:2], START_CENTER, f"{path.name}:start bolt center", tolerance=5e-4)
            close(aabb_center(start_plank)[:2], START_CENTER, f"{path.name}:start plank center", tolerance=5e-4)
            close(aabb_center(target_plank)[:2], TARGET_PLANK_CENTER, f"{path.name}:target plank center", tolerance=5e-4)
            close(aabb_center(target_bolt)[:2], TARGET_BOLT_CENTER, f"{path.name}:target bolt center", tolerance=5e-4)
            close(START_CENTER[0] - table.aabb[0][0], 0.08, f"{path.name}:start bolt A-to-B", tolerance=5e-4)
            close(START_CENTER[1] - box.links["base_link"].aabb[0][1], 0.10, f"{path.name}:start bolt X-to-B", tolerance=5e-4)
            close(start_plank.aabb[0][0], cover.aabb[0][0], f"{path.name}:start plank flush with lid", tolerance=5e-4)
            close(
                cover.aabb[1][0] - target_plank.aabb[1][0],
                TARGET_PLANK_FORWARD_SHIFT,
                f"{path.name}:target plank forward shift",
                tolerance=5e-4,
            )
            close(target_plank.aabb[0][1], box.links["base_link"].aabb[0][1], f"{path.name}:target plank right edge", tolerance=5e-4)
            close(aabb_center(target_bolt)[1] - target_plank.aabb[0][1], 0.10, f"{path.name}:target bolt right offset", tolerance=5e-4)
            close(
                start_plank.get_position_orientation()[0][2],
                horizontal_cover_support_z(START_CENTER[0], 0.03),
                f"{path.name}:start plank on local lid surface",
                tolerance=5e-4,
            )
            close(
                target_plank.get_position_orientation()[0][2],
                horizontal_cover_support_z(TARGET_PLANK_CENTER[0], 0.0425),
                f"{path.name}:target plank on local lid surface",
                tolerance=5e-4,
            )
            close(start_bolt.aabb[0][2] - start_plank.aabb[1][2], FIXED_SUPPORT_CLEARANCE, f"{path.name}:start bolt on plank", tolerance=5e-4)
            close(target_bolt.aabb[0][2] - target_plank.aabb[1][2], FIXED_SUPPORT_CLEARANCE, f"{path.name}:target bolt on plank", tolerance=5e-4)
            for child, parent in (
                (start_plank, box),
                (target_plank, box),
                (start_bolt, start_plank),
                (target_bolt, target_plank),
                (tube, start_plank),
            ):
                support_relations.append((child, parent))
            close(tube.get_position_orientation()[0][:2], start_bolt.get_position_orientation()[0][:2], f"{path.name}:tube pegged XY", tolerance=5e-4)
            close(tube.aabb[0][2], start_bolt.aabb[0][2], f"{path.name}:tube pegged height", tolerance=5e-4)
            if tube.fixed_base:
                raise AssertionError(f"{path.name}:{tube_name} should be movable")

        washer = scene.object_registry("name", "washer")
        if washer is not None:
            lid = scene.object_registry("name", "tin_lid")
            target_plank = scene.object_registry("name", "plank_target_point")
            target_bolt = scene.object_registry("name", "bolt_target_point")
            for fixture in (lid, target_plank, target_bolt):
                if fixture is None or not fixture.fixed_base:
                    raise AssertionError(f"{path.name}:washer fixtures must all exist and be fixed")
            if target_plank.category != "arat_plank" or target_bolt.category != "arat_bolt":
                raise AssertionError(f"{path.name}:washer plank/bolt categories are not split")
            close(lid.aabb[0][0] - table.aabb[0][0], 0.05, f"{path.name}:washer lid A-to-B", tolerance=5e-4)
            close(aabb_center(lid)[1] - box.links["base_link"].aabb[0][1], 0.10, f"{path.name}:washer lid X-to-C", tolerance=5e-4)
            validate_tin_collision_clearance(lid, box, path.name, sloped=True)
            close(aabb_center(washer)[:2], aabb_center(lid)[:2], f"{path.name}:washer centered in lid", tolerance=5e-4)
            close(target_plank.aabb_extent, [0.085, 0.085, 0.015], f"{path.name}:washer plank dimensions", tolerance=5e-4)
            close(target_bolt.aabb_extent, [0.008, 0.008, 0.085], f"{path.name}:washer bolt dimensions", tolerance=5e-4)
            close(
                cover.aabb[1][0] - target_plank.aabb[1][0],
                TARGET_PLANK_FORWARD_SHIFT,
                f"{path.name}:washer plank forward shift",
                tolerance=5e-4,
            )
            close(aabb_center(target_bolt)[:2], aabb_center(target_plank)[:2], f"{path.name}:washer bolt centered", tolerance=5e-4)
            close(aabb_center(target_bolt)[1] - box.links["base_link"].aabb[0][1], 0.10, f"{path.name}:washer bolt X-to-C", tolerance=5e-4)
            close(
                target_plank.get_position_orientation()[0][2],
                horizontal_cover_support_z(TARGET_PLANK_CENTER[0], 0.0425),
                f"{path.name}:washer plank on local lid surface",
                tolerance=5e-4,
            )
            close(target_bolt.aabb[0][2] - target_plank.aabb[1][2], FIXED_SUPPORT_CLEARANCE, f"{path.name}:washer bolt on plank", tolerance=5e-4)
            support_relations.extend(
                ((lid, box), (target_plank, box), (target_bolt, target_plank), (washer, lid))
            )
            if washer.fixed_base:
                raise AssertionError(f"{path.name}:washer should be movable")

        mannequin = scene.object_registry("name", "mannequin")
        if mannequin is not None:
            if mannequin.category != "mannequin" or mannequin.model != "nphsfp":
                raise AssertionError(f"{path.name}:wrong mannequin asset")
            if not mannequin.fixed_base:
                raise AssertionError(f"{path.name}:mannequin must be fixed after unsupported stability failure")
            close(mannequin.get_position_orientation()[0], MANNEQUIN_ROOT, f"{path.name}:mannequin root", tolerance=5e-4)
            _, orientation = mannequin.get_position_orientation()
            expected_orientation = th.tensor(
                MANNEQUIN_ORIENTATION, dtype=orientation.dtype, device=orientation.device
            )
            if 1.0 - abs(float(th.dot(orientation, expected_orientation))) > 1e-5:
                raise AssertionError(f"{path.name}:mannequin orientation changed")
            close(mannequin.aabb[0][2], 0.0, f"{path.name}:mannequin floor contact", tolerance=5e-4)
            close(mannequin.aabb_extent[2], MANNEQUIN_TARGET_HEIGHT, f"{path.name}:mannequin height", tolerance=5e-4)
            close(aabb_center(mannequin)[:2], MANNEQUIN_TARGET_CENTER_XY, f"{path.name}:mannequin XY center", tolerance=5e-4)
            initial_position, initial_orientation = mannequin.get_position_orientation()
            for _ in range(300):
                og.sim.step()
            final_position, final_orientation = mannequin.get_position_orientation()
            close(final_position, initial_position, f"{path.name}:mannequin fixed-position stability", tolerance=1e-5)
            if 1.0 - abs(float(th.dot(initial_orientation, final_orientation))) > 1e-7:
                raise AssertionError(f"{path.name}:mannequin fixed orientation drifted")

        masses = {}
        for name, expected_mass in EXACT_MASSES_KG.items():
            obj = scene.object_registry("name", name)
            if obj is not None:
                actual_mass = float(obj.root_link.mass)
                if abs(actual_mass - expected_mass) > 2e-6:
                    raise AssertionError(f"{path.name}:{name} mass expected {expected_mass}, got {actual_mass}")
                masses[name] = actual_mass

        systems = sorted(scene.active_systems)
        water_validation = None
        if activity == "arat_grip_pour_water":
            water = scene.get_system("water", force_init=False)
            if water.n_particles != WATER_PROXY_PARTICLES:
                raise AssertionError(f"Pouring scene loaded {water.n_particles} water particles instead of {WATER_PROXY_PARTICLES}")
            represented_ml = water.n_particles * (2.0 * float(water.particle_radius)) ** 3 * 1e6
            if abs(represented_ml - WATER_REQUIRED_ML) / WATER_REQUIRED_ML > 0.005:
                raise AssertionError(f"Pouring proxy is {represented_ml} mL, more than 0.5% from {WATER_REQUIRED_ML} mL")
            n_in_tumbler_1 = int(tumbler_1.states[ContainedParticles].get_value(water).n_in_volume)
            n_in_tumbler_2 = int(tumbler_2.states[ContainedParticles].get_value(water).n_in_volume)
            if n_in_tumbler_1 != WATER_PROXY_PARTICLES or n_in_tumbler_2 != 0:
                raise AssertionError(f"Water containment mismatch: tumbler1={n_in_tumbler_1}, tumbler2={n_in_tumbler_2}")
            if not tumbler_1.states[Filled].get_value(water):
                raise AssertionError("plastic_tumbler_1 does not load in the Filled state")
            water_validation = {
                "required_us_fluid_ounces": 4.0,
                "required_volume_ml": WATER_REQUIRED_ML,
                "proxy_particle_count": water.n_particles,
                "proxy_volume_ml": represented_ml,
                "relative_volume_error_percent": 100.0 * (represented_ml - WATER_REQUIRED_ML) / WATER_REQUIRED_ML,
                "particles_in_plastic_tumbler_1": n_in_tumbler_1,
                "particles_in_plastic_tumbler_2": n_in_tumbler_2,
                "plastic_tumbler_1_filled": True,
            }
        elif systems:
            raise AssertionError(f"Unexpected systems in {path.name}: {systems}")

        # The checks above intentionally inspect the exact authored layout.
        # Initialize and synchronize physics only after those checks, then
        # evaluate every semantic support relation once movable objects settle.
        for _ in range(3):
            og.sim.step()
        for child, parent in support_relations:
            if not child.states[OnTop].get_value(parent):
                raise AssertionError(f"{path.name}:{child.name} is not semantically OnTop {parent.name}")

        results.append(
            {
                "file": str(path.relative_to(REPOSITORY_ROOT)),
                "object_count": len(loaded_names),
                "object_names": sorted(loaded_names),
                "systems": systems,
                "water": water_validation,
                "exact_masses_kg": masses,
                "status": "passed",
            }
        )
        print(f"ARAT_BEHAVIOR_DEX_LOAD_PASSED {index + 1}/{len(all_paths)} file={path.name}")
    return results


def robot_integration_validation(scene_file: Path) -> dict:
    """Load one representative template with the intended Franka + Sharpa robot."""

    og.clear()
    env = og.Environment(
        configs={
            "scene": {
                "type": "Scene",
                "scene_file": str(scene_file),
                "use_floor_plane": True,
                "floor_plane_visible": True,
                "floor_plane_color": [0.5, 0.5, 0.5],
                "use_skybox": True,
            },
            "robots": [
                {
                    "name": "franka_sharpa_right",
                    "model": "franka",
                    "dataset_name": "omnigibson-robot-assets",
                    "end_effector": "sharpa_right",
                    "fixed_base": True,
                    "self_collisions": False,
                    "obs_modalities": [],
                    # The launcher pose and reset joints are deliberately
                    # unchanged; the common ARAT layout is aligned to them.
                    "position": ROBOT_POSITION,
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "reset_joint_pos": RESET_JOINT_POSITIONS,
                }
            ],
            "task": {"type": "DummyTask"},
        }
    )
    robot = env.robots[0]
    if robot.model != "franka" or robot.end_effector != "sharpa_right":
        raise AssertionError(f"Loaded unexpected robot model {robot.model}")
    if robot.name != "franka_sharpa_right" or not robot.fixed_base:
        raise AssertionError("Franka + Sharpa integration robot identity or fixed-base setting is wrong")
    if env.scene.object_registry("name", "wooden_block_10") is None:
        raise AssertionError("Representative task apparatus did not load with the robot")
    close(robot.get_position_orientation()[0], ROBOT_POSITION, "unchanged robot world position", tolerance=1e-5)
    realized_joint_positions = robot.get_joint_positions()
    close(
        realized_joint_positions[:7],
        RESET_JOINT_POSITIONS[:7],
        "unchanged Franka arm reset joints",
        tolerance=1e-4,
    )
    if RESET_JOINT_POSITIONS[7:] != [0.0] * 22:
        raise AssertionError("The configured Sharpa reset commands must remain zero")
    og.sim.step()
    return {
        "status": "passed",
        "scene_file": str(scene_file.relative_to(REPOSITORY_ROOT)),
        "model": robot.model,
        "dataset_name": "omnigibson-robot-assets",
        "end_effector": robot.end_effector,
        "name": robot.name,
        "fixed_base": robot.fixed_base,
        "joint_count": len(robot.joints),
        "position_xyz_m": [float(value) for value in robot.get_position_orientation()[0]],
        "configured_reset_joint_positions": RESET_JOINT_POSITIONS,
        "realized_reset_joint_positions": [float(value) for value in realized_joint_positions],
        "finger_reset_note": "Sharpa zero commands are clamped to the hand model's physical joint limits.",
    }


def main() -> None:
    manifest, paths = static_validation()
    results = runtime_validation(paths)
    robot_integration = robot_integration_validation(paths[0])
    report = {
        "status": "passed",
        "environment": "behavior_dex",
        "asset_version": 1,
        "scene_model": manifest["scene_model"],
        "base_scene_count": 1,
        "task_instance_count": len(paths),
        "loaded_scene_count": len(results),
        "robot_integration": robot_integration,
        "scenes": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ARAT_BEHAVIOR_DEX_VALIDATION_PASSED scenes={len(results)} report={REPORT_PATH}")
    og.shutdown()


if __name__ == "__main__":
    main()

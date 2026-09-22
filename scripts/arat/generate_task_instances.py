#!/usr/bin/env python3
"""Generate the ARAT v1 base scene and all 19 task layouts.

Run this script in ``behavior_arat``. It authors canonical OmniGibson scene
JSON from the converted USD assets; ``validate_task_instances.py`` performs
the independent ``behavior_dex`` load tests.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import torch as th

import omnigibson as og
from omnigibson.macros import gm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_DATASET = "arat-assets-v1"
SCENE_MODEL = "arat_base"
ASSET_SCENE_DIR = REPOSITORY_ROOT / "datasets" / ASSET_DATASET / "scenes" / SCENE_MODEL
INSTANCE_ROOT = REPOSITORY_ROOT / "datasets" / "arat-task-instances"
INSTANCE_SCENE_DIR = INSTANCE_ROOT / "scenes" / SCENE_MODEL
DATASET_NAME = "arat-assets-v1"

IDENTITY = [0.0, 0.0, 0.0, 1.0]
YAW_MINUS_90 = [0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)]
STONE_NARROW_SIDE_DIAGONAL = [0.7032331763, 0.0739127852, 0.0739127852, 0.7032331763]

# Preserve the launcher's robot world pose and reset joints. The common ARAT
# layout is raised 10 cm and moved 20 cm farther from the robot relative to
# the original human-reference alignment.
ROBOT_POSITION = [-0.5685013461, -0.1084822643, 0.0156103678]
ROBOT_SHOULDER_HEIGHT_FROM_ROOT = 1.194
HUMAN_SHOULDER_TO_TABLE = 1.033 - 0.75
REFERENCE_TABLE_SURFACE_Z = (
    ROBOT_POSITION[2] + ROBOT_SHOULDER_HEIGHT_FROM_ROOT - HUMAN_SHOULDER_TO_TABLE
)
REFERENCE_TABLE_PROXIMAL_DISTANCE_FROM_ROBOT = 0.235
COMMON_TABLE_HEIGHT_INCREASE = 0.10
COMMON_TABLE_DISTANCE_INCREASE = 0.20
TABLE_SURFACE_Z = REFERENCE_TABLE_SURFACE_Z + COMMON_TABLE_HEIGHT_INCREASE
TABLE_PROXIMAL_DISTANCE_FROM_ROBOT = (
    REFERENCE_TABLE_PROXIMAL_DISTANCE_FROM_ROBOT + COMMON_TABLE_DISTANCE_INCREASE
)
TABLE_PROXIMAL_X = ROBOT_POSITION[0] + TABLE_PROXIMAL_DISTANCE_FROM_ROBOT
TABLE_CENTER_Y = ROBOT_POSITION[1]
TABLE_DEPTH = 0.75
TABLE_DISTAL_X = TABLE_PROXIMAL_X + TABLE_DEPTH
# Measured from breakfast_table/nvoqyl's collision AABB metadata and verified
# in behavior_arat after scaling.
# The width and height retain the ARAT values. The depth is deliberately
# extended from 49 to 75 cm so the articulated box has comfortable support;
# its proximal edge and all task-relative placement datums remain unchanged.
# The root offset places both the feet and floor at z=0 while the top is at
# the human-equivalent task height.
BREAKFAST_TABLE_SOURCE_EXTENT = [0.7630001600354946, 1.2029907715004438, 0.74122540283]
BREAKFAST_TABLE_SOURCE_LO = [-0.38150000580489124, -0.6014904578847167, -0.6571727555588319]
BREAKFAST_TABLE_SOURCE_HI = [0.3815001542306034, 0.6015003136157271, 0.08405264727116812]
BREAKFAST_TABLE_SOURCE_TOP_FROM_ROOT = 0.08405264727116812
BREAKFAST_TABLE_SCALE_X = TABLE_DEPTH / BREAKFAST_TABLE_SOURCE_EXTENT[0]
BREAKFAST_TABLE_SCALE_Y = 0.76 / BREAKFAST_TABLE_SOURCE_EXTENT[1]
BREAKFAST_TABLE_ROOT_X = TABLE_PROXIMAL_X - BREAKFAST_TABLE_SOURCE_LO[0] * BREAKFAST_TABLE_SCALE_X
BREAKFAST_TABLE_ROOT_Y = TABLE_CENTER_Y - (
    (BREAKFAST_TABLE_SOURCE_LO[1] + BREAKFAST_TABLE_SOURCE_HI[1]) * BREAKFAST_TABLE_SCALE_Y / 2.0
)

# The articulated box authored for the prior HTS setup opens along local -Y.
# A -90-degree yaw makes the lid open toward world -X (the operator).
ARAT_BOX_WIDTH = 0.5461
ARAT_BOX_COVER_DEPTH = 0.3302
ARAT_BOX_SUPPORT_HEIGHT = 0.0271
ARAT_BOX_SHELL_HEIGHT = 0.3429
ARAT_BOX_SHELL_DEPTH = 0.1317625
FRONT_EDGE_DISTANCE = 0.05
RIGHT_EDGE_TO_CENTROID = 0.10
ARAT_BOX_COVER_PROXIMAL_X = TABLE_PROXIMAL_X + FRONT_EDGE_DISTANCE
ARAT_BOX_ROOT = [
    ARAT_BOX_COVER_PROXIMAL_X + ARAT_BOX_COVER_DEPTH,
    TABLE_CENTER_Y + ARAT_BOX_WIDTH / 2.0,
    TABLE_SURFACE_Z + ARAT_BOX_SUPPORT_HEIGHT,
]
ARAT_BOX_RIGHTMOST_Y = TABLE_CENTER_Y - ARAT_BOX_WIDTH / 2.0
ARAT_BOX_CENTER_Y = TABLE_CENTER_Y
ARAT_BOX_COVER_DISTAL_X = ARAT_BOX_ROOT[0]
ARAT_BOX_COVER_THICKNESS = 0.017462
# Lowering the cover hinge by this amount makes the cover collision underside
# flush with the table at joint position 0, while its top remains horizontal.
ARAT_BOX_COVER_JOINT_DROP = ARAT_BOX_SUPPORT_HEIGHT - ARAT_BOX_COVER_THICKNESS
ARAT_BOX_COVER_SURFACE_Z = ARAT_BOX_ROOT[2] - ARAT_BOX_COVER_JOINT_DROP
ARAT_BOX_COVER_SLOPE_X = 0.0
# The horizontal cover collision surface ends at the articulated link datum.
ARAT_BOX_COVER_COLLISION_SURFACE_OFFSET_Z = 0.0
ARAT_BOX_SHELL_SURFACE_Z = ARAT_BOX_ROOT[2] + ARAT_BOX_SHELL_HEIGHT
ARAT_BOX_SHELL_DISTAL_X = ARAT_BOX_ROOT[0] + ARAT_BOX_SHELL_DEPTH
ARAT_BOX_BACK_SUPPORT_MARGIN = TABLE_DISTAL_X - ARAT_BOX_SHELL_DISTAL_X
TASK_CENTROID_Y = ARAT_BOX_RIGHTMOST_Y + RIGHT_EDGE_TO_CENTROID

BLOCK_SIDE_LENGTHS = {
    "wooden_block_10": 0.10,
    "wooden_block_7_5": 0.075,
    "wooden_block_5_0": 0.05,
    "wooden_block_2_5": 0.025,
}
BLOCK_CENTROID_Y = TASK_CENTROID_Y
TIN_LID_DIAMETER = 0.09
TIN_LID_RADIUS = TIN_LID_DIAMETER / 2.0
TIN_LID_HEIGHT = 0.01
# The open box cover is horizontal at joint position 0. Keep this formulation
# explicit so any future nonzero cover slope also updates the fixed tin pose.
TIN_LID_1_PITCH = -math.atan(ARAT_BOX_COVER_SLOPE_X)
TIN_LID_1_ORIENTATION = [0.0, math.sin(TIN_LID_1_PITCH / 2.0), 0.0, math.cos(TIN_LID_1_PITCH / 2.0)]
TIN_LID_1_FRONT_OFFSET_X = (
    TIN_LID_RADIUS * math.cos(TIN_LID_1_PITCH) - TIN_LID_HEIGHT * math.sin(TIN_LID_1_PITCH)
)
TIN_LID_1_CENTER_X = TABLE_PROXIMAL_X + FRONT_EDGE_DISTANCE + TIN_LID_1_FRONT_OFFSET_X
TIN_LID_1_CENTER_Y = TASK_CENTROID_Y
TIN_LID_1_AABB_CENTER_X = TIN_LID_1_CENTER_X + TIN_LID_HEIGHT * math.sin(TIN_LID_1_PITCH) / 2.0
TIN_LID_1_CENTER_Z = (
    ARAT_BOX_COVER_SURFACE_Z
    + ARAT_BOX_COVER_COLLISION_SURFACE_OFFSET_Z
    + ARAT_BOX_COVER_SLOPE_X * (TIN_LID_1_CENTER_X - ARAT_BOX_ROOT[0])
)
TIN_LID_2_CENTER_X = ARAT_BOX_ROOT[0] + TIN_LID_RADIUS
TIN_LID_2_CENTER_Y = TASK_CENTROID_Y
CRICKET_LID_1_CENTER_X = TIN_LID_1_AABB_CENTER_X
CRICKET_LID_2_CENTER_X = ARAT_BOX_ROOT[0] + TIN_LID_RADIUS
CRICKET_LID_CENTROID_Y = TASK_CENTROID_Y
# The stone mesh spans local [-.05, .05] x [-.0125, .0125] x [0, .01].
# These are its world AABB values under the unchanged diagonal quaternion.
STONE_WORLD_AABB_CENTER_OFFSET_X = 0.0010395584539723522
STONE_WORLD_AABB_CENTER_OFFSET_Y = -0.00489073800369378
STONE_WORLD_AABB_EXTENT_X = 0.09989387698182028
STONE_CENTROID_X = TABLE_PROXIMAL_X + FRONT_EDGE_DISTANCE + STONE_WORLD_AABB_EXTENT_X / 2.0
STONE_CENTROID_Y = TASK_CENTROID_Y
STONE_ROOT_X = STONE_CENTROID_X - STONE_WORLD_AABB_CENTER_OFFSET_X
STONE_ROOT_Y = STONE_CENTROID_Y - STONE_WORLD_AABB_CENTER_OFFSET_Y
STONE_DISTAL_EDGE_X = TABLE_PROXIMAL_X + FRONT_EDGE_DISTANCE + STONE_WORLD_AABB_EXTENT_X

START_BOLT_CENTER_X = TABLE_PROXIMAL_X + 0.08
START_BOLT_CENTER_Y = TASK_CENTROID_Y
START_PLANK_DEPTH = 0.06
START_PLANK_WIDTH = 0.085
START_PLANK_HEIGHT = 0.015
TARGET_PLANK_DEPTH = 0.085
TARGET_PLANK_WIDTH = 0.34
TARGET_PLANK_HEIGHT = 0.035
TARGET_PLANK_FORWARD_SHIFT = 0.020
START_PLANK_CENTER_X = START_BOLT_CENTER_X
START_PLANK_CENTER_Y = START_BOLT_CENTER_Y
# The cover hinge barrel is 1.905 cm in diameter and occupies the final
# 1.8256 cm of lid depth. Shift the target fixtures 2 cm toward the operator
# so their distal edges clear the black hinge instead of resting on it.
TARGET_PLANK_CENTER_X = (
    ARAT_BOX_COVER_DISTAL_X - TARGET_PLANK_DEPTH / 2.0 - TARGET_PLANK_FORWARD_SHIFT
)
TARGET_PLANK_CENTER_Y = ARAT_BOX_RIGHTMOST_Y + TARGET_PLANK_WIDTH / 2.0
TARGET_BOLT_CENTER_X = TARGET_PLANK_CENTER_X
TARGET_BOLT_CENTER_Y = ARAT_BOX_RIGHTMOST_Y + RIGHT_EDGE_TO_CENTROID
WASHER_PLANK_DEPTH = 0.085
WASHER_PLANK_CENTER_X = (
    ARAT_BOX_COVER_DISTAL_X - WASHER_PLANK_DEPTH / 2.0 - TARGET_PLANK_FORWARD_SHIFT
)
WASHER_PLANK_CENTER_Y = TASK_CENTROID_Y

CUP_OUTER_RADIUS = 0.0375
CUP_FRONT_EDGE_DISTANCE = 0.10
CUP_INNER_EDGE_FROM_BOX_MIDLINE = 0.08
CUP_CENTER_X = TABLE_PROXIMAL_X + CUP_FRONT_EDGE_DISTANCE + CUP_OUTER_RADIUS
TUMBLER_1_CENTER_Y = TABLE_CENTER_Y + CUP_INNER_EDGE_FROM_BOX_MIDLINE + CUP_OUTER_RADIUS
TUMBLER_2_CENTER_Y = TABLE_CENTER_Y - CUP_INNER_EDGE_FROM_BOX_MIDLINE - CUP_OUTER_RADIUS
CONTACT_CLEARANCE = 0.0005
# Fixed fixtures are authored flush with their supports. Unlike movable
# objects, they cannot settle under gravity, and a gap would make their
# initial support relation geometrically false.
FIXED_SUPPORT_CLEARANCE = 0.0
# Keep the fixed tin collision bottoms visibly clear of the wood while still
# remaining within OnTop's 2 mm fixed-support tolerance. The first tin is
# aligned to the cover plane; its shallow perimeter geometry is checked by
# dense footprint ray sampling in the behavior_dex validator.
TIN_FIXED_SUPPORT_CLEARANCE = 0.0015
WATER_US_FLUID_OUNCES = 4.0
WATER_VOLUME_ML = 118.29411825
WATER_PROXY_PARTICLES = 41


def horizontal_cover_support_z(center_x: float, half_extent_x: float) -> float:
    """Return the highest lid-panel Z beneath a horizontal fixed fixture."""

    highest_contact_x = center_x + half_extent_x
    return (
        ARAT_BOX_COVER_SURFACE_Z
        + ARAT_BOX_COVER_SLOPE_X * (highest_contact_x - ARAT_BOX_ROOT[0])
        + FIXED_SUPPORT_CLEARANCE
    )

# The articulated ARAT box handle is 39.54 cm above the tabletop. The common
# 10 cm table elevation therefore puts its top at 1.4220103678 m.
# mannequin/nphsfp is authored lying along local X, with its head at
# local -X. This rotation makes the head world-up and turns the mannequin to
# face the robot side (world -X). Scale and root placement use the asset's
# collision AABB so its feet touch Z=0, its top reaches the box handle, and its
# world-XY AABB center matches the articulated box's world-XY AABB center.
ARAT_BOX_HANDLE_HEIGHT_ABOVE_TABLE = 1.1454 - 0.75
ARAT_BOX_TOTAL_HEIGHT_FROM_FLOOR = TABLE_SURFACE_Z + ARAT_BOX_HANDLE_HEIGHT_ABOVE_TABLE
ARAT_BOX_AABB_CENTER_XY = [
    ARAT_BOX_ROOT[0] + (0.03598145395517349 - 0.1352),
    TABLE_CENTER_Y + 2.9802322387695312e-08,
]
MANNEQUIN_NATIVE_COLLISION_AABB_CENTER = [
    0.07910473742101737,
    1.4772632368931227e-05,
    0.016168891737376692,
]
MANNEQUIN_NATIVE_COLLISION_AABB_EXTENT = [
    1.75698565657241,
    1.2042667289002793,
    0.276470538288757,
]
MANNEQUIN_SCALE = ARAT_BOX_TOTAL_HEIGHT_FROM_FLOOR / MANNEQUIN_NATIVE_COLLISION_AABB_EXTENT[0]
MANNEQUIN_ORIENTATION = [-math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
MANNEQUIN_ROOT = [
    ARAT_BOX_AABB_CENTER_XY[0] + MANNEQUIN_SCALE * MANNEQUIN_NATIVE_COLLISION_AABB_CENTER[2],
    ARAT_BOX_AABB_CENTER_XY[1] + MANNEQUIN_SCALE * MANNEQUIN_NATIVE_COLLISION_AABB_CENTER[1],
    MANNEQUIN_SCALE
    * (MANNEQUIN_NATIVE_COLLISION_AABB_CENTER[0] + MANNEQUIN_NATIVE_COLLISION_AABB_EXTENT[0] / 2.0),
]

GROSS_MOVEMENT_ACTIVITIES = (
    "arat_gross_movement_hand_behind_head",
    "arat_gross_movement_hand_top_head",
    "arat_gross_movement_hand_mouth",
)


def dataset_object(name: str, category: str, model: str, position, **kwargs) -> dict:
    return {
        "type": "DatasetObject",
        "dataset_name": DATASET_NAME,
        "name": name,
        "category": category,
        "model": model,
        "position": list(position),
        "orientation": kwargs.pop("orientation", IDENTITY),
        "in_rooms": ["empty_room_0"],
        **kwargs,
    }


def table() -> dict:
    scale_z = TABLE_SURFACE_Z / BREAKFAST_TABLE_SOURCE_EXTENT[2]
    return dataset_object(
        "table",
        "breakfast_table",
        "nvoqyl",
        [
            BREAKFAST_TABLE_ROOT_X,
            BREAKFAST_TABLE_ROOT_Y,
            TABLE_SURFACE_Z - BREAKFAST_TABLE_SOURCE_TOP_FROM_ROOT * scale_z,
        ],
        scale=[BREAKFAST_TABLE_SCALE_X, BREAKFAST_TABLE_SCALE_Y, scale_z],
        fixed_base=True,
    )
def arat_box() -> dict:
    return dataset_object(
        "arat_box",
        "arat_box",
        "aratbx",
        ARAT_BOX_ROOT,
        orientation=YAW_MINUS_90,
        fixed_base=True,
    )


def mannequin() -> dict:
    return dataset_object(
        "mannequin",
        "mannequin",
        "nphsfp",
        MANNEQUIN_ROOT,
        orientation=MANNEQUIN_ORIENTATION,
        scale=[MANNEQUIN_SCALE] * 3,
        fixed_base=True,
    )


def light_cfgs() -> list[dict]:
    return [
        {
            "type": "LightObject",
            "light_type": "Sphere",
            "name": "arat_task_light_near",
            "radius": 0.01,
            "intensity": 1e5,
            "position": [-2.0, -2.0, 2.0],
        },
        {
            "type": "LightObject",
            "light_type": "Sphere",
            "name": "arat_task_light_far",
            "radius": 0.01,
            "intensity": 1e5,
            "position": [2.0, 2.0, 2.0],
        },
    ]


def tin_lid_1(name: str = "tin_lid_1") -> dict:
    return dataset_object(
        name,
        "arat_tin",
        "tin_lid",
        [
            TIN_LID_1_CENTER_X,
            TIN_LID_1_CENTER_Y,
            TIN_LID_1_CENTER_Z + TIN_FIXED_SUPPORT_CLEARANCE,
        ],
        orientation=TIN_LID_1_ORIENTATION,
        fixed_base=True,
    )


def tin_lid_2(name: str = "tin_lid_2") -> dict:
    return dataset_object(
        name,
        "arat_tin",
        "tin_lid",
        [TIN_LID_2_CENTER_X, TIN_LID_2_CENTER_Y, ARAT_BOX_SHELL_SURFACE_Z + TIN_FIXED_SUPPORT_CLEARANCE],
        fixed_base=True,
    )


def transfer_layout(object_cfg: dict) -> list[dict]:
    return [arat_box(), tin_lid_1(), tin_lid_2(), object_cfg]


def cricket_ball_layout() -> list[dict]:
    return [
        arat_box(),
        tin_lid_1(),
        tin_lid_2(),
        dataset_object(
            "cricket_ball",
            "arat_cricket_ball",
            "cricket_ball",
            [
                CRICKET_LID_1_CENTER_X,
                CRICKET_LID_CENTROID_Y,
                TIN_LID_1_CENTER_Z + TIN_FIXED_SUPPORT_CLEARANCE + 0.0008 + CONTACT_CLEARANCE,
            ],
        ),
    ]


def block(model: str, name: str) -> dict:
    side_length = BLOCK_SIDE_LENGTHS[model]
    center_x = TABLE_PROXIMAL_X + FRONT_EDGE_DISTANCE + side_length / 2.0
    return dataset_object(
        name,
        "arat_wooden_block",
        model,
        [center_x, BLOCK_CENTROID_Y, ARAT_BOX_COVER_SURFACE_Z + CONTACT_CLEARANCE],
    )


def tube_layout(model: str, name: str) -> list[dict]:
    large = model == "large_alloy_tube"
    start_bolt_model = "bolt_large_starting_point" if large else "bolt_small_starting_point"
    target_bolt_model = "bolt_large_target_point" if large else "bolt_small_target_point"
    start_plank_z = horizontal_cover_support_z(START_PLANK_CENTER_X, START_PLANK_DEPTH / 2.0)
    target_plank_z = horizontal_cover_support_z(TARGET_PLANK_CENTER_X, TARGET_PLANK_DEPTH / 2.0)
    start_bolt_z = start_plank_z + START_PLANK_HEIGHT + FIXED_SUPPORT_CLEARANCE
    target_bolt_z = target_plank_z + TARGET_PLANK_HEIGHT + FIXED_SUPPORT_CLEARANCE
    start_plank = dataset_object(
        "plank_starting_point",
        "arat_plank",
        "plank_starting_point",
        [START_PLANK_CENTER_X, START_PLANK_CENTER_Y, start_plank_z],
        fixed_base=True,
    )
    target_plank = dataset_object(
        "plank_target_point",
        "arat_plank",
        "plank_target_point",
        [TARGET_PLANK_CENTER_X, TARGET_PLANK_CENTER_Y, target_plank_z],
        fixed_base=True,
    )
    start_bolt = dataset_object(
        "bolt_starting_point",
        "arat_bolt",
        start_bolt_model,
        [START_BOLT_CENTER_X, START_BOLT_CENTER_Y, start_bolt_z],
        fixed_base=True,
    )
    target_bolt = dataset_object(
        "bolt_target_point",
        "arat_bolt",
        target_bolt_model,
        [TARGET_BOLT_CENTER_X, TARGET_BOLT_CENTER_Y, target_bolt_z],
        fixed_base=True,
    )
    tube = dataset_object(
        name,
        "arat_alloy_tube",
        model,
        [START_BOLT_CENTER_X, START_BOLT_CENTER_Y, start_bolt_z],
    )
    return [target_plank, target_bolt, start_plank, start_bolt, tube]


def build_tasks() -> list[dict]:
    tasks = [
        {
            "item": 1,
            "activity": "arat_grasp_block_10cm",
            "objects": [arat_box(), block("wooden_block_10", "wooden_block_10")],
            "instances": {"arat_box.n.01_1": "arat_box", "arat_wooden_block.n.01_1": "wooden_block_10"},
        },
        {
            "item": 2,
            "activity": "arat_grasp_block_2_5cm",
            "objects": [arat_box(), block("wooden_block_2_5", "wooden_block_2_5")],
            "instances": {"arat_box.n.01_1": "arat_box", "arat_wooden_block.n.01_1": "wooden_block_2_5"},
        },
        {
            "item": 3,
            "activity": "arat_grasp_block_5cm",
            "objects": [arat_box(), block("wooden_block_5_0", "wooden_block_5_0")],
            "instances": {"arat_box.n.01_1": "arat_box", "arat_wooden_block.n.01_1": "wooden_block_5_0"},
        },
        {
            "item": 4,
            "activity": "arat_grasp_block_7_5cm",
            "objects": [arat_box(), block("wooden_block_7_5", "wooden_block_7_5")],
            "instances": {"arat_box.n.01_1": "arat_box", "arat_wooden_block.n.01_1": "wooden_block_7_5"},
        },
        {
            "item": 5,
            "activity": "arat_grasp_cricket_ball",
            "objects": cricket_ball_layout(),
            "instances": {
                "arat_box.n.01_1": "arat_box",
                "arat_cricket_ball.n.01_1": "cricket_ball",
                "arat_tin.n.01_1": "tin_lid_1",
                "arat_tin.n.01_2": "tin_lid_2",
            },
        },
        {
            "item": 6,
            "activity": "arat_grasp_sharpening_stone",
            "objects": [
                arat_box(),
                dataset_object(
                    "sharpening_stone",
                    "arat_sharpening_stone",
                    "sharpening_stone",
                    [STONE_ROOT_X, STONE_ROOT_Y, ARAT_BOX_COVER_SURFACE_Z + 0.0125 + CONTACT_CLEARANCE],
                    orientation=STONE_NARROW_SIDE_DIAGONAL,
                ),
            ],
            "instances": {
                "arat_box.n.01_1": "arat_box",
                "arat_sharpening_stone.n.01_1": "sharpening_stone",
            },
        },
        {
            "item": 7,
            "activity": "arat_grip_pour_water",
            "objects": [
                arat_box(),
                dataset_object(
                    "plastic_tumbler_1",
                    "arat_cup",
                    "cup_blue",
                    [CUP_CENTER_X, TUMBLER_1_CENTER_Y, ARAT_BOX_COVER_SURFACE_Z + CONTACT_CLEARANCE],
                    abilities={"fillable": {}},
                ),
                dataset_object(
                    "plastic_tumbler_2",
                    "arat_cup",
                    "cup_red",
                    [CUP_CENTER_X, TUMBLER_2_CENTER_Y, ARAT_BOX_COVER_SURFACE_Z + CONTACT_CLEARANCE],
                    abilities={"fillable": {}},
                ),
            ],
            "instances": {
                "arat_box.n.01_1": "arat_box",
                "arat_cup.n.01_1": "plastic_tumbler_1",
                "arat_cup.n.01_2": "plastic_tumbler_2",
            },
        },
        {
            "item": 8,
            "activity": "arat_grip_alloy_tube_2_5cm",
            "objects": [arat_box(), *tube_layout("large_alloy_tube", "large_alloy_tube")],
            "instances": {
                "arat_box.n.01_1": "arat_box",
                "arat_alloy_tube.n.01_1": "large_alloy_tube",
                "arat_plank.n.01_1": "plank_target_point",
                "arat_plank.n.01_2": "plank_starting_point",
                "arat_bolt.n.01_1": "bolt_target_point",
                "arat_bolt.n.01_2": "bolt_starting_point",
            },
        },
        {
            "item": 9,
            "activity": "arat_grip_alloy_tube_1cm",
            "objects": [arat_box(), *tube_layout("small_alloy_tube", "small_alloy_tube")],
            "instances": {
                "arat_box.n.01_1": "arat_box",
                "arat_alloy_tube.n.01_1": "small_alloy_tube",
                "arat_plank.n.01_1": "plank_target_point",
                "arat_plank.n.01_2": "plank_starting_point",
                "arat_bolt.n.01_1": "bolt_target_point",
                "arat_bolt.n.01_2": "bolt_starting_point",
            },
        },
        {
            "item": 10,
            "activity": "arat_grip_washer_over_bolt",
            "objects": [
                arat_box(),
                tin_lid_1(name="tin_lid"),
                dataset_object(
                    "plank_target_point",
                    "arat_plank",
                    "plank_washer_target_point",
                    [
                        WASHER_PLANK_CENTER_X,
                        WASHER_PLANK_CENTER_Y,
                        horizontal_cover_support_z(WASHER_PLANK_CENTER_X, WASHER_PLANK_DEPTH / 2.0),
                    ],
                    fixed_base=True,
                ),
                dataset_object(
                    "bolt_target_point",
                    "arat_bolt",
                    "bolt_washer_target_point",
                    [
                        WASHER_PLANK_CENTER_X,
                        WASHER_PLANK_CENTER_Y,
                        horizontal_cover_support_z(WASHER_PLANK_CENTER_X, WASHER_PLANK_DEPTH / 2.0)
                        + START_PLANK_HEIGHT
                        + FIXED_SUPPORT_CLEARANCE,
                    ],
                    fixed_base=True,
                ),
                dataset_object(
                    "washer",
                    "arat_washer",
                    "washer",
                    [
                        TIN_LID_1_AABB_CENTER_X,
                        TIN_LID_1_CENTER_Y,
                        TIN_LID_1_CENTER_Z + TIN_FIXED_SUPPORT_CLEARANCE + 0.0008 + CONTACT_CLEARANCE,
                    ],
                ),
            ],
            "instances": {
                "arat_box.n.01_1": "arat_box",
                "arat_washer.n.01_1": "washer",
                "arat_tin.n.01_1": "tin_lid",
                "arat_plank.n.01_1": "plank_target_point",
                "arat_bolt.n.01_1": "bolt_target_point",
            },
        },
    ]
    pinch_items = (
        (11, "arat_pinch_ball_bearing_ring", "arat_ball_bearing", "ball_bearing", "ball_bearing"),
        (12, "arat_pinch_marble_index", "arat_marble", "marble", "marble"),
        (13, "arat_pinch_ball_bearing_middle", "arat_ball_bearing", "ball_bearing", "ball_bearing"),
        (14, "arat_pinch_ball_bearing_index", "arat_ball_bearing", "ball_bearing", "ball_bearing"),
        (15, "arat_pinch_marble_ring", "arat_marble", "marble", "marble"),
        (16, "arat_pinch_marble_middle", "arat_marble", "marble", "marble"),
    )
    for item, activity, category, model, name in pinch_items:
        synset = "arat_ball_bearing.n.01_1" if category == "arat_ball_bearing" else "arat_marble.n.01_1"
        tasks.append(
            {
                "item": item,
                "activity": activity,
                "objects": transfer_layout(
                    dataset_object(
                        name,
                        category,
                        model,
                        [
                            TIN_LID_1_AABB_CENTER_X,
                            TIN_LID_1_CENTER_Y,
                            TIN_LID_1_CENTER_Z + TIN_FIXED_SUPPORT_CLEARANCE + 0.0008 + CONTACT_CLEARANCE,
                        ],
                    )
                ),
                "instances": {
                    "arat_box.n.01_1": "arat_box",
                    synset: name,
                    "arat_tin.n.01_1": "tin_lid_1",
                    "arat_tin.n.01_2": "tin_lid_2",
                },
            }
        )
    for item, activity in enumerate(GROSS_MOVEMENT_ACTIVITIES, start=17):
        tasks.append(
            {
                "item": item,
                "activity": activity,
                "objects": [mannequin()],
                "instances": {"mannequin.n.02_1": "mannequin"},
                "include_table": False,
                "include_lights": False,
            }
        )
    return tasks


def scene_config(
    objects: list[dict],
    task_metadata: dict | None = None,
    *,
    include_table: bool = True,
    include_lights: bool = True,
) -> dict:
    scene_objects = []
    if include_table:
        scene_objects.append(table())
    if include_lights:
        scene_objects.extend(light_cfgs())
    scene_objects.extend(deepcopy(objects))
    return {
        "env": {"automatic_reset": False},
        "scene": {
            "type": "Scene",
            "use_floor_plane": True,
            "floor_plane_visible": True,
            "floor_plane_color": [0.5, 0.5, 0.5],
            "use_skybox": True,
            "include_robots": False,
            "task_metadata": task_metadata or {},
        },
        "objects": scene_objects,
        "robots": [],
        "task": {"type": "DummyTask"},
    }


def normalize_scene_file(scene_file: dict) -> None:
    """Record the v1 dataset while preserving the authored plain Scene."""

    scene_file["versions"]["arat-assets"] = {"version": 1, "dataset_name": DATASET_NAME}


def normalize_saved_scene(path: Path, task_metadata: dict | None = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    normalize_scene_file(data)
    # behavior_arat currently carries an older OmniGibson serializer that
    # drops Scene(task_metadata=...) from saved JSON. Preserve it explicitly
    # so each raw template is independently usable by an offline BehaviorTask.
    data.setdefault("metadata", {})["task"] = deepcopy(task_metadata or {})
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    return data


def add_four_ounces_of_water(env) -> dict:
    water = env.scene.get_system("water")
    radius = float(water.particle_radius)
    center_x, center_y = CUP_CENTER_X, TUMBLER_1_CENTER_Y
    spacing = 2.0 * radius
    z0 = ARAT_BOX_COVER_SURFACE_Z + CONTACT_CLEARANCE + 0.004 + radius
    positions = []
    # A 3 x 3 lattice fits inside the cup's narrow lower interior. Five
    # layers provide enough samples; select the nearest integer particle count
    # to the exact 4-US-fl-oz volume under OmniGibson's cube-volume convention.
    for layer in range(5):
        z = z0 + layer * spacing
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                positions.append([center_x + ix * spacing, center_y + iy * spacing, z])
    positions = positions[:WATER_PROXY_PARTICLES]
    if len(positions) != WATER_PROXY_PARTICLES:
        raise AssertionError("Failed to generate the requested ARAT water proxy")
    water.generate_particles(positions=th.tensor(positions, dtype=th.float32))
    for _ in range(10):
        og.sim.step()
    represented_ml = WATER_PROXY_PARTICLES * (2.0 * radius) ** 3 * 1e6
    return {
        "system": "water",
        "required_us_fluid_ounces": WATER_US_FLUID_OUNCES,
        "required_volume_ml": WATER_VOLUME_ML,
        "proxy_particle_count": WATER_PROXY_PARTICLES,
        "particle_radius_m": radius,
        "filled_state_proxy_volume_ml": represented_ml,
        "relative_volume_error_percent": 100.0 * (represented_ml - WATER_VOLUME_ML) / WATER_VOLUME_ML,
        "note": (
            "The exact 118.29411825 mL requirement is authoritative. Forty-one fixed-radius particles are the nearest "
            "volume representable by OmniGibson's Filled-state cube-volume convention."
        ),
    }


def save_scene(config: dict, path: Path, add_water: bool = False) -> dict | None:
    env = og.Environment(configs=config)
    box = env.scene.object_registry("name", "arat_box")
    if box is not None:
        box.joints["front_cover_joint"].friction = 20.0
        for joint in box.joints.values():
            joint.set_pos(0.0)
            joint.set_vel(0.0)
    water_info = add_four_ounces_of_water(env) if add_water else None
    # Environment initialization briefly runs physics. Restore every movable
    # task object to its exact requested initial pose before serialization.
    for obj_cfg in config["objects"]:
        if obj_cfg.get("type") == "DatasetObject" and not obj_cfg.get("fixed_base", False):
            movable_obj = env.scene.object_registry("name", obj_cfg["name"])
            movable_obj.set_position_orientation(
                position=th.tensor(obj_cfg["position"], dtype=th.float32),
                orientation=th.tensor(obj_cfg["orientation"], dtype=th.float32),
            )
            movable_obj.keep_still()
    path.parent.mkdir(parents=True, exist_ok=True)
    env.scene.save(str(path))
    normalize_saved_scene(path, task_metadata=config["scene"].get("task_metadata"))
    return water_info


def intended_layout_record(task: dict, water_info: dict | None) -> dict:
    objects = []
    layout_objects = [*([table()] if task.get("include_table", True) else []), *task["objects"]]
    for obj in layout_objects:
        objects.append(
            {
                "name": obj["name"],
                "category": obj["category"],
                "model": obj["model"],
                "position_xyz_m": obj["position"],
                "orientation_xyzw": obj["orientation"],
                "fixed_base": bool(obj.get("fixed_base", False)),
            }
        )
    record = {
        "arat_item": task["item"],
        "activity": task["activity"],
        "coordinate_system": "world X proximal-to-distal, world Y left-to-right from subject, world Z up",
        "objects": objects,
        "common_height_increase_m": COMMON_TABLE_HEIGHT_INCREASE,
        "common_distance_increase_m": COMMON_TABLE_DISTANCE_INCREASE,
    }
    if water_info is not None:
        record["water"] = water_info
    return record


def main() -> None:
    gm.USE_GPU_DYNAMICS = True
    base_best = ASSET_SCENE_DIR / "json" / f"{SCENE_MODEL}_best.json"
    save_scene(scene_config([]), base_best)

    # Match the challenge dataset convention: one stable scene plus one task
    # template for each activity.
    stable = INSTANCE_SCENE_DIR / "json" / f"{SCENE_MODEL}_stable.json"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_text(base_best.read_text(encoding="utf-8"), encoding="utf-8")

    tasks = build_tasks()
    layout_records = []
    for index, task in enumerate(tasks):
        if og.sim is not None:
            og.clear()
        inst_to_name = {"agent.n.01_1": "franka_sharpa_right", **task["instances"]}
        if task.get("include_table", True):
            inst_to_name["breakfast_table.n.01_1"] = "table"
        if task["item"] == 7:
            inst_to_name["water.n.06_1"] = "water"
        metadata = {
            "activity": task["activity"],
            "arat_item": task["item"],
            "arat_asset_version": 1,
            "source": "ARAT_specification.md",
            "tested_side": "right",
            "inst_to_name": inst_to_name,
            "common_height_increase_m": COMMON_TABLE_HEIGHT_INCREASE,
            "common_distance_increase_m": COMMON_TABLE_DISTANCE_INCREASE,
        }
        path = INSTANCE_SCENE_DIR / "json" / f"{SCENE_MODEL}_task_{task['activity']}_0_0_template.json"
        water_info = save_scene(
            scene_config(
                task["objects"],
                metadata,
                include_table=task.get("include_table", True),
                include_lights=task.get("include_lights", True),
            ),
            path,
            add_water=task["item"] == 7,
        )
        layout_records.append(intended_layout_record(task, water_info))
        print(f"ARAT_TASK_INSTANCE_GENERATED {index + 1}/{len(tasks)} item={task['item']} activity={task['activity']}")

    constraints = {
        "asset_version": 1,
        "authority": "ARAT_specification.md",
        "scene_model": SCENE_MODEL,
        "tested_side": "right",
        "table": {
            "height_m": TABLE_SURFACE_Z,
            "width_m": 0.76,
            "depth_m": TABLE_DEPTH,
            "proximal_edge_world_x_m": TABLE_PROXIMAL_X,
            "distal_edge_world_x_m": TABLE_DISTAL_X,
            "center_world_y_m": TABLE_CENTER_Y,
            "proximal_edge_distance_from_robot_m": TABLE_PROXIMAL_DISTANCE_FROM_ROBOT,
            "height_increase_from_reference_m": COMMON_TABLE_HEIGHT_INCREASE,
            "distance_increase_from_reference_m": COMMON_TABLE_DISTANCE_INCREASE,
            "nominal_arat_depth_m": 0.49,
            "depth_extension_reason": "Comfortably support the full articulated ARAT box shell on the tabletop.",
        },
        "ergonomic_alignment": {
            "robot_position_xyz_m": ROBOT_POSITION,
            "robot_pose_changed": False,
            "robot_shoulder_height_from_root_m": ROBOT_SHOULDER_HEIGHT_FROM_ROOT,
            "human_reference_seat_height_m": 0.46,
            "human_reference_shoulder_height_m": 1.033,
            "human_reference_table_height_m": 0.75,
            "shoulder_to_table_offset_m": HUMAN_SHOULDER_TO_TABLE,
            "realized_robot_shoulder_to_table_offset_m": (
                HUMAN_SHOULDER_TO_TABLE - COMMON_TABLE_HEIGHT_INCREASE
            ),
        },
        "common_layout_adjustment": {
            "height_increase_m": COMMON_TABLE_HEIGHT_INCREASE,
            "distance_increase_from_robot_m": COMMON_TABLE_DISTANCE_INCREASE,
            "robot_to_table_proximal_edge_distance_m": TABLE_PROXIMAL_DISTANCE_FROM_ROBOT,
            "table_feet_remain_on_floor": True,
            "robot_pose_changed": False,
            "applies_to_all_table_tasks": True,
            "gross_movement_mannequin_height_increase_m": COMMON_TABLE_HEIGHT_INCREASE,
            "gross_movement_mannequin_distance_increase_m": COMMON_TABLE_DISTANCE_INCREASE,
        },
        "target_box": {
            "category": "arat_box",
            "model": "aratbx",
            "root_xyz_m": ARAT_BOX_ROOT,
            "cover_proximal_edge_world_x_m": ARAT_BOX_COVER_PROXIMAL_X,
            "cover_distal_edge_world_x_m": ARAT_BOX_COVER_DISTAL_X,
            "cover_proximal_edge_from_table_proximal_edge_m": FRONT_EDGE_DISTANCE,
            "center_world_y_m": ARAT_BOX_CENTER_Y,
            "rightmost_edge_world_y_m": ARAT_BOX_RIGHTMOST_Y,
            "shell_surface_height_above_table_m": ARAT_BOX_SHELL_SURFACE_Z - TABLE_SURFACE_Z,
            "shell_proximal_edge_world_x_m": ARAT_BOX_ROOT[0],
            "shell_distal_edge_world_x_m": ARAT_BOX_SHELL_DISTAL_X,
            "back_support_margin_m": ARAT_BOX_BACK_SUPPORT_MARGIN,
            "cover_holding_friction": 20.0,
            "cover_joint_zero_angle_deg": 0.0,
            "cover_joint_drop_m": ARAT_BOX_COVER_JOINT_DROP,
            "cover_bottom_clearance_above_table_m": 0.0,
            "cover_surface_slope_deg": 0.0,
        },
        "block_placement": {
            "front_edge_from_table_proximal_edge_m": FRONT_EDGE_DISTANCE,
            "box_rightmost_edge_world_y_m": ARAT_BOX_RIGHTMOST_Y,
            "centroid_from_box_rightmost_edge_m": RIGHT_EDGE_TO_CENTROID,
            "centroid_world_y_m": BLOCK_CENTROID_Y,
            "supported_by": "arat_box lid",
        },
        "tin_transfer_placement": {
            "tin_lid_1_name": "tin_lid_1",
            "tin_lid_1_front_edge_from_table_proximal_edge_m": FRONT_EDGE_DISTANCE,
            "tin_lid_1_center_xyz_m": [
                TIN_LID_1_CENTER_X,
                TIN_LID_1_CENTER_Y,
                TIN_LID_1_CENTER_Z + TIN_FIXED_SUPPORT_CLEARANCE,
            ],
            "tin_lid_1_orientation_xyzw": TIN_LID_1_ORIENTATION,
            "tin_lid_2_name": "tin_lid_2",
            "tin_lid_2_front_edge_world_x_m": TIN_LID_2_CENTER_X - TIN_LID_RADIUS,
            "tin_lid_2_front_edge_from_box_top_front_edge_m": 0.0,
            "tin_lid_2_center_xyz_m": [
                TIN_LID_2_CENTER_X,
                TIN_LID_2_CENTER_Y,
                ARAT_BOX_SHELL_SURFACE_Z + TIN_FIXED_SUPPORT_CLEARANCE,
            ],
            "box_rightmost_edge_world_y_m": ARAT_BOX_RIGHTMOST_Y,
            "lid_centroids_from_box_rightmost_edge_m": RIGHT_EDGE_TO_CENTROID,
            "lids_fixed": True,
            "fixed_support_clearance_m": TIN_FIXED_SUPPORT_CLEARANCE,
            "applies_to": ["arat_grasp_cricket_ball", "arat_pinch_ball_bearing_*", "arat_pinch_marble_*"],
        },
        "sharpening_stone_placement": {
            "front_edge_from_table_proximal_edge_m": FRONT_EDGE_DISTANCE,
            "box_rightmost_edge_world_y_m": ARAT_BOX_RIGHTMOST_Y,
            "centroid_from_box_rightmost_edge_m": RIGHT_EDGE_TO_CENTROID,
            "centroid_world_xyz_m": [
                STONE_CENTROID_X,
                STONE_CENTROID_Y,
                ARAT_BOX_COVER_SURFACE_Z + 0.013,
            ],
            "root_world_xyz_m": [STONE_ROOT_X, STONE_ROOT_Y, ARAT_BOX_COVER_SURFACE_Z + 0.013],
            "orientation_xyzw": STONE_NARROW_SIDE_DIAGONAL,
            "world_aabb_extent_x_m": STONE_WORLD_AABB_EXTENT_X,
            "distal_edge_world_x_m": STONE_DISTAL_EDGE_X,
            "supported_by": "arat_box lid",
        },
        "pouring_placement": {
            "box_horizontal_midline_world_y_m": ARAT_BOX_CENTER_Y,
            "tumbler_1_center_xyz_m": [
                CUP_CENTER_X,
                TUMBLER_1_CENTER_Y,
                ARAT_BOX_COVER_SURFACE_Z + CONTACT_CLEARANCE,
            ],
            "tumbler_2_center_xyz_m": [
                CUP_CENTER_X,
                TUMBLER_2_CENTER_Y,
                ARAT_BOX_COVER_SURFACE_Z + CONTACT_CLEARANCE,
            ],
            "inner_edges_from_midline_m": CUP_INNER_EDGE_FROM_BOX_MIDLINE,
            "front_edges_from_table_proximal_edge_m": CUP_FRONT_EDGE_DISTANCE,
            "tumbler_1_contains_water": True,
        },
        "tube_placement": {
            "plank_starting_dimensions_depth_width_height_m": [0.06, 0.085, 0.015],
            "plank_target_dimensions_depth_width_height_m": [0.085, 0.34, 0.035],
            "start_bolt_centroid_from_table_proximal_edge_m": 0.08,
            "start_bolt_centroid_from_box_rightmost_edge_m": RIGHT_EDGE_TO_CENTROID,
            "start_bolt_center_xy_m": [START_BOLT_CENTER_X, START_BOLT_CENTER_Y],
            "start_plank_front_edge_from_table_proximal_edge_m": 0.08 - START_PLANK_DEPTH / 2.0,
            "start_plank_front_overhang_beyond_lid_m": FRONT_EDGE_DISTANCE - (0.08 - START_PLANK_DEPTH / 2.0),
            "target_plank_back_edge_aligned_with_lid_back_edge": False,
            "target_plank_forward_shift_from_lid_back_edge_m": TARGET_PLANK_FORWARD_SHIFT,
            "target_plank_clears_black_hinge": True,
            "target_plank_right_edge_aligned_with_box_rightmost_edge": True,
            "target_bolt_center_xy_m": [TARGET_BOLT_CENTER_X, TARGET_BOLT_CENTER_Y],
            "target_bolt_from_target_plank_right_edge_m": RIGHT_EDGE_TO_CENTROID,
            "large_bolts_diameter_start_height_target_height_m": [0.02, 0.135, 0.08],
            "small_bolts_diameter_start_height_target_height_m": [0.008, 0.06, 0.06],
            "fixed_members": [
                "plank_starting_point",
                "bolt_starting_point",
                "plank_target_point",
                "bolt_target_point",
            ],
            "tube_initially_pegged_on_start_bolt": True,
        },
        "washer_placement": {
            "tin_lid_front_edge_from_table_proximal_edge_m": FRONT_EDGE_DISTANCE,
            "tin_lid_centroid_from_box_rightmost_edge_m": RIGHT_EDGE_TO_CENTROID,
            "washer_centered_inside_tin_lid": True,
            "plank_target_dimensions_depth_width_height_m": [0.085, 0.085, 0.015],
            "plank_target_back_edge_aligned_with_lid_back_edge": False,
            "plank_target_forward_shift_from_lid_back_edge_m": TARGET_PLANK_FORWARD_SHIFT,
            "plank_target_clears_black_hinge": True,
            "bolt_target_diameter_height_m": [0.008, 0.085],
            "bolt_target_center_xy_m": [WASHER_PLANK_CENTER_X, WASHER_PLANK_CENTER_Y],
            "bolt_target_centroid_from_box_rightmost_edge_m": RIGHT_EDGE_TO_CENTROID,
            "fixed_members": ["tin_lid", "plank_target_point", "bolt_target_point"],
        },
        "gross_movement_placement": {
            "activities": list(GROSS_MOVEMENT_ACTIVITIES),
            "only_scene_object": "mannequin/nphsfp",
            "fixed_base": True,
            "fixed_base_reason": "The unsupported upright mannequin topples under physics.",
            "target_height_m": ARAT_BOX_TOTAL_HEIGHT_FROM_FLOOR,
            "height_datum": "floor to top of articulated ARAT box handle in items 1-16",
            "target_world_xy_aabb_center_m": ARAT_BOX_AABB_CENTER_XY,
            "root_xyz_m": MANNEQUIN_ROOT,
            "orientation_xyzw": MANNEQUIN_ORIENTATION,
            "uniform_scale": MANNEQUIN_SCALE,
            "floor_contact_world_z_m": 0.0,
            "head_at_top": True,
            "faces_world_direction": "-X (robot side)",
        },
        "tasks": layout_records,
    }
    INSTANCE_ROOT.mkdir(parents=True, exist_ok=True)
    (INSTANCE_ROOT / "layout_manifest.json").write_text(json.dumps(constraints, indent=2) + "\n", encoding="utf-8")
    print(f"ARAT_TASK_INSTANCE_GENERATION_PASSED tasks={len(tasks)} manifest={INSTANCE_ROOT / 'layout_manifest.json'}")
    og.shutdown()


if __name__ == "__main__":
    main()

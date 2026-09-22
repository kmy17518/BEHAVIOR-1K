"""Finger-tracking bench: a fixed right Sharpa hand above a table (no wrist tracking)."""

from dex_teleop.hand_bench.scene import (
    CAMERA_LAYOUT,
    DEFAULT_CAMERA_RIG,
    HAND_BENCH_PATH,
    ROBOT_NAME,
    ROBOT_PRIM_PATH,
    TABLE_NAME,
    ExpectedHandPose,
    HandBenchRobot,
    HandBenchScene,
    HandBenchTable,
    build_hand_bench_config,
    hand_pose_error,
    hidden_prim_names,
    load_hand_bench_scene,
    reset_joint_positions,
    validate_runtime_assets,
)

__all__ = [
    "CAMERA_LAYOUT",
    "DEFAULT_CAMERA_RIG",
    "HAND_BENCH_PATH",
    "ROBOT_NAME",
    "ROBOT_PRIM_PATH",
    "TABLE_NAME",
    "ExpectedHandPose",
    "HandBenchRobot",
    "HandBenchScene",
    "HandBenchTable",
    "build_hand_bench_config",
    "hand_pose_error",
    "hidden_prim_names",
    "load_hand_bench_scene",
    "reset_joint_positions",
    "validate_runtime_assets",
]

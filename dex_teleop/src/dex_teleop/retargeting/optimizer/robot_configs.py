"""Kinematic link configuration for the supported robot hands.

Adapted from AnyDexRetarget (MIT License, Copyright (c) 2025 Shiquan Qiu).
See ``THIRD_PARTY_NOTICES.md``.
"""

ROBOT_CONFIGS = {
    "shadow": {
        "origin_link": "rh_palm",
        "tip_links": ["rh_thtip", "rh_fftip", "rh_mftip", "rh_rftip", "rh_lftip"],
        "link1_names": [
            "rh_thproximal",
            "rh_ffproximal",
            "rh_mfproximal",
            "rh_rfproximal",
            "rh_lfproximal",
        ],
        "link3_names": ["rh_thmiddle", "rh_ffmiddle", "rh_mfmiddle", "rh_rfmiddle", "rh_lfmiddle"],
        "link4_names": ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"],
        "urdf_subdir": "urdf/shadow",
        "urdf_file": {"right": "right_hand_mj.urdf", "left": "left_hand_mj.urdf"},
        "num_fingers": 5,
    },
    "sharpa": {
        "origin_link": "right_hand_C_MC",
        "tip_links": [
            "right_thumb_fingertip",
            "right_index_fingertip",
            "right_middle_fingertip",
            "right_ring_fingertip",
            "right_pinky_fingertip",
        ],
        "link1_names": [
            "right_thumb_MC",
            "right_index_PP",
            "right_middle_PP",
            "right_ring_PP",
            "right_pinky_PP",
        ],
        "link3_names": [
            "right_thumb_PP",
            "right_index_MP",
            "right_middle_MP",
            "right_ring_MP",
            "right_pinky_MP",
        ],
        "link4_names": [
            "right_thumb_DP",
            "right_index_DP",
            "right_middle_DP",
            "right_ring_DP",
            "right_pinky_DP",
        ],
        "urdf_subdir": "urdf/sharpa",
        "urdf_file": {"right": "right_sharpa_wave.urdf", "left": "left_sharpa_wave.urdf"},
        "num_fingers": 5,
        "neutral_qpos": [0.0] * 22,
    },
    "wuji": {
        "origin_link": "right_palm_link",
        "tip_links": [f"right_finger{index}_tip_link" for index in range(1, 6)],
        "link1_names": [f"right_finger{index}_link1" for index in range(1, 6)],
        "link3_names": [f"right_finger{index}_link3" for index in range(1, 6)],
        "link4_names": [f"right_finger{index}_link4" for index in range(1, 6)],
        "urdf_subdir": "urdf/wuji",
        "urdf_file": {"right": "right.urdf", "left": "left.urdf"},
        "num_fingers": 5,
    },
}

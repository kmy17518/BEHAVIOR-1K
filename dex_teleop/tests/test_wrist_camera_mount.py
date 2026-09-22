"""Static kinematic and calibration checks for the Franka + Sharpa wrist cameras."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from dex_teleop.arat.camera_rig import load_camera_rig, opencv_fisheye_parameters


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = (
    REPOSITORY_ROOT
    / "datasets"
    / "omnigibson-robot-assets"
    / "models"
    / "franka"
    / "franka_dexhand"
    / "franka_sharpa_right"
)
BASE_URDF = ASSET_ROOT / "urdf" / "franka_sharpa_right.urdf"
CAMERA_URDF = ASSET_ROOT / "urdf" / "franka_sharpa_right_with_wrist_cameras.urdf"
EXTRINSICS = ASSET_ROOT / "source" / "wrist_camera_extrinsics.yaml"
CAMERA_LINKS = {
    "right_wrist_camera_mount",
    "right_wrist_camera_thumb",
    "right_wrist_camera_thumb_optical",
    "right_wrist_camera_pinky",
    "right_wrist_camera_pinky_optical",
}


def _joint_signature(root: ET.Element, *, movable_only: bool) -> list[tuple]:
    signature = []
    for joint in root.findall("joint"):
        if movable_only and joint.get("type") == "fixed":
            continue
        signature.append(
            (
                joint.get("name"),
                joint.get("type"),
                joint.find("parent").get("link"),
                joint.find("child").get("link"),
                _attributes(joint, "origin"),
                _attributes(joint, "axis"),
                _attributes(joint, "limit"),
                _attributes(joint, "mimic"),
            )
        )
    return signature


def _attributes(parent: ET.Element, tag: str) -> tuple:
    element = parent.find(tag)
    return () if element is None else tuple(sorted(element.attrib.items()))


def _joint(root: ET.Element, name: str) -> ET.Element:
    matches = [joint for joint in root.findall("joint") if joint.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _vector(element: ET.Element, attribute: str) -> np.ndarray:
    return np.array([float(value) for value in element.get(attribute).split()])


def _element_signature(element: ET.Element) -> tuple:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        tuple((child.tag, tuple(sorted(child.attrib.items()))) for child in element),
    )


def test_camera_variant_preserves_original_kinematics_and_one_tree():
    base_root = ET.parse(BASE_URDF).getroot()
    camera_root = ET.parse(CAMERA_URDF).getroot()

    assert _joint_signature(camera_root, movable_only=True) == _joint_signature(base_root, movable_only=True)
    assert len(_joint_signature(camera_root, movable_only=True)) == 29
    assert [ET.tostring(node) for node in camera_root.findall("transmission")] == [
        ET.tostring(node) for node in base_root.findall("transmission")
    ]

    base_flange = _joint(base_root, "panda_right_hand_flange_joint")
    camera_flange = _joint(camera_root, "panda_right_hand_flange_joint")
    assert _element_signature(camera_flange) == _element_signature(base_flange)

    links = {link.get("name") for link in camera_root.findall("link")}
    children = {joint.find("child").get("link") for joint in camera_root.findall("joint")}
    assert links.difference(children) == {"panda_base"}
    assert CAMERA_LINKS.issubset(links)

    expected_camera_joints = {
        "right_wrist_camera_mount_joint": ("right_hand_C_MC", "right_wrist_camera_mount"),
        "right_wrist_camera_thumb_joint": ("right_wrist_camera_mount", "right_wrist_camera_thumb"),
        "right_wrist_camera_thumb_optical_joint": (
            "right_wrist_camera_thumb",
            "right_wrist_camera_thumb_optical",
        ),
        "right_wrist_camera_pinky_joint": ("right_wrist_camera_mount", "right_wrist_camera_pinky"),
        "right_wrist_camera_pinky_optical_joint": (
            "right_wrist_camera_pinky",
            "right_wrist_camera_pinky_optical",
        ),
    }
    for name, (parent, child) in expected_camera_joints.items():
        joint = _joint(camera_root, name)
        assert joint.get("type") == "fixed"
        assert joint.find("parent").get("link") == parent
        assert joint.find("child").get("link") == child


def test_extrinsics_match_urdf_optical_and_omnigibson_frames():
    root = ET.parse(CAMERA_URDF).getroot()
    extrinsics = yaml.safe_load(EXTRINSICS.read_text(encoding="utf-8"))
    expected_forward = np.asarray(extrinsics["camera_defaults"]["principal_ray_in_parent"], dtype=float)
    assert np.allclose(expected_forward, [1.0, 0.0, 0.0])  # Sharpa palmar/table-facing normal
    body_to_optical = Rotation.from_euler(
        "xyz", extrinsics["thumb_camera"]["body_to_optical"]["rpy"]
    )
    assert np.allclose(
        extrinsics["thumb_camera"]["body_to_optical"]["rpy"],
        extrinsics["pinky_camera"]["body_to_optical"]["rpy"],
    )

    rig = load_camera_rig("arat_default")
    for side, camera_id in (("thumb", "thumb_wrist"), ("pinky", "pinky_wrist")):
        data = extrinsics[f"{side}_camera"]
        body_joint = _joint(root, f"right_wrist_camera_{side}_joint")
        body_origin = body_joint.find("origin")
        optical_origin = _joint(root, f"right_wrist_camera_{side}_optical_joint").find("origin")
        assert np.allclose(_vector(body_origin, "xyz"), data["xyz"])
        assert np.allclose(_vector(body_origin, "rpy"), data["rpy"])
        assert np.allclose(_vector(optical_origin, "xyz"), data["body_to_optical"]["xyz"])
        assert np.allclose(_vector(optical_origin, "rpy"), data["body_to_optical"]["rpy"])

        parent_to_body = Rotation.from_euler("xyz", data["rpy"])
        parent_to_optical = parent_to_body * body_to_optical
        assert np.dot(parent_to_optical.apply([0.0, 0.0, 1.0]), expected_forward) > 1.0 - 1.0e-10

        # Convert ROS optical (+Z forward, +Y down) to USD Camera (-Z forward, +Y up).
        parent_to_usd_camera = parent_to_optical * Rotation.from_euler("x", math.pi)
        configured_usd = Rotation.from_quat(data["omnigibson_camera"]["orientation_xyzw"])
        assert (parent_to_usd_camera.inv() * configured_usd).magnitude() < 1.0e-10

        rig_camera = rig.camera(camera_id)
        assert rig_camera["parent"] == {"frame": "robot_link", "link": "right_hand_C_MC"}
        assert np.allclose(rig_camera["pose"]["position"], data["omnigibson_camera"]["position"])
        assert np.allclose(
            rig_camera["pose"]["orientation_xyzw"],
            data["omnigibson_camera"]["orientation_xyzw"],
        )
        fisheye = opencv_fisheye_parameters(rig, camera_id)
        assert fisheye is not None
        assert fisheye["horizontal_fov_degrees"] == extrinsics["camera_defaults"]["horizontal_fov_degrees"]
        assert fisheye["distortion_coefficients"] == [0.0] * 4

        calibration = rig.calibration(camera_id)
        fallback_fov = math.degrees(
            2.0 * math.atan(calibration["horizontal_aperture"] / (2.0 * calibration["focal_length"]))
        )
        assert abs(fallback_fov - extrinsics["camera_defaults"]["rectilinear_fallback_fov_degrees"]) < 0.01


def test_robot_definition_selects_non_destructive_camera_variants():
    definition = yaml.safe_load((ASSET_ROOT.parents[1] / "franka.yaml").read_text(encoding="utf-8"))
    sharpa = definition["manipulation"]["end_effectors"]["sharpa_right"]

    assert sharpa["usd_path"].endswith("franka_sharpa_right_with_wrist_cameras.usda")
    assert sharpa["urdf_path"].endswith("franka_sharpa_right_with_wrist_cameras.urdf")
    assert BASE_URDF.is_file()
    assert (ASSET_ROOT / "usd" / "franka_sharpa_right.usda").is_file()

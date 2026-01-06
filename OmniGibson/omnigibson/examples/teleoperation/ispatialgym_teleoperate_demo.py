import json
import logging
import os
import time
import cv2
import numpy as np
import torch as th
from gello.robots.sim_robot.og_teleop_utils import generate_robot_config, load_available_tasks

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    flatten_obs_dict,
    generate_ispatialgym_environment_config,
)
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import KeyboardEventHandler, KeyboardRobotController, choose_from_options


logger = logging.getLogger("ispatialgym_teleoperate_demo")
logger.setLevel(logging.INFO)

KNOWN_CAMERA_POSES = {
    "relation_satisfaction_on_top_pepper_shaker_countertop": (th.tensor([11.9224,  3.1055,  2.0538]), th.tensor([0.4018, 0.5475, 0.5917, 0.4343])),
    "relation_satisfaction_inside_toothbrush_beaker": (th.tensor([ 2.1993, 21.4388,  2.3746]), th.tensor([0.2634, 0.2587, 0.6512, 0.6630])),
    "relation_satisfaction_under_tray_bowl": (th.tensor([1.0729, 0.5542, 2.2308]), th.tensor([ 0.5339, -0.0078, -0.0123,  0.8454])),
}


def compute_ceiling_height(scene, robot_pos, default_height=2.5, offset_below_ceiling=0.3):
    """
    Compute the ceiling height for camera placement.
    
    Args:
        scene: The scene object
        robot_pos: The robot's position tensor
        default_height (float): Default ceiling height if no ceiling is found (default 2.5m)
        offset_below_ceiling (float): How far below the ceiling to place the camera (default 0.3m)
    
    Returns:
        float: The computed ceiling height for camera placement
    """
    try:
        # Try to find ceiling objects in the scene
        # Ceilings are typically named "ceilings" or have category "ceilings"
        ceiling_obj = None
        
        # Try to get ceiling by name first
        try:
            ceiling_obj = scene.object_registry("name", "ceilings")
        except (KeyError, ValueError):
            pass
        
        # If not found by name, try to find by category
        if ceiling_obj is None:
            try:
                ceiling_objs = scene.object_registry("category", "ceilings", [])
                if ceiling_objs:
                    ceiling_obj = ceiling_objs[0] if isinstance(ceiling_objs, list) else ceiling_objs
            except (KeyError, ValueError):
                pass
        
        if ceiling_obj is not None and hasattr(ceiling_obj, 'aabb'):
            # Get the ceiling's bounding box
            aabb_lo, aabb_hi = ceiling_obj.aabb
            # The ceiling's bottom surface is at aabb_lo[2]
            ceiling_z = aabb_lo[2].item() if hasattr(aabb_lo[2], 'item') else aabb_lo[2]
            # Place camera slightly below the ceiling
            camera_height = ceiling_z - offset_below_ceiling
            logger.info(f"Ceiling height computed from ceiling object: {ceiling_z:.2f}m, camera height: {camera_height:.2f}m")
            return camera_height
        
        # Fallback: try to infer from floor height if available
        if hasattr(scene, 'get_floor_height'):
            floor_height = scene.get_floor_height(0)
            # Assume typical room height of 2.5-3m
            estimated_ceiling = floor_height + default_height
            logger.info(f"Ceiling height estimated from floor height: {estimated_ceiling:.2f}m")
            return estimated_ceiling - offset_below_ceiling
            
    except Exception as e:
        logger.warning(f"Failed to compute ceiling height: {e}")
    
    # Default fallback
    logger.info(f"Using default ceiling height: {default_height}m")
    return default_height - offset_below_ceiling


def compute_auto_camera_pose(
    robot, 
    scene, 
    # Candidate generation parameters
    n_azimuth_samples=24,
    n_elevation_samples=4,
    n_radius_samples=3,
    elevation_range=(10.0, 40.0),  # degrees
    # Scoring weights
    w_visibility=1.0,
    w_occlusion=1.5,
    w_angle=0.5,
    w_occupancy=0.3,
    w_uprightness=0.2,
    # Constraints
    frame_margin=0.1,
    min_occlusion_fraction=0.7,
    min_clearance=0.3,
    target_occupancy_range=(0.08, 0.25),  # Smaller for mobile manipulation (more context visible)
    preferred_angle_range=(30.0, 60.0),  # degrees from robot facing direction
    distance_margin_factor=1.5,
    # FOV and camera parameters
    viewer_fov_deg=None,  # If None, computed from viewer camera
):
    """
    Automatically compute a good viewer camera pose using a sample-and-score approach.
    
    This function generates candidate camera poses on a hemisphere around the robot,
    scores each based on multiple criteria, and returns the best one.
    
    Hard constraints (must pass):
        - Robot keypoints fully in frame with margin
        - Sufficient unblocked rays (occlusion check)
        - Camera not inside geometry (clearance check)
        - Camera within safe distance band from robot
        
    Optimization criteria:
        - Three-quarter view (30-60° from robot facing direction)
        - Slightly elevated, looking down (-15° to -35° pitch)
        - Meaningful look-at point (weighted between base and end-effector)
        - Reasonable screen-space occupancy (8-25% for mobile manipulation)
        - Stable "up" alignment (no roll)
    
    Args:
        robot: The robot object
        scene: The scene object
        n_azimuth_samples: Number of azimuth angle samples around the robot
        n_elevation_samples: Number of elevation angle samples
        n_radius_samples: Number of distance samples
        elevation_range: (min, max) elevation angles in degrees
        w_visibility: Weight for visibility score
        w_occlusion: Weight for occlusion score
        w_angle: Weight for preferred viewing angle score
        w_occupancy: Weight for screen occupancy score
        w_uprightness: Weight for camera uprightness score
        frame_margin: Margin from frame edge (0-0.5, e.g., 0.1 = 10% from edges)
        min_occlusion_fraction: Minimum fraction of unblocked rays required
        min_clearance: Minimum clearance around camera position in meters
        target_occupancy_range: (min, max) target robot screen occupancy fraction (smaller for mobile manipulation)
        preferred_angle_range: (min, max) preferred angle from robot facing direction
        distance_margin_factor: Multiplier for ideal viewing distance
        viewer_fov_deg: Viewer camera field of view in degrees (auto-computed if None)
    
    Returns:
        tuple: (position, orientation) tensors for the camera pose
    """
    import math
    from omnigibson.utils.sampling_utils import raytest, raytest_batch
    from omnigibson.robots.manipulation_robot import ManipulationRobot
    
    # =========================================================================
    # Helper Functions
    # =========================================================================
    
    def get_robot_keypoints(robot):
        """
        Extract meaningful keypoints from the robot for visibility and occlusion checks.
        Returns a list of 3D points in world coordinates.
        """
        keypoints = []
        
        # Get base position
        base_pos, base_quat = robot.get_position_orientation()
        keypoints.append(base_pos.clone())
        
        # Get robot AABB for additional keypoints
        try:
            aabb_lo, aabb_hi = robot.aabb
            center = (aabb_lo + aabb_hi) / 2
            extent = aabb_hi - aabb_lo
            
            # Add AABB corners and center points
            keypoints.append(center)  # Center
            keypoints.append(th.tensor([center[0], center[1], aabb_hi[2]]))  # Top center
            keypoints.append(th.tensor([center[0], center[1], aabb_lo[2] + 0.1]))  # Near base
            
            # Add mid-height points at front/back/sides
            mid_z = (aabb_lo[2] + aabb_hi[2]) / 2
            keypoints.append(th.tensor([aabb_lo[0], center[1], mid_z]))
            keypoints.append(th.tensor([aabb_hi[0], center[1], mid_z]))
            keypoints.append(th.tensor([center[0], aabb_lo[1], mid_z]))
            keypoints.append(th.tensor([center[0], aabb_hi[1], mid_z]))
        except Exception:
            # Fallback: add points at estimated heights
            keypoints.append(base_pos + th.tensor([0, 0, 0.5]))
            keypoints.append(base_pos + th.tensor([0, 0, 1.0]))
            keypoints.append(base_pos + th.tensor([0, 0, 1.5]))
        
        # Get end-effector positions if robot has arms
        if isinstance(robot, ManipulationRobot):
            for arm in robot.arm_names:
                try:
                    eef_pos = robot.get_eef_position(arm=arm)
                    keypoints.append(eef_pos)
                except Exception:
                    pass
        
        return [kp if isinstance(kp, th.Tensor) else th.tensor(kp) for kp in keypoints]
    
    def get_robot_facing_direction(robot):
        """Get the forward direction the robot is facing (in XY plane)."""
        _, quat = robot.get_position_orientation()
        # Convert quaternion to forward direction
        # Robot typically faces +X in its local frame
        forward_local = th.tensor([1.0, 0.0, 0.0])
        rot_matrix = T.quat2mat(quat)
        forward_world = rot_matrix @ forward_local
        # Project to XY plane and normalize
        forward_xy = forward_world[:2]
        forward_xy = forward_xy / (th.norm(forward_xy) + 1e-6)
        return forward_xy
    
    def compute_robot_radius_and_height(robot):
        """Compute approximate robot radius and height from AABB."""
        try:
            aabb_lo, aabb_hi = robot.aabb
            extent = aabb_hi - aabb_lo
            radius = max(extent[0].item(), extent[1].item()) / 2
            height = extent[2].item()
            return radius, height
        except Exception:
            # Default values for humanoid robots
            return 0.5, 1.8
    
    def compute_ideal_distance(robot_radius, fov_deg, margin_factor=1.5):
        """
        Compute ideal viewing distance based on robot size and camera FOV.
        d ≈ (robot_radius / tan(FOV/2)) * margin_factor
        """
        fov_rad = math.radians(fov_deg)
        half_fov = fov_rad / 2
        ideal_dist = (robot_radius / math.tan(half_fov)) * margin_factor
        return max(ideal_dist, 1.5)  # Minimum 1.5m
    
    def compute_look_at_point(robot, base_weight=0.5):
        """
        Compute a meaningful look-at point.
        For manipulation: weighted between base and end-effector
        For navigation: base center
        """
        base_pos, _ = robot.get_position_orientation()
        look_at = base_pos.clone()
        
        # Raise look-at point to approximate torso height
        try:
            aabb_lo, aabb_hi = robot.aabb
            center_z = (aabb_lo[2] + aabb_hi[2]) / 2
            look_at[2] = center_z
        except Exception:
            look_at[2] = base_pos[2] + 0.8  # Default torso height
        
        # If manipulation robot, blend with end-effector
        if isinstance(robot, ManipulationRobot):
            eef_positions = []
            for arm in robot.arm_names:
                try:
                    eef_pos = robot.get_eef_position(arm=arm)
                    eef_positions.append(eef_pos)
                except Exception:
                    pass
            
            if eef_positions:
                avg_eef = th.stack(eef_positions).mean(dim=0)
                look_at = base_weight * look_at + (1 - base_weight) * avg_eef
        
        return look_at
    
    def compute_camera_orientation_from_look_at(camera_pos, look_at_point):
        """
        Compute camera orientation to look at a point with zero roll.
        Camera convention: -Z is forward (looking direction).
        """
        look_direction = look_at_point - camera_pos
        look_direction = look_direction / (th.norm(look_direction) + 1e-6)
        
        forward = -look_direction  # Camera -Z axis points away from look direction
        up_world = th.tensor([0.0, 0.0, 1.0])
        
        # Handle case where looking straight up/down
        if th.abs(th.dot(forward, up_world)) > 0.99:
            right = th.tensor([1.0, 0.0, 0.0])
        else:
            right = th.cross(up_world, forward)
            right = right / (th.norm(right) + 1e-6)
        
        up = th.cross(forward, right)
        up = up / (th.norm(up) + 1e-6)
        
        rot_matrix = th.stack([right, up, forward], dim=1)
        return T.mat2quat(rot_matrix)
    
    def project_points_to_image(points_3d, camera_pos, camera_quat, fov_deg, image_size=(448, 448)):
        """
        Project 3D points to normalized image coordinates [0, 1].
        Returns (u, v) coordinates where (0,0) is top-left.
        """
        # Transform points to camera frame
        cam_rot = T.quat2mat(camera_quat)
        cam_rot_inv = cam_rot.T
        
        projected = []
        for pt in points_3d:
            # Point in camera frame
            pt_cam = cam_rot_inv @ (pt - camera_pos)
            
            # Camera convention: -Z is forward, X is right, Y is up
            # Flip Z because camera looks along -Z
            x, y, z = pt_cam[0], pt_cam[1], -pt_cam[2]
            
            if z <= 0:
                # Point is behind camera
                projected.append((float('nan'), float('nan')))
                continue
            
            # Perspective projection
            fov_rad = math.radians(fov_deg)
            f = 1.0 / math.tan(fov_rad / 2)
            
            u = (x / z) * f
            v = (y / z) * f
            
            # Convert to normalized [0, 1] coordinates
            u_norm = (u + 1) / 2
            v_norm = (1 - v) / 2  # Flip Y for image coordinates
            
            projected.append((u_norm, v_norm))
        
        return projected
    
    def score_visibility(projected_points, margin=0.1):
        """
        Score based on what fraction of keypoints are within the frame with margin.
        Returns score in [0, 1].
        """
        in_frame = 0
        valid_points = 0
        
        for u, v in projected_points:
            if math.isnan(u) or math.isnan(v):
                continue
            valid_points += 1
            if margin <= u <= (1 - margin) and margin <= v <= (1 - margin):
                in_frame += 1
        
        if valid_points == 0:
            return 0.0
        return in_frame / valid_points
    
    def score_occlusion(camera_pos, keypoints, robot):
        """
        Score based on fraction of unblocked rays from camera to robot keypoints.
        Returns score in [0, 1].
        """
        unblocked = 0
        total = len(keypoints)
        
        if total == 0:
            return 0.0
        
        # Get robot prim paths to ignore in raycast
        robot_bodies = []
        try:
            for link in robot.links.values():
                robot_bodies.append(link.prim_path)
        except Exception:
            pass
        
        for kp in keypoints:
            result = raytest(
                start_point=camera_pos,
                end_point=kp,
                only_closest=True,
                ignore_bodies=robot_bodies,
            )
            
            if not result["hit"]:
                unblocked += 1
            else:
                # Check if hit is structural (walls, floors, ceilings)
                # We ignore these as they might be the room boundaries
                collision_path = result.get("collision", "") or result.get("rigidBody", "")
                structural_keywords = ["wall", "floor", "ceiling", "ground"]
                is_structural = any(kw in collision_path.lower() for kw in structural_keywords)
                
                # Also check distance - if hit is very close to keypoint, consider it unblocked
                dist_to_keypoint = th.norm(kp - camera_pos).item()
                if is_structural or result["distance"] > dist_to_keypoint * 0.95:
                    unblocked += 1
        
        return unblocked / total
    
    def score_viewing_angle(camera_pos, robot_pos, robot_facing, preferred_min=30.0, preferred_max=60.0):
        """
        Score based on how close the viewing angle is to the preferred three-quarter view.
        Returns score in [0, 1].
        """
        # Vector from robot to camera in XY plane
        to_camera = (camera_pos[:2] - robot_pos[:2])
        to_camera = to_camera / (th.norm(to_camera) + 1e-6)
        
        # Angle between robot facing and camera direction
        dot = th.dot(robot_facing, to_camera).clamp(-1, 1)
        angle_deg = math.degrees(math.acos(dot.item()))
        
        # Score: 1.0 if within preferred range, decreasing outside
        if preferred_min <= angle_deg <= preferred_max:
            return 1.0
        elif angle_deg < preferred_min:
            return max(0, 1.0 - (preferred_min - angle_deg) / preferred_min)
        else:
            return max(0, 1.0 - (angle_deg - preferred_max) / (180 - preferred_max))
    
    def score_screen_occupancy(projected_points, target_min=0.15, target_max=0.40):
        """
        Score based on approximate screen-space occupancy.
        Returns score in [0, 1].
        """
        valid_points = [(u, v) for u, v in projected_points if not (math.isnan(u) or math.isnan(v))]
        
        if len(valid_points) < 2:
            return 0.0
        
        us = [p[0] for p in valid_points]
        vs = [p[1] for p in valid_points]
        
        # Bounding box area in normalized coordinates
        width = max(us) - min(us)
        height = max(vs) - min(vs)
        area = width * height
        
        # Score: 1.0 if within target range
        target_mid = (target_min + target_max) / 2
        if target_min <= area <= target_max:
            return 1.0
        elif area < target_min:
            return max(0, area / target_min)
        else:
            return max(0, 1.0 - (area - target_max) / (1.0 - target_max))
    
    def score_uprightness(camera_quat):
        """
        Score based on how close the camera up vector is to world up (no roll).
        Returns score in [0, 1].
        """
        rot_matrix = T.quat2mat(camera_quat)
        camera_up = rot_matrix[:, 1]  # Y axis is up in camera frame
        world_up = th.tensor([0.0, 0.0, 1.0])
        
        # Dot product (1 = aligned, 0 = perpendicular)
        alignment = th.dot(camera_up, world_up).item()
        
        # Score: map from [-1, 1] to [0, 1], penalizing negative (inverted)
        return max(0, (alignment + 1) / 2)
    
    def check_clearance(camera_pos, clearance_radius=0.3):
        """
        Check if camera position has sufficient clearance from obstacles.
        Returns True if clear, False otherwise.
        """
        # Test rays in 6 cardinal directions
        directions = [
            th.tensor([1, 0, 0]), th.tensor([-1, 0, 0]),
            th.tensor([0, 1, 0]), th.tensor([0, -1, 0]),
            th.tensor([0, 0, 1]), th.tensor([0, 0, -1]),
        ]
        
        for direction in directions:
            end_point = camera_pos + direction.float() * clearance_radius
            result = raytest(
                start_point=camera_pos,
                end_point=end_point,
                only_closest=True,
            )
            if result["hit"] and result["distance"] < clearance_radius * 0.9:
                return False
        
        return True
    
    def get_ceiling_clearance(camera_pos, scene):
        """Get distance to ceiling above camera position."""
        try:
            result = raytest(
                start_point=camera_pos,
                end_point=camera_pos + th.tensor([0, 0, 10.0]),
                only_closest=True,
            )
            if result["hit"]:
                return result["distance"]
        except Exception:
            pass
        return 10.0  # Default large value
    
    # =========================================================================
    # Main Algorithm
    # =========================================================================
    
    logger.info("Computing auto camera pose using sample-and-score approach...")
    
    # Get robot info
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_radius, robot_height = compute_robot_radius_and_height(robot)
    robot_facing = get_robot_facing_direction(robot)
    keypoints = get_robot_keypoints(robot)
    look_at_point = compute_look_at_point(robot)
    
    logger.info(f"  Robot position: {robot_pos}")
    logger.info(f"  Robot radius: {robot_radius:.2f}m, height: {robot_height:.2f}m")
    logger.info(f"  Number of keypoints: {len(keypoints)}")
    
    # Determine FOV
    if viewer_fov_deg is None:
        # Compute from viewer camera parameters
        # FOV = 2 * atan(aperture / (2 * focal_length))
        # Default: focal_length=17mm, horizontal_aperture=20.995mm
        focal_length = 17.0
        horizontal_aperture = 20.995
        viewer_fov_deg = 2 * math.degrees(math.atan(horizontal_aperture / (2 * focal_length)))
    
    logger.info(f"  Viewer FOV: {viewer_fov_deg:.1f}°")
    
    # Compute distance band
    ideal_distance = compute_ideal_distance(robot_radius, viewer_fov_deg, distance_margin_factor)
    d_min = ideal_distance * 0.6
    d_max = ideal_distance * 1.8
    
    logger.info(f"  Distance band: {d_min:.2f}m - {d_max:.2f}m (ideal: {ideal_distance:.2f}m)")
    
    # Generate candidate poses
    candidates = []
    
    # Sample azimuth angles (around the robot)
    for i in range(n_azimuth_samples):
        azimuth = 2 * math.pi * i / n_azimuth_samples
        
        # Sample elevation angles
        for j in range(n_elevation_samples):
            elevation_deg = elevation_range[0] + (elevation_range[1] - elevation_range[0]) * j / max(1, n_elevation_samples - 1)
            elevation = math.radians(elevation_deg)
            
            # Sample radii
            for k in range(n_radius_samples):
                if n_radius_samples == 1:
                    radius = ideal_distance
                else:
                    radius = d_min + (d_max - d_min) * k / (n_radius_samples - 1)
                
                # Convert spherical to Cartesian
                # Elevation is from horizontal plane (0 = horizontal, 90 = straight up)
                x = radius * math.cos(elevation) * math.cos(azimuth)
                y = radius * math.cos(elevation) * math.sin(azimuth)
                z = radius * math.sin(elevation)
                
                camera_pos = robot_pos + th.tensor([x, y, z], dtype=th.float32)
                
                # Compute orientation to look at the look-at point
                camera_quat = compute_camera_orientation_from_look_at(camera_pos, look_at_point)
                
                candidates.append({
                    'pos': camera_pos,
                    'quat': camera_quat,
                    'azimuth': azimuth,
                    'elevation': elevation_deg,
                    'radius': radius,
                })
    
    logger.info(f"  Generated {len(candidates)} candidate poses")
    
    # Score each candidate
    best_score = -float('inf')
    best_candidate = None
    scores_log = []
    
    for cand in candidates:
        camera_pos = cand['pos']
        camera_quat = cand['quat']
        
        # Hard constraint: clearance check
        if not check_clearance(camera_pos, min_clearance):
            continue
        
        # Project keypoints
        projected = project_points_to_image(keypoints, camera_pos, camera_quat, viewer_fov_deg)
        
        # Compute scores
        v_score = score_visibility(projected, frame_margin)
        o_score = score_occlusion(camera_pos, keypoints, robot)
        a_score = score_viewing_angle(camera_pos, robot_pos, robot_facing, 
                                       preferred_angle_range[0], preferred_angle_range[1])
        s_score = score_screen_occupancy(projected, target_occupancy_range[0], target_occupancy_range[1])
        u_score = score_uprightness(camera_quat)
        
        # Hard constraint: minimum visibility
        if v_score < 0.5:
            continue
        
        # Hard constraint: minimum occlusion score
        if o_score < min_occlusion_fraction:
            continue
        
        # Weighted total score
        total_score = (
            w_visibility * v_score +
            w_occlusion * o_score +
            w_angle * a_score +
            w_occupancy * s_score +
            w_uprightness * u_score
        )
        
        scores_log.append({
            'score': total_score,
            'v': v_score, 'o': o_score, 'a': a_score, 's': s_score, 'u': u_score,
            'azimuth': cand['azimuth'], 'elevation': cand['elevation'], 'radius': cand['radius']
        })
        
        if total_score > best_score:
            best_score = total_score
            best_candidate = cand
    
    # Handle case where no valid candidate found
    if best_candidate is None:
        logger.warning("No valid camera pose found, using fallback")
        
        # Fallback: simple position behind and above robot
        fallback_offset = th.tensor([-2.5, -1.5, 1.5], dtype=th.float32)
        camera_pos = robot_pos + fallback_offset
        camera_quat = compute_camera_orientation_from_look_at(camera_pos, look_at_point)
        
        return camera_pos, camera_quat
    
    # Log best candidate info
    logger.info(f"  Best candidate score: {best_score:.3f}")
    logger.info(f"    Azimuth: {math.degrees(best_candidate['azimuth']):.1f}°")
    logger.info(f"    Elevation: {best_candidate['elevation']:.1f}°")
    logger.info(f"    Distance: {best_candidate['radius']:.2f}m")
    logger.info(f"    Position: {best_candidate['pos']}")
    
    return best_candidate['pos'], best_candidate['quat']

def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    controller_names = robot.controller_order
    for controller_name in controller_names:
        controller_options = default_config[controller_name]
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )

        # Add to user responses
        controller_choices[controller_name] = choice

    return controller_choices

def main(task_name: str, instance_id: int, quickstart: bool = False, record_video: bool = False, video_path: str = None, enable_camera_teleop: bool = False):
    """
    Teleoperate a robot in a iSpatialGym scene.
    """

    # load env
    available_tasks = load_available_tasks()
    task_cfg = available_tasks[task_name][0]
    cfg = generate_ispatialgym_environment_config(task_name=task_name, task_cfg=task_cfg)
    cfg["robots"] = [
        generate_robot_config(
            task_name=task_name,
            task_cfg=task_cfg,
        )
    ]
    # Enable sensors by setting observation modalities (required for camera-based tasks)
    cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
    
    # if quickstart: # only load the building assets (i.e.: the floors, walls, ceilings)
    #     cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]
    
    env = og.Environment(configs=cfg)

    # load robot 
    robot = env.robots[0] 

    # load task instance
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    tro_file_path = os.path.join(
        get_task_instance_path(scene_model),
        f"json/{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state.json",
    )
    with open(tro_file_path, "r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    logger.info(f"Loaded tro_state for task instance {instance_id}")

    for tro_key, tro_value in tro_state.items():
        if tro_key == "robot_poses":
            presampled_robot_poses = tro_value
            robot_pos = presampled_robot_poses[robot.model_name][0]["position"]
            robot_quat = presampled_robot_poses[robot.model_name][0]["orientation"]
            robot.set_position_orientation(robot_pos, robot_quat)
            # Write robot poses to scene metadata
            env.scene.write_task_metadata(key=tro_key, data=tro_value)
        elif tro_key == "pose_goal":
            # Set pose_goal on robot for RobotHasPosePredicate
            # Expected structure: {"goal_pos": [x,y,z], "goal_quat": [x,y,z,w] or null, "pos_tolerance": float, "ori_tolerance": float or null, link_name: str or null}
            robot.pose_goal = tro_value
            logger.info(f"Loaded pose_goal for robot: {tro_value}")
        else:
            env.task.object_scope[tro_key].load_state(tro_value, serialized=False)

    # Try to ensure that all task-relevant objects are stable
    # They should already be stable from the sampled instance, but there is some issue where loading the state
    # causes some jitter (maybe for small mass / thin objects?)
    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()

    # env.scene.update_initial_file()
    # env.scene.reset()

    # teleoperate robot with keyboard
    controller_choices = {
        "base": "HolonomicBaseJointController",
        "trunk": "JointController",
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
    }
    if not quickstart:
        controller_choices = choose_controllers(robot=robot)
    
    control_mode = "teleop"
    
    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()

    # Automatically compute a good viewer camera pose based on the robot's position and room layout
    if task_name in KNOWN_CAMERA_POSES:
        camera_pos, camera_quat = KNOWN_CAMERA_POSES[task_name]
    else:
        camera_pos, camera_quat = compute_auto_camera_pose(robot, env.scene)
    og.sim.viewer_camera.set_position_orientation(
        position=camera_pos,
        orientation=camera_quat,
    )

    # Reset environment    
    env.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Increase movement speed by scaling all keypress values
    speed_multiplier = 5.0  # Adjust this value to control how much faster (2.0 = 2x faster)
    for key, mapping in action_generator.keypress_mapping.items():
        if mapping["val"] is not None:
            mapping["val"] *= speed_multiplier
    logger.info(f"Keyboard teleop speed multiplier: {speed_multiplier}x")

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()
    
    # Enable camera teleoperation
    if enable_camera_teleop:
        og.sim.enable_viewer_camera_teleoperation()

    # Record waypoints
    waypoints = []
    def add_waypoint():
        nonlocal waypoints
        pos = robot.get_position_orientation()[0]
        logger.info(f"Added waypoint at {pos}")
        waypoints.append(pos)

    def clear_waypoints():
        nonlocal waypoints
        logger.info("Cleared all waypoints!")
        waypoints = []

    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.W,
        callback_fn=add_waypoint,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.X,
        callback_fn=clear_waypoints,
    )

    logger.info("\t W: Save the current robot pose as a waypoint")
    logger.info("\t X: Clear all waypoints")

    # Record video
    unique_id = str(int(time.time()))

    def get_save_dir():
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos")
        assert os.path.exists(base_dir), f"Base directory {base_dir} does not exist"        
        os.makedirs(os.path.join(base_dir, task_name, str(instance_id), unique_id), exist_ok=True)
        return os.path.join(base_dir, task_name, str(instance_id), unique_id)

    video_writer = None
    video_saved = False
    if record_video:
        if video_path is None:
            save_dir = get_save_dir()
            video_path = os.path.join(save_dir, f"video_{task_name}_{instance_id}.mp4")            
            # video_path = f"./video_{task_name}_{instance_id}.mp4"
        os.makedirs(os.path.dirname(video_path) if os.path.dirname(video_path) else ".", exist_ok=True)
        video_writer = create_video_writer(
            fpath=video_path,
            resolution=(448, 1120),  # height, width - includes viewer camera (448x448) on right
        )
        logger.info(f"Recording video to: {video_path}")

    def save_video():
        nonlocal video_writer, video_saved
        if video_writer is None:
            logger.info("No video to save (recording not enabled)")
            return
        if video_saved:
            logger.info("Video already saved!")
            return
        try:
            container, stream = video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            video_saved = True
            video_writer = None
            # save waypoints (convert tensors to lists for JSON serialization)
            waypoints_serializable = [wp.tolist() if hasattr(wp, 'tolist') else wp for wp in waypoints]
            with open(os.path.join(os.path.dirname(video_path), f"waypoints_{task_name}_{instance_id}.json"), "w") as f:
                json.dump(waypoints_serializable, f)

            logger.info(f"Video saved to: {video_path}")
            logger.info(f"Waypoints saved to: {os.path.join(os.path.dirname(video_path), f'waypoints_{task_name}_{instance_id}.json')}")
        
        except Exception as e:
            logger.info(f"Error saving video: {e}")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Z,
        callback_fn=save_video,
    )
    logger.info("\t Z: Save video recording and waypoints")

    # Front-view camera capture functionality
    # Use "head" key from ROBOT_CAMERA_NAMES to avoid hardcoding camera name
    front_view_camera_key = "head"
    front_view_camera_name = ROBOT_CAMERA_NAMES["R1Pro"][front_view_camera_key]
    front_view_save_counter = [0]  # Use list to allow modification in closure

    def save_front_view_camera():
        """Save front-view camera image and pose when Y is pressed."""
        # Get the camera sensor
        camera_sensor_name = front_view_camera_name.split("::")[1]
        camera = robot.sensors[camera_sensor_name]
        
        # Get camera pose using the same logic as preprocess_obs
        # camera.get_position_orientation() returns the most recent camera poses, but since og render is "async",
        # it may not be in sync with the visual observations. camera.camera_parameters["cameraViewTransform"] 
        # is guaranteed to be in sync with the visual observations.
        direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
        if np.allclose(direct_cam_pose, np.zeros(16)):
            # First time query returns all zeros, fallback to direct method
            cam_pos, cam_quat = camera.get_position_orientation()
        else:
            # Extract pose from the synchronized cameraViewTransform matrix
            cam_pos, cam_quat = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
        logger.info(f"Front-view camera ({front_view_camera_name}) pose:")
        logger.info(f"  Position: {cam_pos.tolist()}")
        logger.info(f"  Orientation: {cam_quat.tolist()}")
        
        # Determine save directory (same as video_path directory)
        if video_path is not None:
            save_dir = os.path.dirname(video_path) if os.path.dirname(video_path) else "."
        else:
            save_dir = get_save_dir()
        os.makedirs(save_dir, exist_ok=True)
        
        # Get current observation for the camera image
        current_obs, _ = env.get_obs()
        current_obs = flatten_obs_dict(current_obs)
        
        # Get the front-view camera RGB image
        rgb_key = front_view_camera_name + "::rgb"
        if rgb_key in current_obs:
            front_view_rgb = current_obs[rgb_key][:, :, :3].numpy()
            
            # Save image (convert RGB to BGR for cv2)
            img_filename = f"front_view_camera_{front_view_save_counter[0]}.png"
            img_path = os.path.join(save_dir, img_filename)
            cv2.imwrite(img_path, cv2.cvtColor(front_view_rgb, cv2.COLOR_RGB2BGR))
            
            # Save pose as JSON
            pose_filename = f"front_view_camera_pose_{front_view_save_counter[0]}.json"
            pose_path = os.path.join(save_dir, pose_filename)
            pose_data = {
                "position": cam_pos.tolist(),
                "orientation": cam_quat.tolist(),
                "camera_name": front_view_camera_name,
            }
            with open(pose_path, "w") as f:
                json.dump(pose_data, f, indent=2)
            
            logger.info(f"Saved front-view camera image to: {img_path}")
            logger.info(f"Saved front-view camera pose to: {pose_path}")
            
            front_view_save_counter[0] += 1
        else:
            logger.warning(f"Front-view camera RGB key '{rgb_key}' not found in observations")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.Y,
        callback_fn=save_front_view_camera,
    )
    logger.info("\t Y: Save front-view camera image and pose")

    def preprocess_obs(obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        return obs
    
    def write_video_frame(obs):
        """Write current observation frame to video."""
        if video_writer is None:
            return
        # Get RGB observations from robot cameras
        left_wrist_rgb = cv2.resize(
            obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"][:, :, :3].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"][:, :, :3].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"][:, :, :3].numpy(),
            (448, 448),
        )
        # Get viewer camera RGB
        viewer_obs, _ = og.sim.viewer_camera.get_obs()
        viewer_rgb = cv2.resize(
            viewer_obs["rgb"][:, :, :3].numpy(),  # Remove alpha channel if present
            (448, 448),
        )
        # Stack cameras: [left_wrist, right_wrist] on left, head in middle, viewer on right
        frame = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb, viewer_rgb])
        write_video(
            np.expand_dims(frame, 0),
            video_writer=video_writer,
            batch_size=1,
            mode="rgb",
        )

    # Other helpful user info
    logger.info("Running demo.")
    logger.info("Press ESC to quit")
    if record_video:
        logger.info("Recording video... Press ESC when done to save.")

    # Loop control until user quits
    max_steps = -1
    step = 0

    # Capture initial frame before any actions
    if record_video:
        initial_obs, _ = env.get_obs()
        initial_obs = preprocess_obs(initial_obs)
        write_video_frame(initial_obs)

    try:
        while step != max_steps:
            action = action_generator.get_teleop_action()
            
            obs, _, _, _, _ = env.step(action=action)
            obs = preprocess_obs(obs)
            
            # Write video frame if recording
            if record_video:
                write_video_frame(obs)
            
            step += 1
    except Exception as e:
        logger.info(f"Loop exited: {e}")
    finally:
        # Close video writer if not already saved via 'O' key
        save_video()

    # Always shut down the environment cleanly at the end
    og.shutdown()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--instance_id", type=int, required=True)
    parser.add_argument("--quickstart", action="store_true", help="Load the scene without objects and use default controller settings")
    parser.add_argument("--record_video", action="store_true", help="Enable video recording")
    parser.add_argument("--video_path", type=str, default=None, help="Path to save video (default: ./video_{task}_{instance}.mp4)")
    parser.add_argument("--enable_camera_teleop", action="store_true", help="Enable camera teleoperation")
    
    args = parser.parse_args()
    main(args.task_name, args.instance_id, args.quickstart, args.record_video, args.video_path, args.enable_camera_teleop)
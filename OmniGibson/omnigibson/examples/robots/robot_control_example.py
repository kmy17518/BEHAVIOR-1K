"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

from pathlib import Path

import torch as th
from omegaconf import OmegaConf

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics for performance boost
gm.USE_GPU_DYNAMICS = False


def list_end_effectors(robot_name, dataset_name="omnigibson-robot-assets"):
    """Return end-effector variant names for a robot definition, if any."""
    definition_path = Path(get_dataset_path(dataset_name)) / "models" / robot_name / f"{robot_name}.yaml"
    if not definition_path.is_file():
        matches = sorted(Path(gm.DATA_PATH).glob(f"*/models/{robot_name}/{robot_name}.yaml"))
        if not matches:
            return []
        definition_path = matches[0]
    definition = OmegaConf.load(definition_path)
    manipulation = definition.get("manipulation")
    if not manipulation:
        return []
    end_effectors = manipulation.get("end_effectors")
    return list(end_effectors.keys()) if end_effectors else []


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


def main(
    random_selection=False,
    headless=False,
    short_exec=False,
    quickstart=False,
    scene=None,
    robot=None,
    end_effector=None,
    control_mode=None,
    default_controllers=False,
):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    scene_model = "Rs_int" if quickstart and scene is None else scene
    if scene_model is None:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    # Choose robot to create
    robot_options = list(sorted(set(REGISTERED_ROBOTS)))
    robot_name = "fetch" if quickstart and robot is None else robot
    if robot_name is None:
        robot_name = choose_from_options(options=robot_options, name="robot", random_selection=random_selection)

    end_effector_options = list_end_effectors(robot_name)
    if end_effector_options:
        if end_effector is None and not quickstart:
            end_effector = choose_from_options(
                options=end_effector_options,
                name="end effector",
                random_selection=random_selection,
            )
        elif end_effector is None:
            end_effector = "gripper" if "gripper" in end_effector_options else end_effector_options[0]
        if end_effector not in end_effector_options:
            raise ValueError(
                f"Unknown end effector {end_effector!r} for {robot_name}. Available: {end_effector_options}"
            )

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["model"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True
    if end_effector is not None:
        robot0_cfg["end_effector"] = end_effector

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {
        "base": "DifferentialDriveController",
        "arm_0": "InverseKinematicsController",
        "gripper_0": "MultiFingerGripperController",
        "camera": "JointController",
    }
    if default_controllers:
        controller_choices = dict(robot._definition.default_controllers)
    elif not quickstart:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    if control_mode is None:
        if random_selection:
            control_mode = "random"
        elif quickstart:
            control_mode = "teleop"
        else:
            control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    random_action = None
    while step != max_steps:
        if control_mode == "random":
            # Sample new random action every 30 steps
            if step % 30 == 0:
                random_action = action_generator.get_random_action() * 0.05
            action = random_action
        else:
            action = action_generator.get_teleop_action()

        env.step(action=action)
        step += 1

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")

    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    parser.add_argument("--scene", choices=list(SCENES), help="Scene to load, skipping the interactive prompt.")
    parser.add_argument("--robot", help="Registered robot model to load, skipping the interactive prompt.")
    parser.add_argument(
        "--end-effector",
        help="End-effector variant for robots that support multiple EEFs (e.g. franka: gripper, mounted, sharpa_right).",
    )
    parser.add_argument(
        "--control-mode",
        choices=list(CONTROL_MODES),
        help="Control mode to use, skipping the interactive prompt.",
    )
    parser.add_argument(
        "--default-controllers",
        action="store_true",
        help="Use the robot definition's default controllers instead of prompting.",
    )
    args = parser.parse_args()
    main(
        quickstart=args.quickstart,
        scene=args.scene,
        robot=args.robot,
        end_effector=args.end_effector,
        control_mode=args.control_mode,
        default_controllers=args.default_controllers,
    )

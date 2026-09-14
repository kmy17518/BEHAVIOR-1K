import glob
import os
import re

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes, get_scene_path
from omnigibson.utils.ui_utils import choose_from_options, KeyboardEventHandler
from omnigibson.utils.constants import STRUCTURE_CATEGORIES

# Dataset that holds the original cluttered scene json files (<scene>_with_clutter.json)
CLUTTER_DATASET_NAME = "behavior-1k-assets-outdated"
# Current dataset that holds generated clutter variants (<scene>_with_clutter_<i>.json), as produced by
# omnigibson/experiments/create_cluttered_scene.py
CURRENT_DATASET_NAME = "behavior-1k-assets"

# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def _discover_clutter_indices(json_dir, scene_model):
    """Return sorted integer indices i for which <scene_model>_with_clutter_<i>.json exists in json_dir."""
    regex = re.compile(rf"^{re.escape(scene_model)}_with_clutter_(\d+)\.json$")
    indices = []
    for path in glob.glob(os.path.join(json_dir, f"{scene_model}_with_clutter_*.json")):
        match = regex.match(os.path.basename(path))
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and a clutter variant, then loads it.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose the scene model to load
    scenes = get_available_behavior_1k_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)

    # Let the user choose which cluttered json to load. Generated variants (created by
    # create_cluttered_scene.py) live in the current dataset as <scene_model>_with_clutter_<i>.json;
    # "Original" loads the unmodified <scene_model>_with_clutter.json from the outdated dataset.
    current_json_dir = os.path.join(get_scene_path(scene_model, dataset_name=CURRENT_DATASET_NAME), "json")
    clutter_indices = _discover_clutter_indices(current_json_dir, scene_model)
    clutter_options = {"Original": f"Original cluttered scene ({scene_model}_with_clutter.json)"}
    for i in clutter_indices:
        clutter_options[str(i)] = f"Generated variant {scene_model}_with_clutter_{i}.json"
    clutter_choice = choose_from_options(
        options=clutter_options, name="clutter variant", random_selection=random_selection
    )

    # Resolve the scene file: prefer the selected generated variant, falling back to the original
    # <scene_model>_with_clutter.json if no index was chosen or the generated file does not exist.
    scene_file, dataset_name = None, CLUTTER_DATASET_NAME
    if clutter_choice != "Original":
        candidate = os.path.join(current_json_dir, f"{scene_model}_with_clutter_{clutter_choice}.json")
        if os.path.exists(candidate):
            scene_file, dataset_name = candidate, CURRENT_DATASET_NAME
        else:
            print(f"Generated clutter file not found, falling back to original: {candidate}")
    if scene_file is None:
        scene_dir = get_scene_path(scene_model, dataset_name=CLUTTER_DATASET_NAME)
        scene_file = os.path.join(scene_dir, "json", f"{scene_model}_with_clutter.json")
    assert os.path.exists(scene_file), f"Cluttered scene file does not exist: {scene_file}"
    print(f"Loading cluttered scene file: {scene_file}")

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "dataset_name": dataset_name,
            # Overrides scene_instance/scene_model to load the cluttered json directly
            "scene_file": scene_file,
        },
    }

    # Check if we want to quick load or full load the scene
    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
        "Full": "Load all interactive objects in the scene",
    }
    load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
    if load_mode == "Quick":
        cfg["scene"]["load_object_categories"] = list(STRUCTURE_CATEGORIES)

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: og.shutdown(),
    )

    print("Running demo.")
    print("Press ESC to quit")

    # Loop indefinitely
    steps = 0
    max_steps = -1 if not short_exec else 100
    while steps != max_steps:
        env.step([])
        steps += 1

    og.shutdown()


if __name__ == "__main__":
    main()

"""Replay a dex_teleop ARAT HDF5 recording and render its four camera views."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.arat.camera_rig import apply_camera_lens_models, build_camera_sensor_configs, load_camera_rig
from dex_teleop.arat.scene import validate_runtime_assets
from dex_teleop.omnigibson.evaluation_trace import read_evaluation_episode
from dex_teleop.omnigibson.launcher import (
    ROBOT_PRIM_PATH,
    _hide_skybox_from_camera,
    _show_robot_end_effectors,
    _validate_loaded_apparatus,
)


LEFT_CAMERA_KEY = "external::arat_left_shoulder_camera::rgb"
RIGHT_CAMERA_KEY = "external::arat_right_shoulder_camera::rgb"
THUMB_WRIST_CAMERA_KEY = "external::arat_wrist_camera_thumb::rgb"
PINKY_WRIST_CAMERA_KEY = "external::arat_wrist_camera_pinky::rgb"
EVALUATION_TEXT_SCALE = 1.5
EVALUATION_PANEL_HEIGHT = 600
REPLAY_CAMERA_IDS = ("left_shoulder", "right_shoulder", "thumb_wrist", "pinky_wrist")
DEFAULT_CAMERA_RIG_NAME = next(iter(AratTaskCatalog().tasks.values())).camera_rig
_DEFAULT_CAMERA_CALIBRATION = load_camera_rig(DEFAULT_CAMERA_RIG_NAME).calibration("left_shoulder")
DEFAULT_CAMERA_IMAGE_WIDTH = _DEFAULT_CAMERA_CALIBRATION["image_width"]
DEFAULT_CAMERA_IMAGE_HEIGHT = _DEFAULT_CAMERA_CALIBRATION["image_height"]


@dataclass(frozen=True)
class RecordingInfo:
    task_name: str
    episodes: tuple[tuple[int, str, int], ...]


def get_episode_lengths(data_group) -> tuple[tuple[int, str, int], ...]:
    """Return numerically sorted ``(episode_id, group_name, step_count)`` entries."""

    episodes = []
    for key in data_group:
        if not key.startswith("demo_"):
            continue
        try:
            episode_id = int(key.removeprefix("demo_"))
        except ValueError:
            continue
        episodes.append((episode_id, key, int(data_group[key].attrs["num_samples"])))
    return tuple(sorted(episodes))


def _decode_json_attribute(value) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def inspect_recording(input_path: str | Path) -> RecordingInfo:
    """Read task and episode metadata without launching OmniGibson."""

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Recording does not exist: {input_path}")

    with h5py.File(input_path, "r") as recording:
        if "data" not in recording:
            raise ValueError(f"Recording has no 'data' group: {input_path}")
        data = recording["data"]
        episodes = get_episode_lengths(data)
        if not episodes:
            raise ValueError(f"Recording contains no demo episodes: {input_path}")

        try:
            config = _decode_json_attribute(data.attrs["config"])
            task_name = config["task"]["activity_name"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Recording does not identify its BehaviorTask activity: {input_path}") from error

    return RecordingInfo(task_name=str(task_name), episodes=episodes)


def select_episode(info: RecordingInfo, episode_id: int | None = None) -> tuple[int, int]:
    """Select the requested episode, defaulting to the longest trajectory."""

    lengths = {saved_id: length for saved_id, _, length in info.episodes}
    if episode_id is None:
        episode_id = max(info.episodes, key=lambda episode: episode[2])[0]
    if episode_id not in lengths:
        raise ValueError(f"Invalid episode ID {episode_id}; available IDs: {sorted(lengths)}")
    return episode_id, lengths[episode_id]


def default_output_path(input_path: str | Path, episode_id: int) -> Path:
    input_path = Path(input_path).expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_demo_{episode_id}.mp4")


def build_replay_camera_configs(camera_rig_name: str = DEFAULT_CAMERA_RIG_NAME) -> list[dict]:
    """Enable RGB observations on the four cameras used during ARAT teleoperation."""

    camera_rig = load_camera_rig(camera_rig_name)
    cameras = build_camera_sensor_configs(
        camera_rig,
        "teleop",
        robot_prim_path=ROBOT_PRIM_PATH,
        camera_ids=REPLAY_CAMERA_IDS,
        viewport_name=None,
    )
    for camera in cameras:
        camera["modalities"] = ["rgb"]
        camera["include_in_obs"] = True
    return cameras


def print_episode_lengths(info: RecordingInfo, selected_episode_id: int | None = None) -> None:
    print(f"Task: {info.task_name}")
    print("episode | group    | trajectory length")
    for episode_id, key, length in info.episodes:
        selected = "  < selected" if episode_id == selected_episode_id else ""
        print(f"{episode_id:>7} | {key:<8} | {length:>17}{selected}")


def prompt_for_episode(info: RecordingInfo, input_fn=None) -> int:
    """Show the available episodes and prompt until the user selects one."""

    input_fn = input if input_fn is None else input_fn
    available = {episode_id for episode_id, _, _ in info.episodes}
    print_episode_lengths(info)
    while True:
        try:
            raw_value = input_fn("Select episode ID to replay: ").strip()
        except (EOFError, KeyboardInterrupt) as error:
            raise SystemExit("No episode selected; use --episode-id for a non-interactive replay") from error
        try:
            episode_id = int(raw_value)
        except ValueError:
            print(f"Enter one of these episode IDs: {sorted(available)}")
            continue
        if episode_id in available:
            return episode_id
        print(f"Episode {episode_id} is unavailable; choose one of: {sorted(available)}")


def _load_font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _pixel_wrapped_lines(draw, text: str, font, max_width: int) -> list[str]:
    """Wrap text using rendered width instead of a fixed character count."""

    words = str(text).split()
    if not words:
        return [""]
    lines = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def _draw_lines(draw, lines, *, x, y, width, font, color, line_height, bottom) -> int:
    for line in lines:
        for wrapped in _pixel_wrapped_lines(draw, line, font, width):
            if y + line_height > bottom:
                return y
            draw.text((x, y), wrapped, font=font, fill=color)
            y += line_height
    return y


def _yes_no(value) -> str:
    if value is None:
        return "N/A"
    return "yes" if value else "no"


def _evaluation_evidence_lines(evidence: dict) -> list[str]:
    supports = evidence.get("environment_supports", [])
    support_text = ", ".join(f"{item['object']}:{item['relation']}" for item in supports) or "none"
    apertures = evidence.get("apertures_m", {})
    aperture_text = ", ".join(f"{finger}={value:.3f}m" for finger, value in apertures.items()) or "none"
    hold = evidence.get("scoring_hold")
    if hold is None:
        hold_text = "no confirmed scoring hold"
    else:
        hold_text = (
            f"{hold['start_t']:.2f}-{hold['end_t']:.2f}s, {hold['steps']} steps, "
            f"appropriate={hold['appropriate_fraction']:.1%}, reasons={hold['dominant_reasons'] or ['none']}"
        )
    speed = evidence.get("tracked_speed_m_s")
    speed_text = "N/A" if speed is None else f"{speed:.3f} m/s"
    water = evidence.get("water")
    water_text = (
        "N/A"
        if water is None
        else (
            f"total={water['total']}, source={water['in_source']}, destination={water['in_destination']}, "
            f"outside={water['outside']}"
        )
    )
    return [
        (
            f"Contacts: pads={evidence.get('contact_pads', [])}; dorsals={evidence.get('contact_dorsals', [])}; "
            f"palm={_yes_no(evidence.get('palm_contact'))}"
        ),
        (
            f"Target: at_target={_yes_no(evidence.get('at_target'))}; height={_yes_no(evidence.get('reached_target_height'))}; "
            f"palmar_region={_yes_no(evidence.get('palmar_region_contact'))}; "
            f"dorsal_region={_yes_no(evidence.get('dorsal_region_contact'))}; "
            f"progress={evidence.get('approach_progress_m', 0.0):.3f} m"
        ),
        f"Finger apertures: {aperture_text}",
        f"Object: speed={speed_text}; environmental supports={support_text}",
        f"Hold: active={_yes_no(evidence.get('active_hold'))}; {hold_text}",
        (
            f"Release: in_progress={_yes_no(evidence.get('release_in_progress'))}; "
            f"fumbled={_yes_no(evidence.get('release_fumbled'))}"
        ),
        (
            f"Movement flags: dorsum_push={_yes_no(evidence.get('dorsum_push_detected'))}; "
            f"bracing={_yes_no(evidence.get('braced_grasp_suspected'))}"
        ),
        f"Water: {water_text}",
    ]


def _condition_is_applicable(condition: dict, arat: dict) -> bool:
    """Handle current traces and infer task-level N/A status for older traces."""

    if "applicable" in condition:
        return bool(condition["applicable"])
    if condition.get("key") == "correct_pinch_opposition" and condition.get("detail") == "not a pinch item":
        return False
    return True


SCORE_CONDITION_LEVELS = {
    "movement_started": 1,
    "voluntary_opening": 1,
    "held_and_lifted": 1,
    "qualifying_score1_hold": 1,
    "correct_pinch_opposition": 1,
    "target_height": 1,
    "target_contact_confirmed": 2,
    "completion_or_release_flaw": 2,
    "palmar_contact": 3,
    "appropriate_hand_movement": 3,
    "clean_release": 3,
    "completed_under_5s": 3,
    "no_spill": 3,
}

SCORE_SECTION_TITLES = {
    1: "SCORE 1",
    2: "SCORE 2",
    3: "SCORE 3",
}

SCORE_CONDITION_LABELS = {
    "movement_started": "Partial target progress",
    "voluntary_opening": "Opening",
    "held_and_lifted": "Hold/lift",
    "qualifying_score1_hold": "Qualifying score-1 movement",
    "correct_pinch_opposition": "Named-finger opposition",
    "target_height": "Target height",
    "target_contact_confirmed": "Target contact",
    "completion_or_release_flaw": "Task completion",
    "palmar_contact": "Palmar contact",
    "appropriate_hand_movement": "Grasp quality",
    "clean_release": "Clean release",
    "completed_under_5s": "Under 5 seconds",
    "no_spill": "No spill",
}

EVENT_LABELS = {
    "first_hold_and_lift": "lift",
    "reached_target_height": "height",
    "release_completed": "release",
    "item_finished": "finish",
    "target_region_contact": "target contact",
    "release_fumble": "fumble",
    "drop": "drop",
}


def score_condition_sections(conditions: list[dict], arat: dict) -> tuple[tuple[int, tuple[dict, ...]], ...]:
    """Return stable, task-specific score rows grouped by ARAT score level."""

    sections = {1: [], 2: [], 3: []}
    for condition in conditions:
        if not _condition_is_applicable(condition, arat):
            continue
        sections[SCORE_CONDITION_LEVELS.get(condition.get("key"), 3)].append(condition)
    return tuple((level, tuple(sections[level])) for level in (1, 2, 3) if sections[level])


def render_evaluation_panel(
    trace: dict | None,
    *,
    task_name: str,
    step_index: int,
    total_steps: int,
    width: int = DEFAULT_CAMERA_IMAGE_WIDTH * 3,
    height: int = EVALUATION_PANEL_HEIGHT,
) -> np.ndarray:
    """Render a readable BDDL/ARAT decision dashboard for one video frame."""

    image = Image.new("RGB", (width, height), color=(12, 15, 20))
    draw = ImageDraw.Draw(image)
    def font_px(value: int) -> int:
        return round(value * EVALUATION_TEXT_SCALE)

    header_font = _load_font(font_px(24), bold=True)
    title_font = _load_font(font_px(19), bold=True)
    body_font = _load_font(font_px(15))
    body_bold = _load_font(font_px(15), bold=True)
    muted = (175, 184, 198)
    white = (235, 239, 245)
    green = (93, 222, 138)
    red = (255, 112, 112)
    amber = (255, 205, 92)
    # Keep the 600-pixel panel while reserving visible breathing room around
    # score-section headers. The 26-pixel body font remains comfortably legible
    # with 30-pixel leading.
    line_height = 30
    margin = 18
    split = width // 2

    if trace is None:
        draw.text((margin, 5), f"{task_name} | initial state", font=header_font, fill=white)
        draw.text(
            (margin, 60),
            "Evaluation begins after the first recorded teleoperation action.",
            font=title_font,
            fill=muted,
        )
        return np.asarray(image)

    arat = trace["arat"]
    task_time = arat.get("task_time_s") if arat.get("enabled") else None
    time_text = "ARAT disabled" if task_time is None else f"task time {task_time:.3f} s"
    draw.text(
        (margin, 5),
        f"{task_name} | action {step_index + 1}/{total_steps} | {time_text}",
        font=header_font,
        fill=white,
    )
    draw.line((split, 54, split, height - 12), fill=(58, 66, 80), width=2)

    bddl = trace["bddl"]
    left_x = margin
    left_width = split - margin * 2
    y = 60
    draw.text(
        (left_x, y),
        f"BDDL goal conditions — {bddl['satisfied_count']}/{bddl['total_count']} satisfied",
        font=title_font,
        fill=white,
    )
    y += 38
    for condition in bddl["conditions"]:
        passed = condition["satisfied"]
        status = "PASS" if passed else "FAIL"
        lines = _pixel_wrapped_lines(
            draw,
            f"[{status}] {condition['index']}: {condition['text']}",
            body_font,
            left_width,
        )
        y = _draw_lines(
            draw,
            lines,
            x=left_x,
            y=y,
            width=left_width,
            font=body_font,
            color=green if passed else red,
            line_height=line_height,
            bottom=height - 18,
        )
        y += 2

    if arat.get("enabled") and y < height - 60:
        y += 5
        draw.text((left_x, y), "Current ARAT evidence", font=title_font, fill=white)
        y += 37
        evidence_lines = _evaluation_evidence_lines(arat["evidence"])
        _draw_lines(
            draw,
            evidence_lines,
            x=left_x,
            y=y,
            width=left_width,
            font=body_font,
            color=muted,
            line_height=line_height,
            bottom=height - 18,
        )

    right_x = split + margin
    right_width = width - right_x - margin
    y = 60
    header = "ARAT score calculation - "
    draw.text((right_x, y), header, font=title_font, fill=white)
    score_x = right_x + draw.textlength(header, font=title_font)
    score_text = "N/A" if not arat.get("enabled") else f"{arat['provisional_score']}/3"
    draw.text((score_x, y), score_text, font=title_font, fill=amber if arat.get("enabled") else muted)
    y += 38
    if not arat.get("enabled"):
        _draw_lines(
            draw,
            ["ARAT evaluation was disabled while this trajectory was recorded."],
            x=right_x,
            y=y,
            width=right_width,
            font=body_font,
            color=muted,
            line_height=line_height,
            bottom=height - 18,
        )
        return np.asarray(image)

    for level, conditions in score_condition_sections(arat["conditions"], arat):
        if y >= height - 38:
            break
        draw.line(
            (right_x, y + 5, right_x + right_width, y + 5),
            fill=(58, 66, 80),
            width=1,
        )
        y += 12
        y = _draw_lines(
            draw,
            [SCORE_SECTION_TITLES[level]],
            x=right_x,
            y=y,
            width=right_width,
            font=body_bold,
            color=white,
            line_height=line_height,
            bottom=height - 38,
        )
        y += 5
        for condition in conditions:
            met = condition["met"]
            status = "PASS" if met is True else "FAIL" if met is False else "WAIT"
            color = green if met is True else red if met is False else muted
            label = SCORE_CONDITION_LABELS.get(condition.get("key"), condition["label"])
            if condition.get("key") == "movement_started":
                # Older recordings called this a "gate", which incorrectly implied
                # that partial score-1 performance was a prerequisite for completion.
                label = "Partial target progress (if not completed)"
            y = _draw_lines(
                draw,
                [f"[{status}] {label}: {condition['detail']}"],
                x=right_x,
                y=y,
                width=right_width,
                font=body_font,
                color=color,
                line_height=line_height,
                bottom=height - 38,
            )
            if y >= height - 38:
                break

    events = arat.get("events", [])
    if y < height - 20:
        event_text = "none yet" if not events else "; ".join(
            f"{EVENT_LABELS.get(event['name'], event['name'])} {event['t']:.2f}s" for event in events[-4:]
        )
        _draw_lines(
            draw,
            [f"Events: {event_text}"],
            x=right_x,
            y=y + 2,
            width=right_width,
            font=body_font,
            color=white,
            line_height=line_height,
            bottom=height - 5,
        )
    return np.asarray(image)


def _create_video_playback_wrapper_class(
    evaluation_steps: list[dict] | None = None,
    task_name: str = "",
    camera_width: int = DEFAULT_CAMERA_IMAGE_WIDTH,
    camera_height: int = DEFAULT_CAMERA_IMAGE_HEIGHT,
):
    from omnigibson.envs import DataPlaybackWrapper
    from omnigibson.eval.utils.obs_utils import create_video_writer, write_video

    class AratVideoPlaybackWrapper(DataPlaybackWrapper):
        def _create_video_writers(self, video_keys):
            output_path = Path(self.video_output_dir) / f"{video_keys['aggregated']}.mp4"
            output_height = camera_height + (EVALUATION_PANEL_HEIGHT if evaluation_steps is not None else 0)
            container, stream = create_video_writer(
                fpath=str(output_path),
                resolution=(output_height, camera_width * 4),
                rate=self.fps,
                stream_options={"crf": "30"},
            )
            self.video_writers.append((container, stream, "aggregated"))
            self._evaluation_frame_index = -1

        def _write_video_frames(self):
            container, stream, _ = self.video_writers[0]
            frames = [
                self._extract_frame_from_obs(self.current_obs, LEFT_CAMERA_KEY),
                self._extract_frame_from_obs(self.current_obs, RIGHT_CAMERA_KEY),
                self._extract_frame_from_obs(self.current_obs, THUMB_WRIST_CAMERA_KEY),
                self._extract_frame_from_obs(self.current_obs, PINKY_WRIST_CAMERA_KEY),
            ]
            frame = np.concatenate(frames, axis=1)
            if evaluation_steps is not None:
                trace = (
                    None
                    if self._evaluation_frame_index < 0
                    else evaluation_steps[self._evaluation_frame_index]
                )
                panel = render_evaluation_panel(
                    trace,
                    task_name=task_name,
                    step_index=self._evaluation_frame_index,
                    total_steps=len(evaluation_steps),
                    width=frame.shape[1],
                )
                frame = np.concatenate((frame, panel), axis=0)
                self._evaluation_frame_index += 1
            frame = frame[np.newaxis, ...]
            write_video(frame, (container, stream), mode="rgb")

        def create_dataset(self, output_path, env, overwrite=True):
            pass

        def close_dataset(self):
            pass

    return AratVideoPlaybackWrapper


def _restore_replay_visual_state(env) -> None:
    _show_robot_end_effectors(env.robots[0])


def replay_hdf5_to_video(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    episode_id: int | None = None,
    task_name: str | None = None,
    n_render_iterations: int = 1,
    evaluation_overlay: bool = False,
) -> Path:
    """Replay one recorded episode from serialized state and render it to MP4."""

    input_path = Path(input_path).expanduser().resolve()
    info = inspect_recording(input_path)
    if task_name is not None and task_name != info.task_name:
        raise ValueError(f"Requested task {task_name!r} does not match recording task {info.task_name!r}")
    task_name = info.task_name
    episode_id, n_steps = select_episode(info, episode_id)
    evaluation_steps = (
        read_evaluation_episode(input_path, episode_id, expected_steps=n_steps) if evaluation_overlay else None
    )

    output_path = (
        default_output_path(input_path, episode_id)
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"Replay output must use the .mp4 extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = AratTaskCatalog()
    try:
        task = catalog.tasks[task_name]
    except KeyError as error:
        raise ValueError(f"Recording contains an unknown ARAT task: {task_name}") from error
    validate_runtime_assets((task,))

    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    gm.USE_PBR_MATERIALS = True
    gm.RENDER_VIEWER_CAMERA = False

    replay_camera_configs = build_replay_camera_configs(task.camera_rig)
    replay_resolutions = {
        (camera["sensor_kwargs"]["image_width"], camera["sensor_kwargs"]["image_height"])
        for camera in replay_camera_configs
    }
    if len(replay_resolutions) != 1:
        raise ValueError("Replay requires both shoulder and both wrist cameras to share a resolution")
    camera_width, camera_height = replay_resolutions.pop()
    wrapper_class = _create_video_playback_wrapper_class(
        evaluation_steps=evaluation_steps,
        task_name=task_name,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    env = wrapper_class.create_from_hdf5(
        input_path=str(input_path),
        output_path=str(output_path),
        n_render_iterations=n_render_iterations,
        robot_obs_modalities=[],
        external_sensors_config=replay_camera_configs,
        include_task=True,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,
    )
    try:
        apply_camera_lens_models(load_camera_rig(task.camera_rig), env.external_sensors, REPLAY_CAMERA_IDS)
        _validate_loaded_apparatus(env, task)
        _hide_skybox_from_camera()
        _restore_replay_visual_state(env)
        print_episode_lengths(info, selected_episode_id=episode_id)
        print(f"Replaying episode {episode_id} ({n_steps} steps) to {output_path}")
        env.playback_episode(
            episode_id=episode_id,
            record_data=False,
            video_keys={"aggregated": output_path.stem},
            post_state_update_callback=lambda: _restore_replay_visual_state(env),
        )
    finally:
        env.input_hdf5.close()

    print(f"Replay complete: {output_path}")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="HDF5 recording produced by dex_teleop launch_og.py")
    parser.add_argument("--output", help="Output MP4 path; defaults beside the recording")
    parser.add_argument("--episode-id", type=int, help="Episode to replay; omit to select interactively")
    parser.add_argument("--task", help="Optional expected task name used to validate the recording")
    parser.add_argument("--n-render-iterations", type=int, default=1)
    parser.add_argument(
        "--evaluation-overlay",
        "--log-eval",
        dest="evaluation_overlay",
        action="store_true",
        help="Add detailed per-step BDDL and ARAT scoring evidence to the demo video",
    )
    parser.add_argument("--list-episodes", action="store_true", help="List recorded episodes without launching OmniGibson")
    parser.add_argument("--headless", action="store_true", help="Launch OmniGibson without a display")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    info = inspect_recording(args.input)
    if args.list_episodes:
        print_episode_lengths(info)
        return
    if args.episode_id is None:
        args.episode_id = prompt_for_episode(info)
    if args.n_render_iterations <= 0:
        raise SystemExit("--n-render-iterations must be positive")
    if args.headless:
        os.environ["OMNIGIBSON_HEADLESS"] = "1"

    try:
        replay_hdf5_to_video(
            args.input,
            output_path=args.output,
            episode_id=args.episode_id,
            task_name=args.task,
            n_render_iterations=args.n_render_iterations,
            evaluation_overlay=args.evaluation_overlay,
        )
    finally:
        if "omnigibson" in sys.modules:
            import omnigibson as og

            og.shutdown()


if __name__ == "__main__":
    main()

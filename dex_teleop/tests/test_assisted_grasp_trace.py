"""Tests for simulator-independent assisted-grasp HDF5 diagnostics."""

import h5py
import pytest

from dex_teleop.omnigibson.assisted_grasp_trace import (
    AssistedGraspTraceCollector,
    _contact_aggregate,
    _contact_force,
    read_assisted_grasp_config,
    read_assisted_grasp_episode,
    summarize_assisted_grasp_episode,
    write_assisted_grasp_episodes,
)


def _recording(path, samples=3):
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.create_group("demo_0").attrs["num_samples"] = samples


def test_assisted_grasp_trace_round_trips_and_classifies_physx_break(tmp_path):
    path = tmp_path / "recording.hdf5"
    _recording(path)
    steps = [
        {
            "episode_time_s": 0.0,
            "supervisor_event": {"kind": "welded", "object": "ball"},
            "joint_break_events": [],
            "held_contacts": {"aggregate": {"max_force_norm_n": None, "min_separation_m": None}},
        },
        {
            "episode_time_s": 1.0 / 30.0,
            "supervisor_event": None,
            "joint_break_events": [{"joint_path": "/robot/hand/ag_constraint", "simulation_time_s": 1.2}],
            "held_contacts": {"aggregate": {"max_force_norm_n": 42.0, "min_separation_m": -0.003}},
            "external_contacts": [{"held_link": "/ball/base", "other": "/tin/base"}],
        },
        {
            "episode_time_s": 2.0 / 30.0,
            "supervisor_event": {"kind": "broke", "object": "ball", "drift_m": 0.08},
            "joint_break_events": [],
            "held_contacts": {"aggregate": {"max_force_norm_n": 15.0, "min_separation_m": -0.001}},
        },
    ]

    write_assisted_grasp_episodes(path, [steps], {"weld_break_force": 100.0})

    assert read_assisted_grasp_config(path) == {"weld_break_force": 100.0}
    assert read_assisted_grasp_episode(path, 0, expected_steps=3) == steps
    failure = summarize_assisted_grasp_episode(steps)["breaks"][0]
    assert failure["classification"] == "physx_break_limit"
    assert failure["max_finger_contact_force_n"] == 42.0
    assert failure["min_contact_separation_m"] == -0.003
    assert failure["external_contacts"] == ["/tin/base"]


def test_assisted_grasp_trace_distinguishes_drift_with_external_contact():
    steps = [
        {
            "supervisor_event": {"kind": "welded", "object": "ball"},
            "joint_break_events": [],
            "external_contacts": [],
        },
        {
            "supervisor_event": {"kind": "broke", "object": "ball", "drift_m": 0.06},
            "joint_break_events": [],
            "external_contacts": [{"held_link": "/ball/base", "other": "/box/base"}],
        },
    ]

    failure = summarize_assisted_grasp_episode(steps)["breaks"][0]

    assert failure["classification"] == "drift_with_external_contact"
    assert failure["external_contacts"] == ["/box/base"]


def test_assisted_grasp_trace_rejects_step_misalignment(tmp_path):
    path = tmp_path / "recording.hdf5"
    _recording(path, samples=1)

    with pytest.raises(RuntimeError, match="has 0 steps; trajectory has 1"):
        write_assisted_grasp_episodes(path, [[]], {})


def test_scalar_isaac_contact_force_is_converted_along_normal():
    reported_force, force_vector = _contact_force((0.0, 0.0, -1.0), [12.5])

    assert reported_force == [12.5]
    assert force_vector == [0.0, 0.0, -12.5]
    aggregate = _contact_aggregate(
        [{"force_n": force_vector, "force_norm_n": 12.5, "separation_m": -0.002}]
    )
    assert aggregate["net_force_n"] == [0.0, 0.0, -12.5]
    assert aggregate["max_force_norm_n"] == 12.5


def test_collection_error_becomes_aligned_diagnostic_row():
    class BrokenSupervisor:
        arm = "0"
        control_dt = 1.0 / 30.0
        last_event = {"kind": "welded", "object": "ball"}

        @staticmethod
        def diagnostic_state():
            raise RuntimeError("unsupported simulator value")

    collector = object.__new__(AssistedGraspTraceCollector)
    collector._supervisor = BrokenSupervisor()
    collector._robot = object()
    collector._joint_break_events = [{"joint_path": "/robot/hand/ag_constraint"}]

    step = collector.capture_before_supervisor(
        episode_step=4,
        sim_time_before_s=1.0,
        sim_time_after_s=1.1,
        live_fingers={"finger": 0.5},
        measured_fingers={"finger": 0.4},
        frozen_fingers=None,
    )
    step = collector.finish_after_supervisor(step)

    assert step["episode_step"] == 4
    assert step["joint_break_events"] == [{"joint_path": "/robot/hand/ag_constraint"}]
    assert step["supervisor_event"] == {"kind": "welded", "object": "ball"}
    assert [error["phase"] for error in step["diagnostic_errors"]] == [
        "capture_before_supervisor",
        "finish_after_supervisor",
    ]

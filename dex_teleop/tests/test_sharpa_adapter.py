"""Unit tests for SharpaActionAdapter's finger freeze and wrist gate (no simulator required)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from dex_teleop.hands import SHARPA_ACTION_JOINTS
from dex_teleop.omnigibson.sharpa_adapter import SharpaActionAdapter, SharpaAdapterConfig
from dex_teleop.runtime import SafetyConfig


class FakeLink:
    def get_position_orientation(self):
        return th.tensor([0.3, 0.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0])


class FakeRobot:
    def __init__(self):
        self.arm_names = ["0"]
        self.action_dim = 28
        self.dof_names_ordered = [f"panda_joint{i}" for i in range(1, 8)] + list(SHARPA_ACTION_JOINTS)
        self.controller_joint_idx = {"gripper_0": list(range(7, 29))}
        self.eef_links = {"0": FakeLink()}

    def get_position_orientation(self):
        return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

    def get_joint_positions(self):
        return th.arange(29, dtype=th.float32) / 100.0


def snapshot(flexion=0.6, wrist_quat=None, timestamp=0.0):
    wrist_quat = np.array([0.0, 0.0, 0.0, 1.0]) if wrist_quat is None else np.asarray(wrist_quat, dtype=np.float64)
    return SimpleNamespace(
        command=SimpleNamespace(
            hand_model="sharpa",
            handedness=SimpleNamespace(value="right"),
            joint_names=SHARPA_ACTION_JOINTS,
            joint_positions=np.full(len(SHARPA_ACTION_JOINTS), flexion),
        ),
        frame=SimpleNamespace(timestamp=timestamp, wrist_position=np.zeros(3), wrist_quaternion_xyzw=wrist_quat),
    )


def make_adapter(config=None):
    return SharpaActionAdapter(
        FakeRobot(),
        config=config or SharpaAdapterConfig(control_hz=30.0),
        safety_config=SafetyConfig(enabled=False),
    )


def rotz(degrees):
    angle = math.radians(degrees)
    return np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


class Clock:
    """Feed the adapter frames with advancing timestamps and read the wrist-Z command."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.t = 0.0

    def act(self, degrees, dt=1.0 / 30.0):
        self.t += dt
        action = self.adapter.action(snapshot(wrist_quat=rotz(degrees), timestamp=self.t))
        return float(action[5])  # axis-angle Z component of the wrist command


def finger_slice(action):
    return {name: float(action[6 + i]) for i, name in enumerate(SHARPA_ACTION_JOINTS)}


def test_action_passes_live_fingers_through_without_freeze():
    adapter = make_adapter()
    fingers = finger_slice(adapter.action(snapshot()))
    assert all(value == pytest.approx(0.6) for value in fingers.values())
    assert adapter.last_live_fingers == {name: pytest.approx(0.6) for name in SHARPA_ACTION_JOINTS}


def test_action_overrides_only_frozen_joints_and_keeps_live_readable():
    adapter = make_adapter()
    frozen = {name: 0.85 for name in SHARPA_ACTION_JOINTS if name.startswith("right_thumb_")}
    fingers = finger_slice(adapter.action(snapshot(), frozen_fingers=frozen))
    for name, value in fingers.items():
        expected = 0.85 if name.startswith("right_thumb_") else 0.6
        assert value == pytest.approx(expected)
    # The unfrozen operator command stays observable for the release decision.
    assert adapter.last_live_fingers == {name: pytest.approx(0.6) for name in SHARPA_ACTION_JOINTS}


def test_action_rejects_unknown_frozen_joint_names():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="Unknown frozen finger joints"):
        adapter.action(snapshot(), frozen_fingers={"right_thumb_bogus": 0.5})


def test_measured_fingers_reads_current_joint_positions_by_name():
    adapter = make_adapter()
    measured = adapter.measured_fingers
    assert set(measured) == set(SHARPA_ACTION_JOINTS)
    for i, name in enumerate(SHARPA_ACTION_JOINTS):
        assert measured[name] == pytest.approx((7 + i) / 100.0)


# ------------------------------------------------------------------ wrist-orientation gate


def test_wrist_gate_tracks_human_steps_and_holds_a_flip():
    clock = Clock(make_adapter())
    clock.act(0)  # anchor
    assert clock.act(20) == pytest.approx(math.radians(20), abs=1e-3)
    # A 180-degree tracker flip is impossibly fast: the command holds the last orientation.
    assert clock.act(200) == pytest.approx(math.radians(20), abs=1e-3)
    assert clock.act(200) == pytest.approx(math.radians(20), abs=1e-3)
    # The estimate returns near the held orientation: tracking resumes immediately.
    assert clock.act(25) == pytest.approx(math.radians(25), abs=1e-3)


def test_wrist_gate_resyncs_when_the_new_orientation_persists():
    config = SharpaAdapterConfig(control_hz=30.0, orientation_resync_s=0.15)
    clock = Clock(make_adapter(config))
    clock.act(0)  # anchor
    clock.act(10)
    # Rejection starts on the first flipped frame; five 33 ms frames later (~0.167 s of
    # persistence) the new orientation is accepted.
    held = [clock.act(190) for _ in range(6)]
    assert held[:-1] == [pytest.approx(math.radians(10), abs=1e-3)] * 5
    # Re-synchronized; the axis-angle hemisphere is ambiguous at exactly 180 degrees, so
    # compare the commanded rotation modulo a full turn.
    assert held[-1] % math.tau == pytest.approx(math.radians(190), abs=1e-3)
    assert clock.act(195) % math.tau == pytest.approx(math.radians(195), abs=1e-3)


def test_wrist_gate_scales_allowance_for_stale_frames():
    clock = Clock(make_adapter())
    clock.act(0)  # anchor
    # 55 degrees in one 33 ms frame would be rejected, but over a 100 ms frame gap it is
    # within the 600 deg/s human-burst allowance.
    assert clock.act(55, dt=0.1) == pytest.approx(math.radians(55), abs=1e-3)


def test_wrist_gate_caps_the_stale_frame_allowance():
    clock = Clock(make_adapter())
    clock.act(0)  # anchor
    # Even a very stale frame may not jump past the 60-degree hard cap: a flip cannot
    # slip through as a couple of stale jumps.
    assert clock.act(80, dt=0.15) == pytest.approx(0.0, abs=1e-3)


def test_wrist_gate_limits_sustained_slew():
    clock = Clock(make_adapter())
    clock.act(0)  # anchor
    # Each step passes the 45-degree frame gate, but the cumulative rotation exceeds the
    # slew budget against the horizon baseline: a flip reached in sub-threshold steps.
    assert clock.act(40) == pytest.approx(math.radians(40), abs=1e-3)
    assert clock.act(80) == pytest.approx(math.radians(40), abs=1e-3)  # held
    assert clock.act(80) == pytest.approx(math.radians(40), abs=1e-3)  # still held
    # The estimate returns within the budget: tracking resumes without a re-sync.
    assert clock.act(50) == pytest.approx(math.radians(50), abs=1e-3)


def test_wrist_gate_treats_hemisphere_sign_flips_as_the_same_rotation():
    adapter = make_adapter()
    clock = Clock(adapter)
    clock.act(0)  # anchor
    reference = clock.act(20)
    clock.t += 1.0 / 30.0
    action = adapter.action(snapshot(wrist_quat=-rotz(20), timestamp=clock.t))
    assert float(action[5]) == pytest.approx(reference, abs=1e-6)


def test_request_anchor_resets_the_wrist_gate():
    clock = Clock(make_adapter())
    clock.act(0)  # anchor
    clock.act(10)
    clock.adapter.request_anchor()
    # The first frame after re-anchoring is accepted unconditionally and defines the new
    # offset; a step from it then tracks normally instead of being held.
    clock.act(200)
    assert clock.act(220) == pytest.approx(math.radians(20), abs=1e-3)

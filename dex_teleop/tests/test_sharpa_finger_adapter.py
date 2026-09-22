"""SharpaFingerActionAdapter: 22-D finger-only execution (no simulator required)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from dex_teleop.hands import SHARPA_ACTION_JOINTS
from dex_teleop.omnigibson.sharpa_finger_adapter import (
    SharpaFingerActionAdapter,
    SharpaFingerAdapterConfig,
)
from dex_teleop.runtime import SafetyConfig


class FakeRobot:
    """Franka + Sharpa whose arm is held by a zero-width NullJointController."""

    def __init__(self, action_dim=22, finger_positions=None):
        self.arm_names = ["0"]
        self.action_dim = action_dim
        self.dof_names_ordered = [f"panda_joint{i}" for i in range(1, 8)] + list(SHARPA_ACTION_JOINTS)
        self.controller_joint_idx = {"gripper_0": th.arange(7, 29)}
        self.finger_positions = np.zeros(22) if finger_positions is None else np.asarray(finger_positions)

    def get_joint_positions(self):
        return th.as_tensor(np.concatenate([np.zeros(7), self.finger_positions]), dtype=th.float32)


def snapshot(positions_by_name):
    names = tuple(positions_by_name)
    return SimpleNamespace(
        command=SimpleNamespace(
            hand_model="sharpa",
            handedness=SimpleNamespace(value="right"),
            joint_names=names,
            joint_positions=np.array([positions_by_name[name] for name in names]),
        ),
        frame=SimpleNamespace(wrist_position=np.array([5.0, 5.0, 5.0])),
    )


def make_adapter(robot=None, config=None, safety_config=None):
    return SharpaFingerActionAdapter(
        robot or FakeRobot(),
        config=config or SharpaFingerAdapterConfig(control_hz=30.0),
        safety_config=safety_config or SafetyConfig(enabled=False),
    )


def test_action_is_finger_only_and_ordered_by_declared_joint_names():
    adapter = make_adapter()
    targets = {name: 0.01 * index for index, name in enumerate(SHARPA_ACTION_JOINTS)}
    shuffled = dict(reversed(list(targets.items())))

    action = adapter.action(snapshot(shuffled))

    assert action.shape == (22,)
    assert action.dtype == th.float32
    assert np.allclose(action.numpy(), [targets[name] for name in SHARPA_ACTION_JOINTS])
    assert adapter.last_live_fingers == pytest.approx(targets)


def test_requires_a_22d_action_space():
    with pytest.raises(ValueError, match="NullJointController.*28-D"):
        make_adapter(FakeRobot(action_dim=28))


def test_rejects_wrong_hand_or_joint_set():
    adapter = make_adapter()
    bad_hand = snapshot({name: 0.0 for name in SHARPA_ACTION_JOINTS})
    bad_hand.command.handedness = SimpleNamespace(value="left")
    with pytest.raises(ValueError, match="right-hand Sharpa"):
        adapter.action(bad_hand)
    missing = {name: 0.0 for name in SHARPA_ACTION_JOINTS[:-1]}
    missing["unknown_joint"] = 0.0
    with pytest.raises(ValueError, match="missing=.*right_pinky_DIP.*extra=.*unknown_joint"):
        adapter.action(snapshot(missing))


def test_finger_target_scale_and_rate_limit():
    robot = FakeRobot()
    adapter = make_adapter(
        robot,
        config=SharpaFingerAdapterConfig(control_hz=30.0, finger_target_scale=0.5),
        safety_config=SafetyConfig(enabled=True, smoothing_alpha=1.0, max_hand_velocity=100.0, max_hand_delta_per_tick=0.1),
    )
    action = adapter.action(snapshot({name: 1.0 for name in SHARPA_ACTION_JOINTS}))

    # Scaled to 0.5 but rate-limited to 0.1 rad on the first tick from the measured zeros.
    assert np.allclose(action.numpy(), 0.1)
    assert adapter.last_live_fingers[SHARPA_ACTION_JOINTS[0]] == pytest.approx(0.5)


def test_reset_reseeds_from_measured_fingers():
    robot = FakeRobot(finger_positions=np.full(22, 0.3))
    adapter = make_adapter(
        robot,
        safety_config=SafetyConfig(enabled=True, smoothing_alpha=1.0, max_hand_velocity=100.0, max_hand_delta_per_tick=0.05),
    )
    assert adapter.measured_fingers[SHARPA_ACTION_JOINTS[3]] == pytest.approx(0.3)
    adapter.action(snapshot({name: 0.0 for name in SHARPA_ACTION_JOINTS}))
    robot.finger_positions = np.full(22, 0.8)
    adapter.reset()

    action = adapter.action(snapshot({name: 0.8 for name in SHARPA_ACTION_JOINTS}))

    assert np.allclose(action.numpy(), 0.8)
    assert adapter.last_live_fingers is not None


@pytest.mark.parametrize("config", [SharpaFingerAdapterConfig(control_hz=0.0), SharpaFingerAdapterConfig(finger_target_scale=1.5)])
def test_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        make_adapter(config=config)

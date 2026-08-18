import numpy as np
import pytest

from dex_teleop.hands import HAND_PROFILES
from dex_teleop.retargeting import LandmarkRetargeter


LANDMARKS = np.array(
    [
        [0, 0, 0],
        [-0.03, 0.01, 0], [-0.05, 0.03, 0], [-0.065, 0.055, 0], [-0.075, 0.08, 0],
        [-0.02, 0.04, 0], [-0.02, 0.075, 0], [-0.02, 0.105, 0], [-0.02, 0.13, 0],
        [0, 0.045, 0], [0, 0.085, 0], [0, 0.12, 0], [0, 0.15, 0],
        [0.02, 0.04, 0], [0.025, 0.078, 0], [0.03, 0.11, 0], [0.035, 0.137, 0],
        [0.04, 0.03, 0], [0.05, 0.06, 0], [0.058, 0.085, 0], [0.065, 0.108, 0],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize("hand_model", tuple(HAND_PROFILES))
def test_each_hand_model_retargets_landmarks(hand_model):
    profile = HAND_PROFILES[hand_model]
    retargeter = LandmarkRetargeter.from_hand_model(hand_model, "right")

    positions = retargeter.retarget_landmarks(LANDMARKS, apply_filter=False)

    assert positions.shape == (profile.degrees_of_freedom,)
    assert np.isfinite(positions).all()


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("hand_model", tuple(HAND_PROFILES))
def test_packaged_urdfs_cover_both_hands(hand_model, side):
    retargeter = LandmarkRetargeter.from_hand_model(hand_model, side)

    assert len(retargeter.joint_names) == HAND_PROFILES[hand_model].degrees_of_freedom

"""Unit tests for movement-driven assisted grasping (no simulator required)."""

from dataclasses import replace

import pytest

from dex_teleop.arat.eval.hand_model import HandSemantics
from dex_teleop.hands import SHARPA_ACTION_JOINTS
from dex_teleop.omnigibson.assisted_grasp import (
    AssistedGraspConfig,
    AssistedGraspSupervisor,
    DigitContact,
    can_hold,
    can_hold_digits,
    has_opposition,
)

UP = (0.0, 0.0, 1.0)
DOWN = (0.0, 0.0, -1.0)
LEFT = (-1.0, 0.0, 0.0)
RIGHT = (1.0, 0.0, 0.0)

# Clean multiples of the 0.1 s test step so no window check lands on a float boundary:
# welds on the 3rd consecutive step, releases on the 4th.
DT = 0.1
TEST_CONFIG = AssistedGraspConfig(grasp_window_s=0.25, release_window_s=0.35)
WELD_STEPS = 3
RELEASE_STEPS = 4

MAX_DOT = TEST_CONFIG.opposition_max_normal_dot

LIVE_CLOSED = 0.9
MEASURED_CLOSED = 0.8  # fingers physically stopped at the object surface


def digit_of(joint_name):
    return joint_name.split("_")[1]


def hand(flexion, aa=0.0):
    return {name: (aa if name.endswith("_AA") else flexion) for name in SHARPA_ACTION_JOINTS}


def opened(fingers, digits, amount=0.6):
    out = dict(fingers)
    for name in out:
        if digit_of(name) in digits and not name.endswith("_AA"):
            out[name] -= amount
    return out


LIVE = hand(LIVE_CLOSED)
MEASURED = hand(MEASURED_CLOSED)


def contact(digit, normal, part="pad"):
    return DigitContact(digit=digit, part=part, normal=normal)


# --------------------------------------------------------------------------- decision layer


def test_can_hold_requires_two_fingers_or_palm_plus_finger():
    assert not can_hold([])
    assert not can_hold([contact("index", UP)])
    assert not can_hold([contact("index", UP), contact("index", DOWN)])  # one finger, two links
    assert not can_hold([contact("palm", UP)])
    assert can_hold([contact("thumb", UP), contact("ring", DOWN)])
    assert can_hold([contact("palm", UP), contact("index", DOWN)])
    assert not can_hold_digits({"thumb"})
    assert not can_hold_digits({"palm"})
    assert can_hold_digits({"palm", "middle"})
    assert can_hold_digits({"index", "middle"})


def test_pinch_between_thumb_and_ring_opposes():
    assert has_opposition([contact("thumb", LEFT), contact("ring", RIGHT)], MAX_DOT)


def test_two_links_of_one_finger_do_not_oppose():
    assert not has_opposition([contact("index", LEFT), contact("index", RIGHT)], MAX_DOT)


def test_flat_hand_resting_on_object_does_not_oppose():
    resting = [contact("thumb", UP), contact("index", UP), contact("middle", UP)]
    assert not has_opposition(resting, MAX_DOT)


def test_corner_grasp_at_ninety_degrees_opposes():
    assert has_opposition([contact("thumb", UP), contact("index", RIGHT)], MAX_DOT)


def test_palm_and_finger_oppose():
    assert has_opposition([contact("palm", UP, part="palm"), contact("middle", DOWN)], MAX_DOT)


def test_contacts_without_normals_never_oppose():
    assert not has_opposition([contact("thumb", None), contact("ring", None)], MAX_DOT)


# --------------------------------------------------------------------------- supervisor


class FakeLink:
    def __init__(self, prim_path, position=(0.0, 0.0, 0.0), mass=0.1):
        self.prim_path = prim_path
        self.mass = mass
        self._position = position

    def get_position_orientation(self):
        return self._position, (0.0, 0.0, 0.0, 1.0)


class FakeObject:
    def __init__(self, name, prim_path, link_names, *, fixed_base=False, kinematic_only=False, mass=0.1):
        self.name = name
        self.prim_path = prim_path
        self.fixed_base = fixed_base
        self.kinematic_only = kinematic_only
        self.root_link_name = link_names[0]
        self.links = {link_name: FakeLink(f"{prim_path}/{link_name}", mass=mass) for link_name in link_names}

    def get_position_orientation(self):
        return self.links[self.root_link_name].get_position_orientation()


class FakeScene:
    def __init__(self, objects):
        self._by_prim_path = {obj.prim_path: obj for obj in objects}

    def object_registry(self, key, value, default=None):
        assert key == "prim_path"
        return self._by_prim_path.get(value, default)


class FakeRobot:
    _grasping_direction = "upper"

    def __init__(self, objects, grasping_mode="assisted", disable_grasp_handling=True, weld_succeeds=True):
        self.grasping_mode = grasping_mode
        self._disable_grasp_handling = disable_grasp_handling
        self.arm_names = ["0"]
        self._ag_obj_in_hand = {"0": None}
        self.scene = FakeScene(objects)
        self.eef_links = {"0": FakeLink("/World/robot/right_hand_C_MC")}
        self.contact_rows = []
        self.establish_calls = []
        self.release_calls = 0
        self._weld_succeeds = weld_succeeds

    def get_finger_contact_data(self, arm):
        assert arm == "0"
        return list(self.contact_rows)

    def _get_assisted_grasp_joint_type(self, target_obj, target_link_name):
        return None if target_obj.links[target_link_name].mass > 10.0 else "FixedJoint"

    def _maybe_establish_grasp(self, target_obj, target_link_name, arm):
        self.establish_calls.append((target_obj.name, target_link_name))
        if self._weld_succeeds:
            self._ag_obj_in_hand[arm] = target_obj

    def release_grasp_immediately(self, arm="default"):
        if self._ag_obj_in_hand[arm] is not None:
            self.release_calls += 1
        self._ag_obj_in_hand[arm] = None


def hand_path(link_name):
    return f"/World/robot/{link_name}"


def contact_row(hand_link, obj, normal, link_name=None):
    link_name = obj.root_link_name if link_name is None else link_name
    return (hand_path(hand_link), f"{obj.prim_path}/{link_name}", (0.0, 0.0, 0.0), normal, (0.0, 0.0, 0.0), 0.0)


def pinch_rows(obj, link_name=None):
    return [
        contact_row("right_thumb_elastomer", obj, LEFT, link_name),
        contact_row("right_ring_elastomer", obj, RIGHT, link_name),
    ]


def make_supervisor(robot, announce=None, config=TEST_CONFIG):
    return AssistedGraspSupervisor(robot, HandSemantics.sharpa("right"), dt=DT, config=config, announce=announce)


def block(name="block", position=(0.1, 0.0, 0.0), **kwargs):
    obj = FakeObject(name, f"/World/scene_0/{name}", ["base_link"], **kwargs)
    obj.links["base_link"]._position = position
    return obj


def weld_pinch(supervisor, robot, obj):
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj


def test_supervisor_welds_after_sustained_opposition():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []
    supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("block", "base_link")]
    assert robot._ag_obj_in_hand["0"] is obj
    assert messages == ["assisted grasp welded block:base_link via ring+thumb"]


def test_weld_freezes_grasping_digits_at_measured_posture():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    frozen = supervisor.frozen_fingers
    assert frozen is not None
    assert set(frozen) == {n for n in SHARPA_ACTION_JOINTS if digit_of(n) in ("thumb", "ring")}
    for name, value in frozen.items():
        if name.endswith("_AA"):
            assert value == pytest.approx(0.0)  # abduction frozen where the finger physically is
        else:
            # min(live 0.9, measured 0.8 + bias 0.05): surface posture plus gentle pressure
            assert value == pytest.approx(MEASURED_CLOSED + TEST_CONFIG.frozen_squeeze_bias_rad)


def test_freeze_never_exceeds_the_operators_live_command():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    barely_closed = hand(0.3)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS):
        supervisor.step(barely_closed, MEASURED)
    for name, value in supervisor.frozen_fingers.items():
        if not name.endswith("_AA"):
            assert value == pytest.approx(0.3)  # capped at live, not measured + bias


def test_supervisor_window_resets_when_contact_is_lost():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    robot.contact_rows = []
    supervisor.step(LIVE, MEASURED)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []
    supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj


def test_supervisor_never_welds_a_resting_hand():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = [
        contact_row("right_thumb_elastomer", obj, UP),
        contact_row("right_index_elastomer", obj, UP),
        contact_row("right_middle_elastomer", obj, UP),
    ]
    for _ in range(WELD_STEPS * 4):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []


def test_supervisor_skips_fixed_root_and_kinematic_and_heavy_objects():
    fixed = block("mannequin", fixed_base=True)
    kinematic = block("prop", kinematic_only=True)
    heavy = block("anvil", mass=25.0)
    robot = FakeRobot([fixed, kinematic, heavy])
    supervisor = make_supervisor(robot)
    for obj in (fixed, kinematic, heavy):
        robot.contact_rows = pinch_rows(obj)
        for _ in range(WELD_STEPS * 2):
            supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []


def test_supervisor_allows_articulated_link_of_fixed_object():
    box = FakeObject("arat_box", "/World/scene_0/arat_box", ["base_link", "front_cover_link"], fixed_base=True)
    robot = FakeRobot([box])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(box, link_name="front_cover_link")
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("arat_box", "front_cover_link")]


def test_supervisor_prefers_the_candidate_closest_to_the_palm():
    near = block("near", position=(0.1, 0.0, 0.0))
    far = block("far", position=(0.5, 0.0, 0.0))
    robot = FakeRobot([near, far])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(far) + pinch_rows(near)
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("near", "base_link")]


def test_supervisor_releases_when_the_grasping_digits_open():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    weld_pinch(supervisor, robot, obj)
    open_live = opened(LIVE, ("thumb", "ring"))
    for _ in range(RELEASE_STEPS - 1):
        supervisor.step(open_live, MEASURED)
    assert robot.release_calls == 0
    supervisor.step(open_live, MEASURED)
    assert robot.release_calls == 1
    assert robot._ag_obj_in_hand["0"] is None
    assert supervisor.frozen_fingers is None
    assert messages[-1] == "assisted grasp released block"


def test_supervisor_releases_when_opposition_breaks_on_one_side():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    open_live = opened(LIVE, ("ring",))  # only the ring opens; thumb alone cannot hold
    for _ in range(RELEASE_STEPS):
        supervisor.step(open_live, MEASURED)
    assert robot.release_calls == 1


def test_supervisor_spherical_hold_survives_a_single_digit_glitch():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = [
        contact_row("right_thumb_elastomer", obj, LEFT),
        contact_row("right_index_elastomer", obj, RIGHT),
        contact_row("right_middle_elastomer", obj, RIGHT),
        contact_row("right_ring_elastomer", obj, RIGHT),
    ]
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    glitched = opened(LIVE, ("pinky", "ring"))  # pinky is not even a grasp digit
    for _ in range(RELEASE_STEPS * 5):
        supervisor.step(glitched, MEASURED)
    assert robot.release_calls == 0  # thumb+index+middle still engaged


def test_supervisor_ignores_sub_margin_and_abduction_movement():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    slightly_open = opened(LIVE, ("thumb", "ring"), amount=0.1)  # below the 0.25 rad margin
    for name in slightly_open:
        if name.endswith("_AA"):
            slightly_open[name] = 0.4  # abduction wiggles must not count as opening
    for _ in range(RELEASE_STEPS * 5):
        supervisor.step(slightly_open, MEASURED)
    assert robot.release_calls == 0


def test_supervisor_release_survives_opening_flicker():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    open_live = opened(LIVE, ("thumb", "ring"))
    for _ in range(2):
        for _ in range(RELEASE_STEPS - 1):
            supervisor.step(open_live, MEASURED)
        supervisor.step(LIVE, MEASURED)  # re-closed before the window elapsed
    assert robot.release_calls == 0
    assert robot._ag_obj_in_hand["0"] is obj


def test_supervisor_palm_grasp_releases_only_when_its_fingers_open():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = [
        contact_row("right_hand_C_MC", obj, UP),
        contact_row("right_thumb_elastomer", obj, DOWN),
        contact_row("right_index_elastomer", obj, DOWN),
    ]
    for _ in range(WELD_STEPS):
        supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    one_open = opened(LIVE, ("index",))
    for _ in range(RELEASE_STEPS * 3):
        supervisor.step(one_open, MEASURED)  # palm + thumb still hold
    assert robot.release_calls == 0
    both_open = opened(LIVE, ("thumb", "index"))
    for _ in range(RELEASE_STEPS):
        supervisor.step(both_open, MEASURED)  # the palm alone cannot hold
    assert robot.release_calls == 1


def test_supervisor_only_welds_when_the_thumb_is_involved():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    thumbless = [
        contact_row("right_index_elastomer", obj, LEFT),
        contact_row("right_middle_elastomer", obj, RIGHT),
        contact_row("right_hand_C_MC", obj, UP),
    ]
    robot.contact_rows = thumbless
    for _ in range(WELD_STEPS * 3):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []  # opposed and holdable, but no thumb
    permissive = make_supervisor(FakeRobot([obj]), config=replace(TEST_CONFIG, require_thumb=False))
    permissive.robot.contact_rows = thumbless
    for _ in range(WELD_STEPS):
        permissive.step(LIVE, MEASURED)
    assert permissive.robot.establish_calls == [("block", "base_link")]


def test_supervisor_treats_a_broken_weld_as_released():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    weld_pinch(supervisor, robot, obj)
    obj.links["base_link"]._position = (0.13, 0.0, 0.0)  # 3 cm of drift: weld strained, still credible
    supervisor.step(LIVE, MEASURED)
    assert robot.release_calls == 0
    obj.links["base_link"]._position = (0.22, 0.0, 0.0)  # 12 cm: the weld is gone
    supervisor.step(LIVE, MEASURED)
    assert robot.release_calls == 1
    assert supervisor.frozen_fingers is None
    assert "broke" in messages[-1]


def test_supervisor_adopts_an_unknown_weld_until_the_hand_opens():
    obj = block()
    robot = FakeRobot([obj])
    messages = []
    supervisor = make_supervisor(robot, announce=messages.append)
    robot._ag_obj_in_hand["0"] = obj  # e.g. restored from a saved state
    supervisor.step(LIVE, MEASURED)
    assert "adopted" in messages[0]
    assert set(supervisor.frozen_fingers) == set(SHARPA_ACTION_JOINTS)
    all_open = opened(LIVE, ("thumb", "index", "middle", "ring", "pinky"))
    for _ in range(RELEASE_STEPS):
        supervisor.step(all_open, MEASURED)
    assert robot.release_calls == 1


def test_supervisor_regrasps_after_release():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    weld_pinch(supervisor, robot, obj)
    robot.contact_rows = []
    open_live = opened(LIVE, ("thumb", "ring"))
    for _ in range(RELEASE_STEPS):
        supervisor.step(open_live, MEASURED)
    assert robot._ag_obj_in_hand["0"] is None
    weld_pinch(supervisor, robot, obj)
    assert len(robot.establish_calls) == 2


def test_supervisor_retries_a_full_window_after_a_failed_weld():
    obj = block()
    robot = FakeRobot([obj], weld_succeeds=False)
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS * 2):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == [("block", "base_link")] * 2


def test_supervisor_reset_clears_pending_window_freeze_and_weld():
    obj = block()
    robot = FakeRobot([obj])
    supervisor = make_supervisor(robot)
    robot.contact_rows = pinch_rows(obj)
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    supervisor.reset()
    for _ in range(WELD_STEPS - 1):
        supervisor.step(LIVE, MEASURED)
    assert robot.establish_calls == []
    supervisor.step(LIVE, MEASURED)
    assert robot._ag_obj_in_hand["0"] is obj
    assert supervisor.frozen_fingers is not None
    supervisor.reset()
    assert robot.release_calls == 1
    assert robot._ag_obj_in_hand["0"] is None
    assert supervisor.frozen_fingers is None


def test_supervisor_requires_assisted_mode_with_handling_disabled():
    with pytest.raises(ValueError, match="grasping_mode"):
        make_supervisor(FakeRobot([], grasping_mode="physical"))
    with pytest.raises(ValueError, match="disable_grasp_handling"):
        make_supervisor(FakeRobot([], disable_grasp_handling=False))
    lower = FakeRobot([])
    lower._grasping_direction = "lower"
    with pytest.raises(ValueError, match="grasping_direction"):
        make_supervisor(lower)

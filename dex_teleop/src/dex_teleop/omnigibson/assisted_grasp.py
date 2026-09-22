"""Movement-driven assisted grasping for the teleoperated dexterous hand.

JoyLo enables OmniGibson's built-in assisted grasping by constructing its robot with
``grasping_mode="assisted"``: once the gripper closes around an object, a constraint welds
the object to the hand so it can be carried reliably. The built-in decision logic is
written for simple grippers and pre-determines the participating fingers:

- Its ray filter runs only between the robot definition's fixed AG points (for the Sharpa
  hand: thumb tip to index and middle tips), so a grasp using any other digit combination
  -- e.g. the ARAT thumb-ring pinch -- can never trigger.
- Its closing test treats any gripper DOF commanded away from its open limit as "applying
  a grasp". A retargeted 22-DOF hand always has joints away from their limits, so a grasp
  would trigger on any sustained touch and, once established, never release.

This module keeps OmniGibson's assisted-grasp machinery (constraint creation, release,
state serialization for recordings and replay) but derives the grasp decision from what
the teleoperator's hand is actually doing: which digits touch the candidate object, and
whether their contact normals oppose. The thumb must be among the touching digits (every
ARAT grasp is thumb-opposed), but beyond that the digits taking part in a grasp are
chosen by the operator's movement, never by the robot definition.

While an object is welded, the grasping digits' joints are frozen (the launcher feeds
:attr:`AssistedGraspSupervisor.frozen_fingers` back into the action adapter) so the hand
visibly holds its grasp pose instead of wiggling around a rigidly attached object, and so
the operator's deepening squeeze cannot fight the weld through the arm. Uninvolved
fingers keep tracking. Because frozen fingers hold their contacts forever, release is
decided from the operator's *commanded* hand instead of from contacts: the weld drops
once the digits still flexed near their grasp posture no longer form a holdable set.

The supervisor drives the robot's grasp-handling methods directly, which is the usage
OmniGibson documents for ``disable_grasp_handling=True``. The decision layer operates on
plain data so it can be unit-tested without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import logging
import math
from typing import Callable, Iterable, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

PALM = "palm"
THUMB = "thumb"


@dataclass(frozen=True)
class DigitContact:
    """One contact point between a hand part and a candidate object link."""

    digit: str  # "thumb" | "index" | "middle" | "ring" | "pinky" | "palm"
    part: str  # "pad" | "dorsal" | "palm"
    normal: tuple[float, float, float] | None  # world frame; None when unavailable


@dataclass(frozen=True)
class AssistedGraspConfig:
    # Mirrors OmniGibson's m.GRASP_WINDOW: opposition must persist this long to weld.
    grasp_window_s: float = 0.3
    # How long the operator's hand must stay opened before the weld drops; absorbs
    # tracking noise and brief dropouts while carrying.
    release_window_s: float = 0.15
    # Two contacts from different digits oppose when their world normals are at least
    # 90 degrees apart (dot <= 0.0). This accepts corner and lateral grasps while
    # rejecting several digits resting on the same face of the object.
    opposition_max_normal_dot: float = 0.0
    # A grasping digit counts as opened once its flexion commands retreat this far (mean
    # over the digit's non-abduction joints) from their values at weld time.
    release_open_margin_rad: float = 0.25
    # Frozen flexion targets sit at the measured joint positions plus this closing bias,
    # capped at the operator's live command: enough sustained pressure to keep firm
    # contact, without the deep-penetration squeeze that kicks the arm through the weld.
    frozen_squeeze_bias_rad: float = 0.05
    # A weld only establishes when the thumb touches the object: every ARAT grasp is
    # thumb-opposed, and thumb-less patterns (the scoring guide's inappropriate or
    # score-0 movements) should not be magically assisted. Release is unaffected.
    require_thumb: bool = True
    # Snap the weld before the operator can wedge the held object deep into static
    # geometry (a real grip would slip); authored as physics:breakForce / breakTorque on
    # the weld joint. None disables either limit.
    weld_break_force: float | None = 100.0
    weld_break_torque: float | None = 30.0
    # If the held object separates this far from its weld-time offset to the palm, the
    # weld has broken (or is being violated beyond credibility); treat it as released.
    weld_drift_release_m: float = 0.05


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def can_hold_digits(digits: Iterable[str]) -> bool:
    """Digits able to keep holding: two distinct fingers, or the palm plus a finger."""
    digits = frozenset(digits)
    fingers = digits - {PALM}
    return len(fingers) >= 2 or (PALM in digits and len(fingers) >= 1)


def can_hold(contacts: Sequence[DigitContact]) -> bool:
    """Contact-pattern form of :func:`can_hold_digits`."""
    return can_hold_digits(contact.digit for contact in contacts)


def has_opposition(contacts: Sequence[DigitContact], max_normal_dot: float) -> bool:
    """Whether the contact pattern pinches the object rather than resting on it.

    All normals come from the robot's contact view with one consistent sensor/filter
    convention, so the pairwise comparison is invariant to which way that convention
    points: a pinch yields anti-parallel normals, digits resting on the same face
    yield parallel ones.
    """
    if not can_hold(contacts):
        return False
    for first, second in combinations(contacts, 2):
        if first.digit == second.digit or first.normal is None or second.normal is None:
            continue
        if _dot(first.normal, second.normal) <= max_normal_dot:
            return True
    return False


def _as_normal(vector) -> tuple[float, float, float] | None:
    normal = (float(vector[0]), float(vector[1]), float(vector[2]))
    if normal == (0.0, 0.0, 0.0):
        return None
    return normal


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotate_inverse(quaternion_xyzw, vector) -> tuple[float, float, float]:
    """Rotate a world-frame vector into the frame of the given (x, y, z, w) quaternion."""
    x, y, z, w = (float(component) for component in quaternion_xyzw)
    axis = (-x, -y, -z)
    vector = tuple(float(component) for component in vector)
    doubled = tuple(2.0 * component for component in _cross(axis, vector))
    correction = _cross(axis, doubled)
    return tuple(vector[i] + w * doubled[i] + correction[i] for i in range(3))


class AssistedGraspSupervisor:
    """Drives one arm's OmniGibson assisted-grasp machinery from hand contacts.

    Call :meth:`step` once per control step while teleoperation is engaged, and
    :meth:`reset` whenever the environment resets.
    """

    def __init__(
        self,
        robot,
        semantics,
        *,
        dt: float,
        config: AssistedGraspConfig = AssistedGraspConfig(),
        announce: Callable[[str], None] | None = None,
    ) -> None:
        if robot.grasping_mode != "assisted":
            raise ValueError("Assisted-grasp supervision requires grasping_mode='assisted'")
        if not robot._disable_grasp_handling:
            raise ValueError(
                "Assisted-grasp supervision requires disable_grasp_handling=True so the built-in "
                "closing-command heuristic does not fight the contact-driven decisions"
            )
        if getattr(robot, "_grasping_direction", "upper") != "upper":
            raise ValueError(
                "Assisted-grasp supervision assumes grasping_direction='upper' (larger joint value = closed)"
            )
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.robot = robot
        self.config = config
        self._semantics = semantics
        self._dt = dt
        self._announce = announce
        self._arm = robot.arm_names[0]
        self._pending_key: tuple[str, str] | None = None
        self._pending_elapsed = 0.0
        self._release_elapsed = 0.0
        # While holding: the digits that formed the grasp, the frozen targets fed back to
        # the action adapter, the operator's live posture at weld time (release baseline),
        # each grasping digit's flexion joints, and the object's weld-time offset in the
        # palm frame (for detecting a broken or violated weld).
        self._grasp_digits: frozenset[str] = frozenset()
        self._frozen_fingers: dict[str, float] | None = None
        self._open_baseline: dict[str, float] = {}
        self._flexion_by_digit: dict[str, tuple[str, ...]] = {}
        self._weld_offset: tuple[float, float, float] | None = None
        self._last_event: dict | None = None

    @property
    def arm(self) -> str:
        return self._arm

    @property
    def control_dt(self) -> float:
        return self._dt

    @property
    def last_event(self) -> dict | None:
        """Structured event emitted by the most recent :meth:`step`, if any."""

        return None if self._last_event is None else dict(self._last_event)

    @property
    def frozen_fingers(self) -> dict[str, float] | None:
        """Joint targets to hold while grasping (by joint name), or None when not holding.

        The launcher feeds this into ``SharpaActionAdapter.action(..., frozen_fingers=...)``
        so the grasping digits keep their weld-time posture while uninvolved fingers track.
        """
        return None if self._frozen_fingers is None else dict(self._frozen_fingers)

    def reset(self) -> None:
        """Clear decision windows and drop any active weld (no-op when none exists)."""
        self._last_event = None
        self._pending_key = None
        self._pending_elapsed = 0.0
        self._release_elapsed = 0.0
        self._clear_freeze()
        self.robot.release_grasp_immediately(arm=self._arm)

    def step(self, live_fingers: Mapping[str, float], measured_fingers: Mapping[str, float]) -> None:
        """Advance the grasp/release decision by one control step.

        Args:
            live_fingers: the operator's current retargeted finger targets by joint name
                (what would be commanded without any freeze)
            measured_fingers: the robot's current measured finger joint positions by joint name
        """
        self._last_event = None
        held = self.robot._ag_obj_in_hand[self._arm]
        if held is not None:
            if self._frozen_fingers is None:
                # A weld this supervisor did not create (e.g. restored from a saved state).
                self._adopt_existing_grasp(held, live_fingers, measured_fingers)
            self._step_release(held, live_fingers)
        else:
            self._release_elapsed = 0.0
            self._clear_freeze()
            self._step_establish(self._contacts_by_object_link(), live_fingers, measured_fingers)

    def _step_release(self, held, live_fingers) -> None:
        self._pending_key = None
        self._pending_elapsed = 0.0
        if self._weld_offset is not None:
            offset = self._held_offset(held)
            drift = math.sqrt(sum((a - b) ** 2 for a, b in zip(offset, self._weld_offset)))
            if drift > self.config.weld_drift_release_m:
                # PhysX broke the weld (break force) or is violating it beyond credibility
                # (object wedged into static geometry); stop pretending it is held.
                held_name = held.name
                self.robot.release_grasp_immediately(arm=self._arm)
                self._release_elapsed = 0.0
                self._clear_freeze()
                self._last_event = {"kind": "broke", "object": held_name, "drift_m": drift}
                self._say(f"assisted grasp broke on {held_name} (weld separated {drift * 100:.0f} cm)")
                return
        engaged = {digit for digit in self._grasp_digits if self._digit_engaged(digit, live_fingers)}
        if can_hold_digits(engaged):
            self._release_elapsed = 0.0
            return
        self._release_elapsed += self._dt
        if self._release_elapsed < self.config.release_window_s:
            return
        held_name = held.name
        self.robot.release_grasp_immediately(arm=self._arm)
        self._release_elapsed = 0.0
        self._clear_freeze()
        self._last_event = {"kind": "released", "object": held_name}
        self._say(f"assisted grasp released {held_name}")

    def _digit_engaged(self, digit: str, live_fingers: Mapping[str, float]) -> bool:
        """Whether the operator still holds this digit near its weld-time flexion."""
        flexion = self._flexion_by_digit.get(digit, ())
        retreats = [self._open_baseline[name] - float(live_fingers[name]) for name in flexion if name in live_fingers]
        if not retreats:
            return True  # the palm (or a digit without joint data) cannot signal opening
        return sum(retreats) / len(retreats) <= self.config.release_open_margin_rad

    def _capture_freeze(self, digits, live_fingers, measured_fingers) -> None:
        """Freeze the grasping digits' joints and record the operator's grasp posture."""
        side = self._semantics.side
        frozen: dict[str, float] = {}
        baseline: dict[str, float] = {}
        flexion_by_digit: dict[str, tuple[str, ...]] = {}
        for digit in digits:
            if digit == PALM:
                continue
            joints = tuple(name for name in live_fingers if name.startswith(f"{side}_{digit}_"))
            flexion = tuple(name for name in joints if not name.endswith("_AA"))
            flexion_by_digit[digit] = flexion
            for name in joints:
                live = float(live_fingers[name])
                measured = float(measured_fingers.get(name, live))
                if name in flexion:
                    # Hold gentle pressure at the surface instead of the operator's deep
                    # squeeze target, which would fight the weld through the arm.
                    frozen[name] = min(live, measured + self.config.frozen_squeeze_bias_rad)
                    baseline[name] = live
                else:
                    frozen[name] = measured
        self._grasp_digits = frozenset(digits)
        self._frozen_fingers = frozen
        self._open_baseline = baseline
        self._flexion_by_digit = flexion_by_digit

    def _adopt_existing_grasp(self, held, live_fingers, measured_fingers) -> None:
        side = self._semantics.side
        digits = frozenset(
            name.removeprefix(f"{side}_").split("_", 1)[0] for name in live_fingers if name.startswith(f"{side}_")
        )
        self._capture_freeze(digits, live_fingers, measured_fingers)
        self._weld_offset = self._held_offset(held)
        self._author_weld_break_limits()
        self._last_event = {"kind": "adopted", "object": held.name, "digits": sorted(digits)}
        self._say("assisted grasp adopted an existing weld; all fingers hold until the hand opens")

    def diagnostic_state(self) -> dict:
        """Return the decision and weld state in a JSON-safe representation."""

        held = self.robot._ag_obj_in_hand[self._arm]
        drift = None
        target_link_name = None
        if held is not None:
            params = getattr(self.robot, "_ag_obj_constraint_params", {}).get(self._arm)
            target_link_name = None if params is None else params.get("target_link_name")
            if self._weld_offset is not None:
                offset = self._held_offset(held)
                drift = math.sqrt(sum((a - b) ** 2 for a, b in zip(offset, self._weld_offset)))
        return {
            "status": "held" if held is not None else ("pending" if self._pending_key is not None else "idle"),
            "held_object": None if held is None else held.name,
            "target_link_name": target_link_name,
            "grasp_digits": sorted(self._grasp_digits),
            "pending_object": None if self._pending_key is None else self._pending_key[0],
            "pending_link_name": None if self._pending_key is None else self._pending_key[1],
            "pending_elapsed_s": self._pending_elapsed,
            "release_elapsed_s": self._release_elapsed,
            "weld_drift_m": drift,
        }

    def classify_hand_link(self, link_name: str):
        """Expose the shared hand-link classification to diagnostic collectors."""

        return self._semantics.classify_link(link_name)

    def _held_offset(self, held) -> tuple[float, float, float]:
        """The held object's position in the palm frame; constant while the weld is intact."""
        object_position = held.get_position_orientation()[0]
        eef_position, eef_quaternion = self.robot.eef_links[self._arm].get_position_orientation()
        world = tuple(float(o) - float(e) for o, e in zip(object_position, eef_position))
        return _rotate_inverse(eef_quaternion, world)

    def _author_weld_break_limits(self) -> None:
        """Author break force/torque on the weld joint so a wedged grip slips like a real one."""
        if self.config.weld_break_force is None and self.config.weld_break_torque is None:
            return
        constraints = getattr(self.robot, "_ag_obj_constraints", None)
        joint_prim = None if constraints is None else constraints.get(self._arm)
        if joint_prim is None:
            return
        import omnigibson as og

        with og.sim.editing_usd():
            if self.config.weld_break_force is not None:
                joint_prim.GetAttribute("physics:breakForce").Set(float(self.config.weld_break_force))
            if self.config.weld_break_torque is not None:
                joint_prim.GetAttribute("physics:breakTorque").Set(float(self.config.weld_break_torque))

    def _clear_freeze(self) -> None:
        self._grasp_digits = frozenset()
        self._frozen_fingers = None
        self._open_baseline = {}
        self._flexion_by_digit = {}
        self._weld_offset = None

    def _step_establish(self, contacts_by_link, live_fingers, measured_fingers) -> None:
        candidate = self._select_candidate(contacts_by_link)
        key = None if candidate is None else (candidate[0].name, candidate[1])
        if key != self._pending_key:
            self._pending_key = key
            self._pending_elapsed = 0.0
        if key is None:
            return
        self._pending_elapsed += self._dt
        if self._pending_elapsed < self.config.grasp_window_s:
            return
        self._pending_key = None
        self._pending_elapsed = 0.0
        obj, link_name, contacts = candidate
        self.robot._maybe_establish_grasp(target_obj=obj, target_link_name=link_name, arm=self._arm)
        if self.robot._ag_obj_in_hand[self._arm] is obj:
            digits = frozenset(contact.digit for contact in contacts)
            self._capture_freeze(digits, live_fingers, measured_fingers)
            self._weld_offset = self._held_offset(obj)
            self._author_weld_break_limits()
            self._last_event = {
                "kind": "welded",
                "object": obj.name,
                "target_link_name": link_name,
                "digits": sorted(digits),
            }
            self._say(f"assisted grasp welded {obj.name}:{link_name} via {'+'.join(sorted(digits))}")

    def _contacts_by_object_link(self):
        """Group the arm's current contact points as {(object name, link name): (obj, link name, contacts)}."""
        grouped: dict[tuple[str, str], tuple[object, str, list[DigitContact]]] = {}
        for hand_path, other_path, _position, normal, _force, _separation in self.robot.get_finger_contact_data(
            self._arm
        ):
            classified = self._semantics.classify_link(hand_path.rsplit("/", 1)[-1])
            if classified is None:
                continue
            digit, part = classified
            obj_prim_path, link_name = other_path.rsplit("/", 1)
            obj = self.robot.scene.object_registry("prim_path", obj_prim_path, None)
            if obj is None or link_name not in obj.links:
                continue
            entry = grouped.setdefault((obj.name, link_name), (obj, link_name, []))
            entry[2].append(DigitContact(digit=digit, part=part, normal=_as_normal(normal)))
        return grouped

    def _select_candidate(self, contacts_by_link):
        candidates = []
        for obj, link_name, contacts in contacts_by_link.values():
            if self.config.require_thumb and not any(contact.digit == THUMB for contact in contacts):
                continue
            if not has_opposition(contacts, self.config.opposition_max_normal_dot):
                continue
            if not self._graspable(obj, link_name):
                continue
            candidates.append((self._distance_to_eef(obj.links[link_name]), (obj, link_name, contacts)))
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[0])[1]

    def _graspable(self, obj, link_name) -> bool:
        if getattr(obj, "kinematic_only", False):
            return False
        # The root link of a fixed-base object can never be lifted; welding to it would
        # pin the hand (table, tins, mannequin). Articulated links of fixed objects
        # (e.g. the box lid) keep OmniGibson's stock door-grasp behavior.
        if obj.fixed_base and link_name == obj.root_link_name:
            return False
        return self.robot._get_assisted_grasp_joint_type(obj, link_name) is not None

    def _distance_to_eef(self, link) -> float:
        eef_position = self.robot.eef_links[self._arm].get_position_orientation()[0]
        link_position = link.get_position_orientation()[0]
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(link_position, eef_position)))

    def _say(self, message: str) -> None:
        if self._announce is not None:
            self._announce(message)
        else:
            LOGGER.info(message)

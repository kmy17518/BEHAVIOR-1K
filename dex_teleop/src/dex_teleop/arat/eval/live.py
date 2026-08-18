"""Live snapshot extraction from the OmniGibson sim, feeding the ARAT item scorer.

This is the only module in ``dex_teleop.arat.eval`` that imports OmniGibson. It reads
per-link hand contacts (with contact points where available), classifies the tracked
object's support contacts, evaluates the item's placement target, and advances an
:class:`AratItemScorer` one engaged control step at a time.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
import math

import torch as th

import omnigibson as og  # noqa: F401  (imported for side effects / parity with launcher)
from omnigibson.object_states import AABB, ContainedParticles, OnShelf, Pegged
from omnigibson.utils.usd_utils import RigidContactAPI

from dex_teleop.arat.catalog import AratTask
from dex_teleop.arat.eval.hand_model import FINGERS, HandSemantics
from dex_teleop.arat.eval.regions import MANNEQUIN_REGIONS
from dex_teleop.arat.eval.rubrics import ItemRubric, ScoringConfig
from dex_teleop.arat.eval.scorer import AratItemScorer, ItemResult
from dex_teleop.arat.eval.snapshot import (
    AratStepSnapshot,
    HandContact,
    SupportContact,
    TargetState,
    TrackedObjectState,
    WaterState,
)

LOGGER = logging.getLogger(__name__)

# The ARAT mannequin faces world -X (toward the robot)
MANNEQUIN_FRONT_DIR_XY = (-1.0, 0.0)
# Vertical slack used when classifying a contact as support-from-below
SUPPORT_BELOW_SLACK_M = 0.005
SUPPORT_VERTICAL_RANGE_M = 0.01
SUPPORT_FOOTPRINT_MARGIN_M = 0.01
TIN_DISC_SHRINK = 0.8
TARGET_HEIGHT_SLACK_M = 0.02


def _to_tuple(tensor) -> tuple[float, float, float]:
    values = [float(v) for v in tensor]
    return (values[0], values[1], values[2])


class LiveAratEvaluator:
    """Owns snapshot extraction and one :class:`AratItemScorer` for one ARAT item."""

    def __init__(self, env, robot, task: AratTask, rubric: ItemRubric, config: ScoringConfig | None = None):
        self._env = env
        self._robot = robot
        self._task = task
        self.rubric = rubric
        self.config = config or ScoringConfig()
        self._hand = HandSemantics.sharpa(side="right")
        self._arm = robot.default_arm
        self._warned_contact_data = False

        self._tracked = self._resolve_optional(rubric.tracked_instance)
        target = rubric.target
        kind = target["kind"]
        self._shelf_obj = self._resolve_optional(target.get("shelf_instance"))
        self._tin_obj = self._resolve_optional(target.get("tin_instance"))
        self._peg_obj = self._resolve_optional(target.get("peg_instance"))
        self._region_obj = self._resolve_optional(target.get("region_instance"))
        self._region = MANNEQUIN_REGIONS[target["region"]] if kind == "region" else None

        self._water_system = None
        self._pour_source = None
        self._pour_dest = None
        if kind == "pour":
            self._water_system = env.scene.get_system("water")
            self._pour_source = self._resolve(target["source_instance"])
            self._pour_dest = self._resolve(target["dest_instance"])

        # The object whose hand contacts the scorer cares about
        self._object_of_interest = self._region_obj if kind == "region" else self._tracked

        self.scorer = AratItemScorer(rubric, self.config)
        self._task_time = 0.0
        self._palm_start_position: tuple[float, float, float] | None = None
        self._events_consumed = 0

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        """Restart scoring for this item (e.g. after the launcher's R-reset)."""
        self.scorer = AratItemScorer(self.rubric, self.config)
        self._task_time = 0.0
        self._palm_start_position = None
        self._events_consumed = 0

    @property
    def finished(self) -> bool:
        return self.scorer.finished

    @property
    def task_time(self) -> float:
        return self._task_time

    def finalize(self) -> ItemResult:
        return self.scorer.finalize()

    def consume_new_events(self):
        events = self.scorer.events[self._events_consumed :]
        self._events_consumed = len(self.scorer.events)
        return events

    def evaluation_trace(self, snapshot: AratStepSnapshot) -> dict:
        """Return the complete recorded evidence and score decision for one step."""

        return {
            "snapshot": asdict(snapshot),
            **self.scorer.evaluation_breakdown(snapshot),
        }

    # ------------------------------------------------------------------ stepping

    def step(self, engaged: bool) -> AratStepSnapshot | None:
        """Advance one control step. Only engaged steps consume task time."""
        if not engaged or self.scorer.finished:
            return None
        self._task_time += self.config.control_dt
        snapshot = self._build_snapshot()
        self.scorer.step(snapshot)
        return snapshot

    # ------------------------------------------------------------------ extraction

    def _resolve(self, instance: str):
        obj = self._resolve_optional(instance)
        if obj is None:
            raise RuntimeError(f"{self.rubric.activity}: BDDL instance {instance!r} did not resolve to a scene object")
        return obj

    def _resolve_optional(self, instance):
        if instance is None:
            return None
        if instance == "agent.n.01_1":
            return self._robot
        scene_name = self._task.instances.get(instance)
        if scene_name is None:
            raise RuntimeError(f"{self.rubric.activity}: instance {instance!r} missing from the task catalog map")
        return self._env.scene.object_registry("name", scene_name)

    def _owner_of(self, link_prim_path: str):
        parent = link_prim_path.rsplit("/", 1)[0]
        return self._env.scene.object_registry("prim_path", parent, None)

    def _build_snapshot(self) -> AratStepSnapshot:
        hand_contacts = self._hand_contacts_with(self._object_of_interest)
        tracked_state = self._tracked_state() if self._tracked is not None else None
        apertures = self._apertures()
        water = self._water_state()
        target = self._target_state(hand_contacts, tracked_state, water)
        return AratStepSnapshot(
            t=self._task_time,
            hand_contacts=hand_contacts,
            tracked=tracked_state,
            apertures=apertures,
            target=target,
            water=water,
        )

    def _hand_contacts_with(self, obj) -> tuple[HandContact, ...]:
        if obj is None:
            return ()
        contacts: dict[tuple[str, str], HandContact] = {}

        # Contact points (position/normal) from the robot's private contact view
        try:
            point_data = self._robot.get_finger_contact_data(self._arm)
        except Exception:
            point_data = []
            if not self._warned_contact_data:
                LOGGER.warning("Finger contact-point data unavailable; falling back to boolean contacts")
                self._warned_contact_data = True
        for hand_path, other_path, position, normal, _force, _separation in point_data:
            owner = self._owner_of(other_path)
            if owner is not obj:
                continue
            hand_leaf = hand_path.rsplit("/", 1)[-1]
            classified = self._hand.classify_link(hand_leaf)
            if classified is None:
                continue
            finger, part = classified
            key = (hand_leaf, other_path.rsplit("/", 1)[-1])
            contacts[key] = HandContact(
                finger=finger,
                part=part,
                link_name=hand_leaf,
                other_link=key[1],
                position=_to_tuple(position),
                normal=_to_tuple(normal),
            )

        # Boolean contacts (windowed over the last sim step) fill in anything the
        # point view missed, with the hand link's own position as the location proxy
        _, robot_contact_links = self._robot._find_gripper_contacts(self._arm)
        palm_link = self._robot.links.get(self._hand.palm_link)
        extra_pairs = []
        for other_path, finger_paths in robot_contact_links.items():
            for finger_path in finger_paths:
                extra_pairs.append((finger_path, other_path))
        if palm_link is not None and RigidContactAPI.is_in_contact(
            scene_idx=self._env.scene.idx,
            query_set=[palm_link],
            with_set=[obj],
            ignore_set=None,
            current_only=False,
        ):
            extra_pairs.append((palm_link.prim_path, obj.root_link.prim_path))

        for hand_path, other_path in extra_pairs:
            owner = self._owner_of(other_path)
            if owner is not obj:
                continue
            hand_leaf = hand_path.rsplit("/", 1)[-1]
            classified = self._hand.classify_link(hand_leaf)
            if classified is None:
                continue
            key = (hand_leaf, other_path.rsplit("/", 1)[-1])
            if key in contacts:
                continue
            finger, part = classified
            link = self._robot.links.get(hand_leaf)
            position = _to_tuple(link.get_position_orientation()[0]) if link is not None else None
            contacts[key] = HandContact(
                finger=finger,
                part=part,
                link_name=hand_leaf,
                other_link=key[1],
                position=position,
                normal=None,
            )
        return tuple(contacts.values())

    def _tracked_state(self) -> TrackedObjectState:
        obj = self._tracked
        lower, upper = obj.states[AABB].get_value()
        position = _to_tuple((lower + upper) / 2.0)
        speed = float(th.norm(obj.root_link.get_linear_velocity()))
        supports = self._support_contacts(lower, upper)
        return TrackedObjectState(
            position=position,
            aabb_lo=_to_tuple(lower),
            aabb_hi=_to_tuple(upper),
            speed=speed,
            supports=supports,
        )

    def _support_contacts(self, lower, upper) -> tuple[SupportContact, ...]:
        pairs = RigidContactAPI.get_contact_pairs(
            scene_idx=self._env.scene.idx,
            query_set=[self._tracked],
            with_set=None,
            current_only=True,
        )
        robot_paths = set(self._robot.link_prim_paths)
        center_x = float(lower[0] + upper[0]) / 2.0
        center_y = float(lower[1] + upper[1]) / 2.0
        tracked_lo_z = float(lower[2])
        tracked_hi_z = float(upper[2])

        supports = []
        seen = set()
        for _tracked_path, other_path in pairs:
            if other_path in robot_paths or other_path in seen:
                continue
            seen.add(other_path)
            owner = self._owner_of(other_path)
            other_leaf = other_path.rsplit("/", 1)[-1]
            if owner is None:
                # Ground plane or other unregistered body
                relation = "below" if tracked_lo_z <= 0.02 else "side"
                supports.append(SupportContact(other_object=other_leaf, other_link=other_leaf, relation=relation))
                continue
            link = owner.links.get(other_leaf)
            if link is None:
                continue
            other_lo, other_hi = link.aabb
            over = (
                float(other_lo[0]) - SUPPORT_FOOTPRINT_MARGIN_M <= center_x <= float(other_hi[0]) + SUPPORT_FOOTPRINT_MARGIN_M
                and float(other_lo[1]) - SUPPORT_FOOTPRINT_MARGIN_M <= center_y <= float(other_hi[1]) + SUPPORT_FOOTPRINT_MARGIN_M
            )
            vertical = (
                float(other_lo[2]) - SUPPORT_VERTICAL_RANGE_M
                <= tracked_lo_z
                <= float(other_hi[2]) + SUPPORT_BELOW_SLACK_M
            )
            if over and vertical:
                relation = "below"
            elif tracked_hi_z <= float(other_lo[2]) + TARGET_HEIGHT_SLACK_M:
                relation = "above"
            else:
                relation = "side"
            supports.append(SupportContact(other_object=owner.name, other_link=other_leaf, relation=relation))
        return tuple(supports)

    def _apertures(self) -> dict[str, float]:
        links = self._robot.links
        thumb = links.get(self._hand.pad_links["thumb"])
        if thumb is None:
            return {}
        thumb_pos = thumb.get_position_orientation()[0]
        apertures = {}
        for finger in FINGERS:
            if finger == "thumb":
                continue
            pad = links.get(self._hand.pad_links[finger])
            if pad is None:
                continue
            apertures[finger] = float(th.norm(pad.get_position_orientation()[0] - thumb_pos))
        return apertures

    def _water_state(self) -> WaterState | None:
        if self._water_system is None:
            return None
        n_total = int(self._water_system.n_particles)
        n_source = int(self._pour_source.states[ContainedParticles].get_value(self._water_system).n_in_volume)
        n_dest = int(self._pour_dest.states[ContainedParticles].get_value(self._water_system).n_in_volume)
        return WaterState(n_total=n_total, n_in_source=n_source, n_in_dest=n_dest)

    # ------------------------------------------------------------------ targets

    def _target_state(self, hand_contacts, tracked_state, water) -> TargetState:
        kind = self.rubric.target["kind"]
        if kind == "region":
            return self._region_target(hand_contacts)
        if tracked_state is None:
            return TargetState()
        if kind == "shelf":
            return TargetState(
                at_target=bool(self._tracked.states[OnShelf].get_value(self._shelf_obj)),
                reached_target_height=self._reached_shelf_height(tracked_state),
            )
        if kind == "shelf_or_tin":
            at_target = bool(self._tracked.states[OnShelf].get_value(self._shelf_obj)) or self._in_tin(tracked_state)
            return TargetState(at_target=at_target, reached_target_height=self._reached_shelf_height(tracked_state))
        if kind == "tin":
            return TargetState(
                at_target=self._in_tin(tracked_state),
                reached_target_height=self._reached_tin_height(tracked_state),
            )
        if kind == "pegged":
            return TargetState(at_target=bool(self._tracked.states[Pegged].get_value(self._peg_obj)))
        if kind == "pour":
            transferred = False
            if water is not None and water.n_total > 0:
                emptied = water.n_in_source <= self.config.source_empty_tolerance_particles
                received = water.n_in_dest >= max(1, int(water.n_total * self.config.min_transfer_fraction))
                transferred = emptied and received
            supported = any(s.relation == "below" for s in tracked_state.supports)
            return TargetState(at_target=transferred and supported)
        raise ValueError(f"Unknown target kind {kind!r}")

    def _reached_shelf_height(self, tracked_state) -> bool:
        shelf_lower, shelf_upper = self._shelf_obj.root_link.aabb
        return tracked_state.aabb_lo[2] >= float(shelf_upper[2]) - TARGET_HEIGHT_SLACK_M

    def _reached_tin_height(self, tracked_state) -> bool:
        tin_lower, _tin_upper = self._tin_obj.root_link.aabb
        return tracked_state.aabb_lo[2] >= float(tin_lower[2]) - TARGET_HEIGHT_SLACK_M

    def _in_tin(self, tracked_state) -> bool:
        tin_lower, tin_upper = self._tin_obj.root_link.aabb
        center_x = (tracked_state.aabb_lo[0] + tracked_state.aabb_hi[0]) / 2.0
        center_y = (tracked_state.aabb_lo[1] + tracked_state.aabb_hi[1]) / 2.0
        tin_center_x = float(tin_lower[0] + tin_upper[0]) / 2.0
        tin_center_y = float(tin_lower[1] + tin_upper[1]) / 2.0
        tin_radius = min(float(tin_upper[0] - tin_lower[0]), float(tin_upper[1] - tin_lower[1])) / 2.0
        if math.hypot(center_x - tin_center_x, center_y - tin_center_y) > tin_radius * TIN_DISC_SHRINK:
            return False
        return (
            float(tin_lower[2]) - SUPPORT_VERTICAL_RANGE_M
            <= tracked_state.aabb_lo[2]
            <= float(tin_upper[2]) + SUPPORT_BELOW_SLACK_M
        )

    def _region_target(self, hand_contacts) -> TargetState:
        region_lower, region_upper = self._region_obj.states[AABB].get_value()
        aabb_lo = _to_tuple(region_lower)
        aabb_hi = _to_tuple(region_upper)

        palmar = False
        dorsal = False
        for contact in hand_contacts:
            position = contact.position
            if position is None:
                continue
            if not self._region.contains(position, aabb_lo, aabb_hi, MANNEQUIN_FRONT_DIR_XY):
                continue
            if contact.part in ("pad", "palm"):
                palmar = True
            else:
                dorsal = True

        progress = 0.0
        palm_link = self._robot.links.get(self._hand.palm_link)
        if palm_link is not None:
            palm_pos = _to_tuple(palm_link.get_position_orientation()[0])
            if self._palm_start_position is None:
                self._palm_start_position = palm_pos
            target_point = self._region.center(aabb_lo, aabb_hi, MANNEQUIN_FRONT_DIR_XY)
            d0 = math.dist(self._palm_start_position, target_point)
            progress = d0 - math.dist(palm_pos, target_point)

        return TargetState(
            at_target=palmar or dorsal,
            palmar_region_contact=palmar,
            dorsal_region_contact=dorsal,
            approach_progress_m=progress,
        )

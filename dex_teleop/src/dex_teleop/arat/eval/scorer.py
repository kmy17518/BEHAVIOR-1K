"""ARAT item scorer: milestone state machine + 0-3 score assignment.

Consumes one :class:`AratStepSnapshot` per engaged control step and produces an
:class:`ItemResult`. Scoring rules follow ``ARAT_scoring_guide.md``:

- 3: task completed in under 5 s with the item's appropriate hand-movement components.
- 2: task completed, but slowly (5-60 s) or with inappropriate hand movement, a spill,
     an inability to release, or the object escaping while release was attempted.
- 1: a voluntary grasp that holds and lifts the object was achieved within 60 s but the
     task was not completed (never for arm movement alone; pinch items additionally
     require the correct finger opposition).
- 0: no qualifying hand movement (no voluntary opening, dorsum pushing, or — for pinch —
     wrong finger opposition).

Best-performance semantics: drops are recorded but not penalized; the score reflects the
best qualifying attempt within the 60 s window.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field

from dex_teleop.arat.eval.detectors import (
    BracingTracker,
    ContactSummary,
    DorsumPushTracker,
    VoluntaryOpeningTracker,
    WaterTracker,
    is_env_supported,
    summarize_hand_contacts,
)
from dex_teleop.arat.eval.grasp_classifiers import classify_step
from dex_teleop.arat.eval.rubrics import ItemRubric, ScoringConfig
from dex_teleop.arat.eval.snapshot import AratStepSnapshot


@dataclass
class Event:
    t: float
    name: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"t": round(self.t, 3), "name": self.name, **({"detail": self.detail} if self.detail else {})}


@dataclass
class HoldWindow:
    """One continuous hand-only hold of the tracked object."""

    start_t: float
    end_t: float = 0.0
    steps: int = 0
    appropriate_steps: int = 0
    wrong_opposition_steps: int = 0
    reason_counts: Counter = field(default_factory=Counter)

    def record(self, appropriate: bool, reasons: tuple[str, ...]) -> None:
        self.steps += 1
        if appropriate:
            self.appropriate_steps += 1
        if "wrong_opposition" in reasons:
            self.wrong_opposition_steps += 1
        self.reason_counts.update(reasons)

    @property
    def appropriate_fraction(self) -> float:
        return self.appropriate_steps / self.steps if self.steps else 0.0

    @property
    def wrong_opposition_fraction(self) -> float:
        return self.wrong_opposition_steps / self.steps if self.steps else 0.0

    def reason_fraction(self, reason: str) -> float:
        return self.reason_counts.get(reason, 0) / self.steps if self.steps else 0.0

    def dominant_reasons(self, min_fraction: float = 0.25) -> tuple[str, ...]:
        if not self.steps:
            return ()
        return tuple(sorted(r for r, n in self.reason_counts.items() if n / self.steps >= min_fraction))

    def to_dict(self) -> dict:
        return {
            "start_t": round(self.start_t, 3),
            "end_t": round(self.end_t, 3),
            "steps": self.steps,
            "appropriate_fraction": round(self.appropriate_fraction, 3),
            "dominant_reasons": list(self.dominant_reasons()),
        }


@dataclass
class ItemResult:
    activity: str
    subscale: str
    score: int
    reasons: tuple[str, ...]
    completed: bool
    t_complete: float | None
    elapsed: float
    events: tuple[Event, ...]
    hold_windows: tuple[HoldWindow, ...]
    water: dict | None = None
    override: int | None = None  # a human reviewer may override the automated score

    def to_dict(self) -> dict:
        return {
            "activity": self.activity,
            "subscale": self.subscale,
            "score": self.score,
            "reasons": list(self.reasons),
            "completed": self.completed,
            "t_complete": None if self.t_complete is None else round(self.t_complete, 3),
            "elapsed": round(self.elapsed, 3),
            "events": [event.to_dict() for event in self.events],
            "hold_windows": [window.to_dict() for window in self.hold_windows],
            "water": self.water,
            "override": self.override,
        }


class AratItemScorer:
    """Scores one ARAT item from a stream of engaged-step snapshots."""

    def __init__(self, rubric: ItemRubric, config: ScoringConfig | None = None):
        self.rubric = rubric
        self.config = config or ScoringConfig()
        self.events: list[Event] = []
        self.hold_windows: list[HoldWindow] = []

        self._t = 0.0
        self._baseline_z: float | None = None
        self._is_region_item = rubric.target["kind"] == "region"
        self._is_pour_item = rubric.target["kind"] == "pour"

        required = None
        if rubric.object_size_m is not None:
            required = rubric.object_size_m * self.config.opening_factor
        self._opening = VoluntaryOpeningTracker(required, rubric.relevant_opening_fingers)
        self._dorsum = DorsumPushTracker()
        self._bracing = BracingTracker(dt=self.config.control_dt)
        self._water = WaterTracker() if self._is_pour_item else None

        self._active_hold: HoldWindow | None = None
        self._lift_elapsed = 0.0
        self._ever_held = False
        self._reached_target_height = False

        # Release confirmation
        self._release_started_at: float | None = None
        self._was_at_target_in_hand = False
        self._fumbled = False

        # Region (gross movement)
        self._region_contact_elapsed = 0.0
        self._region_palmar = False
        self._max_progress = 0.0

        self._completed = False
        self._t_complete: float | None = None
        self._finished = False

    # ------------------------------------------------------------------ stepping

    def step(self, snapshot: AratStepSnapshot) -> None:
        if self._finished:
            return
        self._t = snapshot.t

        if self._is_region_item:
            self._step_region(snapshot)
        else:
            self._step_manipulation(snapshot)

        if self._t >= self.config.time_limit_s and not self._finished:
            self._finish("time_limit")

    def _step_region(self, snapshot: AratStepSnapshot) -> None:
        target = snapshot.target
        self._max_progress = max(self._max_progress, target.approach_progress_m)
        if target.palmar_region_contact or target.dorsal_region_contact:
            self._region_contact_elapsed += self.config.control_dt
            self._region_palmar = self._region_palmar or target.palmar_region_contact
            if self._region_contact_elapsed >= self.config.region_contact_confirm_s and not self._completed:
                self._completed = True
                self._t_complete = self._t
                self._event("target_region_contact", palmar=self._region_palmar)
                self._finish("completed")
        else:
            self._region_contact_elapsed = 0.0

    def _step_manipulation(self, snapshot: AratStepSnapshot) -> None:
        tracked = snapshot.tracked
        if tracked is None:
            return
        if self._baseline_z is None:
            self._baseline_z = tracked.aabb_lo[2]

        contacts = summarize_hand_contacts(snapshot)
        self._opening.step(snapshot, contacts)
        if self._water is not None and snapshot.water is not None:
            self._water.step(snapshot.water)

        env_supported = is_env_supported(tracked)
        hand_only = contacts.any and not env_supported
        lifted = hand_only and tracked.aabb_lo[2] > self._baseline_z + self.config.lift_height_m
        at_target = snapshot.target.at_target

        # Release handling first, so a clean release closes the hold window before the
        # hold tracker would misread the lost contact as a drop
        self._update_release(snapshot, contacts)
        held_now = self._update_hold(lifted, contacts, at_target)

        self._dorsum.step(tracked, contacts, ever_held=self._ever_held)
        self._bracing.step(tracked, contacts, held_now=held_now)

        if snapshot.target.reached_target_height and not self._reached_target_height:
            self._reached_target_height = True
            self._event("reached_target_height")

    def _update_hold(self, lifted: bool, contacts: ContactSummary, at_target: bool) -> bool:
        if not lifted:
            if self._active_hold is not None:
                dropped = not self._completed and self._release_started_at is None and not at_target
                self._close_hold(dropped=dropped)
            self._lift_elapsed = 0.0
            return False

        self._lift_elapsed += self.config.control_dt
        if self._active_hold is None and self._lift_elapsed >= self.config.hold_min_s:
            self._active_hold = HoldWindow(start_t=self._t - self._lift_elapsed)
            if not self._ever_held:
                self._event("first_hold_and_lift")
            self._ever_held = True

        if self._active_hold is not None:
            assessment = classify_step(self.rubric.grasp, contacts)
            self._active_hold.record(assessment.appropriate, assessment.reasons)
            return True
        return False

    def _close_hold(self, dropped: bool) -> None:
        window = self._active_hold
        self._active_hold = None
        if window is None:
            return
        window.end_t = self._t
        self.hold_windows.append(window)
        if dropped:
            self._event("drop")

    def _update_release(self, snapshot: AratStepSnapshot, contacts: ContactSummary) -> None:
        at_target = snapshot.target.at_target
        tracked = snapshot.tracked

        if at_target and contacts.any:
            self._was_at_target_in_hand = True
            self._release_started_at = None
            return

        if at_target and not contacts.any:
            if self._release_started_at is None:
                self._release_started_at = self._t
                if self._active_hold is not None:
                    self._close_hold(dropped=False)
            settled = tracked.speed < self.config.settle_speed
            if settled and self._t - self._release_started_at >= self.config.release_confirm_s:
                if not self._completed:
                    self._completed = True
                    self._t_complete = self._release_started_at
                    self._event("release_completed")
                    self._finish("completed")
            return

        # Not at target anymore: if a release was in progress (or the object just left
        # the hand at the target), the object escaped during the release attempt
        if not self._completed and not self._fumbled:
            release_in_progress = self._release_started_at is not None
            just_left_hand_at_target = self._was_at_target_in_hand and not contacts.any
            if release_in_progress or just_left_hand_at_target:
                self._fumbled = True
                self._event("release_fumble")
        self._release_started_at = None
        self._was_at_target_in_hand = False

    # ------------------------------------------------------------------ lifecycle

    def _event(self, name: str, **detail) -> None:
        self.events.append(Event(t=self._t, name=name, detail=dict(detail)))

    def _finish(self, reason: str) -> None:
        if self._active_hold is not None:
            self._close_hold(dropped=False)
        self._finished = True
        self._event("item_finished", reason=reason)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def elapsed(self) -> float:
        return self._t

    def evaluation_breakdown(self, snapshot: AratStepSnapshot) -> dict:
        """Return a non-mutating, detailed explanation of the score at this step.

        The provisional score is the score the item would receive if it ended at
        this exact step. A deep copy is finalized so this diagnostic always uses
        the same decision path as the authoritative final score.
        """

        preview = deepcopy(self)
        result = preview.finalize()
        contacts = summarize_hand_contacts(snapshot)
        window = preview._scoring_window()
        unable_to_release = not preview._completed and not preview._fumbled and preview._was_at_target_in_hand
        completed_for_scoring = preview._completed or preview._fumbled or unable_to_release

        def condition(
            key: str,
            label: str,
            met: bool | None,
            detail: str,
            *,
            applicable: bool = True,
        ) -> dict:
            return {"key": key, "label": label, "met": met, "detail": detail, "applicable": applicable}

        conditions = []
        if self._is_region_item:
            conditions.extend(
                [
                    condition(
                        "movement_started",
                        "Partial movement toward target (score 1 when target is not reached)",
                        self._max_progress >= self.config.gross_start_progress_m,
                        f"maximum progress {self._max_progress:.3f} m; required {self.config.gross_start_progress_m:.3f} m",
                    ),
                    condition(
                        "target_contact_confirmed",
                        "Target-region contact confirmed",
                        self._completed,
                        f"contact held {self._region_contact_elapsed:.3f} s; required {self.config.region_contact_confirm_s:.3f} s",
                    ),
                    condition(
                        "palmar_contact",
                        "Contact used the palmar surface (score 3)",
                        self._region_palmar if self._completed else None,
                        "dorsal/side-only completion is capped at score 2",
                    ),
                    condition(
                        "completed_under_5s",
                        "Completion occurred before 5 seconds (score 3)",
                        self._t_complete < self.config.score3_time_s if self._t_complete is not None else None,
                        f"completion {self._t_complete:.3f} s" if self._t_complete is not None else "not completed",
                    ),
                ]
            )
        else:
            qualifying_hold = preview._qualifying_hold_for_score1()
            appropriate_fraction = None if window is None else window.appropriate_fraction
            wrong_opposition_fraction = None if window is None else window.wrong_opposition_fraction
            required_aperture = (
                None
                if self.rubric.object_size_m is None
                else self.rubric.object_size_m * self.config.opening_factor
            )
            conditions.extend(
                [
                    condition(
                        "voluntary_opening",
                        "Voluntary opening before contact",
                        self._opening.satisfied,
                        (
                            "not required for this item"
                            if required_aperture is None
                            else f"maximum aperture {self._opening.max_open_aperture:.3f} m; required {required_aperture:.3f} m"
                        ),
                    ),
                    condition(
                        "held_and_lifted",
                        "Object held and lifted",
                        self._ever_held,
                        f"lift must exceed {self.config.lift_height_m:.3f} m for {self.config.hold_min_s:.3f} s",
                    ),
                    condition(
                        "qualifying_score1_hold",
                        "Qualifying hand movement for score 1",
                        qualifying_hold,
                        "requires voluntary opening, hold/lift, and correct named-finger opposition for pinch items",
                    ),
                    condition(
                        "appropriate_hand_movement",
                        "Appropriate grasp over scoring hold (score 3)",
                        None
                        if appropriate_fraction is None
                        else appropriate_fraction >= self.config.grasp_ok_fraction,
                        (
                            "no scoring hold yet"
                            if appropriate_fraction is None
                            else f"appropriate {appropriate_fraction:.1%}; required {self.config.grasp_ok_fraction:.1%}"
                        ),
                    ),
                    condition(
                        "correct_pinch_opposition",
                        "Correct named-finger opposition",
                        (
                            None
                            if self.rubric.grasp["class"] != "pinch" or wrong_opposition_fraction is None
                            else wrong_opposition_fraction < 0.5
                        ),
                        (
                            "not a pinch item"
                            if self.rubric.grasp["class"] != "pinch"
                            else (
                                "no scoring hold yet"
                                if wrong_opposition_fraction is None
                                else f"wrong opposition on {wrong_opposition_fraction:.1%} of hold steps; score 0 at 50% or more"
                            )
                        ),
                        applicable=self.rubric.grasp["class"] == "pinch",
                    ),
                    condition(
                        "completion_or_release_flaw",
                        "Target reached and release attempted",
                        completed_for_scoring,
                        (
                            "clean completion"
                            if self._completed
                            else "release fumble"
                            if self._fumbled
                            else "object still held at target"
                            if unable_to_release
                            else "not completed"
                        ),
                    ),
                    condition(
                        "clean_release",
                        "Clean release and settle (score 3)",
                        self._completed if completed_for_scoring else None,
                        f"settle speed below {self.config.settle_speed:.3f} m/s for {self.config.release_confirm_s:.3f} s",
                    ),
                    condition(
                        "completed_under_5s",
                        "Completion occurred before 5 seconds (score 3)",
                        self._t_complete < self.config.score3_time_s if self._t_complete is not None else None,
                        f"completion {self._t_complete:.3f} s" if self._t_complete is not None else "not completed",
                    ),
                ]
            )
            if self.rubric.target["kind"] in ("shelf", "shelf_or_tin", "tin"):
                conditions.append(
                    condition(
                        "target_height",
                        "Object reached target height",
                        self._reached_target_height,
                        "required to demonstrate progress toward shelf/tin placement",
                    )
                )
            if self._water is not None:
                conditions.append(
                    condition(
                        "no_spill",
                        "No water spilled (score 3)",
                        self._water.spilled <= self.config.spill_tolerance_particles,
                        f"spilled {self._water.spilled}; allowed {self.config.spill_tolerance_particles}",
                    )
                )

        tracked = snapshot.tracked
        evidence = {
            "contact_pads": sorted(contacts.pads),
            "contact_dorsals": sorted(contacts.dorsals),
            "palm_contact": contacts.palm,
            "at_target": snapshot.target.at_target,
            "reached_target_height": snapshot.target.reached_target_height,
            "palmar_region_contact": snapshot.target.palmar_region_contact,
            "dorsal_region_contact": snapshot.target.dorsal_region_contact,
            "approach_progress_m": snapshot.target.approach_progress_m,
            "apertures_m": {key: round(value, 5) for key, value in snapshot.apertures.items()},
            "tracked_speed_m_s": None if tracked is None else tracked.speed,
            "environment_supports": []
            if tracked is None
            else [
                {"object": support.other_object, "link": support.other_link, "relation": support.relation}
                for support in tracked.supports
            ],
            "dorsum_push_detected": self._dorsum.detected,
            "braced_grasp_suspected": self._bracing.detected,
            "release_in_progress": self._release_started_at is not None,
            "release_fumbled": self._fumbled,
            "active_hold": self._active_hold is not None,
            "scoring_hold": None if window is None else window.to_dict(),
            "water": None
            if snapshot.water is None
            else {
                "total": snapshot.water.n_total,
                "in_source": snapshot.water.n_in_source,
                "in_destination": snapshot.water.n_in_dest,
                "outside": snapshot.water.n_outside,
            },
        }
        reason_text = ", ".join(result.reasons) if result.reasons else "all score-3 requirements met"
        return {
            "task_time_s": round(self._t, 6),
            "provisional_score": result.score,
            "score_reasons": list(result.reasons),
            "decision": f"{result.score}/3 if the item ended now: {reason_text}",
            "completed_for_scoring": result.completed,
            "finished": self._finished,
            "conditions": conditions,
            "evidence": evidence,
            "events": [event.to_dict() for event in self.events],
        }

    # ------------------------------------------------------------------ scoring

    def _scoring_window(self) -> HoldWindow | None:
        """The hold window that determines hand-movement quality: the last one."""
        return self.hold_windows[-1] if self.hold_windows else None

    def _qualifying_hold_for_score1(self) -> bool:
        """A voluntary grasp that held and lifted the object.

        Pinch items require the correct finger opposition ("correct hand movement
        components MUST be present to score more than 0"); using the correct finger
        without its pad still qualifies (the pad requirement caps at 2, not 0).
        """
        if not self._ever_held or not self._opening.satisfied:
            return False
        if self.rubric.grasp["class"] != "pinch":
            return True
        for window in self.hold_windows:
            if window.wrong_opposition_fraction < 0.5 and (
                window.appropriate_fraction >= 0.5 or window.reason_fraction("pads_not_used") >= 0.5
            ):
                return True
        return False

    def finalize(self) -> ItemResult:
        if not self._finished:
            self._finish("finalized")

        # "Unable to release": the item ended with the object held at its target
        unable_to_release = not self._completed and not self._fumbled and self._was_at_target_in_hand
        completed = self._completed or self._fumbled or unable_to_release

        reasons: list[str] = []
        water_summary = None
        if self._water is not None:
            water_summary = {
                "n_total": self._water.n_total,
                "n_in_source": self._water.n_in_source,
                "n_in_dest": self._water.n_in_dest,
                "spilled": self._water.spilled,
                "max_outside_seen": self._water.max_outside_seen,
            }

        if self._is_region_item:
            score = self._score_region(reasons)
        elif completed:
            score = self._score_completed(reasons, unable_to_release=unable_to_release)
        elif self._qualifying_hold_for_score1():
            score = 1
            reasons.append("held_and_lifted_only")
            if not self._reached_target_height and self.rubric.target["kind"] in ("shelf", "shelf_or_tin", "tin"):
                reasons.append("did_not_reach_target_height")
        elif self._t == 0.0:
            score = 0
            reasons.append("no_engaged_steps")
        else:
            score = 0
            reasons.extend(self._score0_reasons())

        return ItemResult(
            activity=self.rubric.activity,
            subscale=self.rubric.subscale,
            score=score,
            reasons=tuple(reasons),
            completed=completed,
            t_complete=self._t_complete if self._completed else None,
            elapsed=self._t,
            events=tuple(self.events),
            hold_windows=tuple(self.hold_windows),
            water=water_summary,
        )

    def _score_region(self, reasons: list[str]) -> int:
        if self._completed:
            if not self._region_palmar:
                reasons.append("dorsal_or_side_contact")
                return 2
            assert self._t_complete is not None
            if self._t_complete >= self.config.score3_time_s:
                reasons.append("exceeded_5s")
                return 2
            return 3
        if self._max_progress >= self.config.gross_start_progress_m:
            reasons.append("movement_started_target_not_reached")
            return 1
        reasons.append("no_movement_toward_target")
        return 0

    def _score_completed(self, reasons: list[str], unable_to_release: bool) -> int:
        window = self._scoring_window()

        # Pinch hard rule: wrong finger opposition scores 0 even if "completed"
        if self.rubric.grasp["class"] == "pinch" and window is not None and window.wrong_opposition_fraction >= 0.5:
            reasons.append("wrong_opposition")
            return 0

        flaws = []
        if window is None or window.appropriate_fraction < self.config.grasp_ok_fraction:
            flaws.append("inappropriate_hand_movement")
            if window is not None:
                flaws.extend(window.dominant_reasons())
        if self._fumbled and not self._completed:
            flaws.append("fell_before_release_completed")
        if unable_to_release:
            flaws.append("unable_to_release")
        if self._water is not None and self._water.spilled > self.config.spill_tolerance_particles:
            flaws.append("spilled_water")
        if self._completed and self._t_complete is not None and self._t_complete >= self.config.score3_time_s:
            flaws.append("exceeded_5s")

        if flaws:
            reasons.extend(dict.fromkeys(flaws))
            return 2
        return 3

    def _score0_reasons(self) -> list[str]:
        reasons = []
        if self._dorsum.detected:
            reasons.append("dorsum_push_only")
        if not self._opening.satisfied:
            reasons.append("no_voluntary_opening")
        if self.rubric.grasp["class"] == "pinch" and self._ever_held:
            reasons.append("wrong_opposition")
        if self._bracing.detected:
            reasons.append("braced_grasp_suspected")
        if not reasons:
            reasons.append("no_qualifying_hand_movement")
        return reasons

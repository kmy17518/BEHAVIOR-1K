"""Unit tests for the ARAT 0-3 scoring layer (no simulator required)."""

import pytest

from dex_teleop.arat import AratTaskCatalog
from dex_teleop.arat.eval import (
    AratItemScorer,
    AratSessionScorer,
    AratStepSnapshot,
    HandContact,
    MANNEQUIN_REGIONS,
    SUBSCALE_MAX_SCORES,
    SupportContact,
    TargetState,
    TrackedObjectState,
    WaterState,
    load_rubrics,
)
from dex_teleop.arat.eval.detectors import ContactSummary
from dex_teleop.arat.eval.grasp_classifiers import classify_step
from dex_teleop.arat.eval.hand_model import HandSemantics

DT = 1.0 / 30.0

OPEN_APERTURES = {"index": 0.12, "middle": 0.12, "ring": 0.11, "pinky": 0.10}
CLOSED_APERTURES = {"index": 0.02, "middle": 0.02, "ring": 0.02, "pinky": 0.02}


def contact(finger, part="pad"):
    return HandContact(finger=finger, part=part, link_name=f"right_{finger}_link")


def tracked_supported(z=0.954, speed=0.0):
    return TrackedObjectState(
        position=(0.0, 0.0, z + 0.05),
        aabb_lo=(-0.05, -0.05, z),
        aabb_hi=(0.05, 0.05, z + 0.1),
        speed=speed,
        supports=(SupportContact("arat_box", "front_cover_link", "below"),),
    )


def tracked_held(z=1.1, speed=0.2):
    return TrackedObjectState(
        position=(0.0, 0.0, z + 0.05),
        aabb_lo=(-0.05, -0.05, z),
        aabb_hi=(0.05, 0.05, z + 0.1),
        speed=speed,
        supports=(),
    )


def tracked_at_rest_on_target(z=1.2966, speed=0.01):
    return TrackedObjectState(
        position=(0.1, 0.0, z + 0.05),
        aabb_lo=(0.05, -0.05, z),
        aabb_hi=(0.15, 0.05, z + 0.1),
        speed=speed,
        supports=(SupportContact("arat_box", "base_link", "below"),),
    )


def drive(scorer, phases):
    """Run (duration_s, snapshot_builder(t)) phases through the scorer."""
    t = 0.0
    for duration, builder in phases:
        for _ in range(int(round(duration / DT))):
            t += DT
            scorer.step(builder(t))
    return t


def scorer_for(activity):
    return AratItemScorer(load_rubrics()[activity])


def block_phases(grasp_contacts, release_at_target=True, hold_duration=1.5):
    """A canonical block pick-and-place timeline."""
    return [
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES)),
        (0.5, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp_contacts, tracked=tracked_supported(), apertures=CLOSED_APERTURES)),
        (hold_duration, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp_contacts, tracked=tracked_held(), apertures=CLOSED_APERTURES)),
        (0.3, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp_contacts, tracked=tracked_at_rest_on_target(speed=0.05),
            apertures=CLOSED_APERTURES,
            target=TargetState(at_target=True, reached_target_height=True))),
        (0.8, lambda t: AratStepSnapshot(
            t=t, tracked=tracked_at_rest_on_target(),
            apertures=OPEN_APERTURES,
            target=TargetState(at_target=release_at_target, reached_target_height=True))),
    ]


GOOD_BLOCK_GRASP = (contact("thumb"), contact("index"), contact("middle"))


# ------------------------------------------------------------------ item scoring


def test_block_completed_fast_with_opposition_scores_3():
    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, block_phases(GOOD_BLOCK_GRASP))
    result = scorer.finalize()
    assert result.score == 3
    assert result.completed
    assert result.t_complete is not None and result.t_complete < 5.0


def test_block_completed_slowly_scores_2():
    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, block_phases(GOOD_BLOCK_GRASP, hold_duration=6.0))
    result = scorer.finalize()
    assert result.score == 2
    assert "exceeded_5s" in result.reasons


def test_block_completed_without_thumb_scores_2():
    no_thumb = (contact("index"), contact("middle"), contact("ring"))
    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, block_phases(no_thumb))
    result = scorer.finalize()
    assert result.score == 2
    assert "inappropriate_hand_movement" in result.reasons


def test_block_held_but_never_placed_scores_1():
    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, [
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES)),
        (3.0, lambda t: AratStepSnapshot(
            t=t, hand_contacts=GOOD_BLOCK_GRASP, tracked=tracked_held(z=1.0), apertures=CLOSED_APERTURES)),
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES)),
    ])
    result = scorer.finalize()
    assert result.score == 1
    assert "held_and_lifted_only" in result.reasons


def test_dorsum_push_scores_0():
    def pushing(t):
        x = 0.01 * t * 30
        tracked = TrackedObjectState(
            position=(x, 0.0, 1.0),
            aabb_lo=(x - 0.05, -0.05, 0.954),
            aabb_hi=(x + 0.05, 0.05, 1.054),
            speed=0.3,
            supports=(SupportContact("arat_box", "front_cover_link", "below"),),
        )
        return AratStepSnapshot(
            t=t, hand_contacts=(contact("index", part="dorsal"),), tracked=tracked, apertures=CLOSED_APERTURES,
        )

    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, [(3.0, pushing)])
    result = scorer.finalize()
    assert result.score == 0
    assert "dorsum_push_only" in result.reasons


def test_release_fumble_caps_at_2():
    fumble_phases = block_phases(GOOD_BLOCK_GRASP, release_at_target=False)
    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, fumble_phases + [
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES)),
    ])
    result = scorer.finalize()
    assert result.score == 2
    assert "fell_before_release_completed" in result.reasons
    assert any(event.name == "release_fumble" for event in result.events)


def test_unable_to_release_scores_2():
    scorer = scorer_for("arat_grip_alloy_tube_1cm")
    grasp = (contact("thumb"), contact("index"))
    drive(scorer, [
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES)),
        (2.0, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp, tracked=tracked_held(), apertures=CLOSED_APERTURES)),
        (58.0, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp, tracked=tracked_held(z=0.99, speed=0.01),
            apertures=CLOSED_APERTURES, target=TargetState(at_target=True))),
    ])
    result = scorer.finalize()
    assert result.score == 2
    assert "unable_to_release" in result.reasons


# ------------------------------------------------------------------ pinch rules


def pinch_phases(grasp_contacts):
    in_tin = TrackedObjectState(
        position=(0.09, -0.28, 1.30),
        aabb_lo=(0.085, -0.285, 1.297),
        aabb_hi=(0.095, -0.275, 1.313),
        speed=0.01,
        supports=(SupportContact("tin_lid_2", "base_link", "below"),),
    )
    return [
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES)),
        (1.5, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp_contacts, tracked=tracked_held(), apertures=CLOSED_APERTURES)),
        (0.3, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp_contacts, tracked=in_tin, apertures=CLOSED_APERTURES,
            target=TargetState(at_target=True, reached_target_height=True))),
        (0.8, lambda t: AratStepSnapshot(
            t=t, tracked=in_tin, apertures=OPEN_APERTURES,
            target=TargetState(at_target=True, reached_target_height=True))),
    ]


def test_pinch_correct_finger_scores_3():
    scorer = scorer_for("arat_pinch_marble_index")
    drive(scorer, pinch_phases((contact("thumb"), contact("index"))))
    assert scorer.finalize().score == 3


def test_pinch_wrong_finger_scores_0_even_if_completed():
    scorer = scorer_for("arat_pinch_marble_index")
    drive(scorer, pinch_phases((contact("thumb"), contact("middle"))))
    result = scorer.finalize()
    assert result.score == 0
    assert "wrong_opposition" in result.reasons


def test_pinch_correct_finger_without_pad_caps_at_2():
    scorer = scorer_for("arat_pinch_marble_index")
    drive(scorer, pinch_phases((contact("thumb"), contact("index", part="dorsal"))))
    result = scorer.finalize()
    assert result.score == 2
    assert "inappropriate_hand_movement" in result.reasons


def test_pinch_drop_with_correct_opposition_scores_1():
    scorer = scorer_for("arat_pinch_marble_index")
    drive(scorer, [
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES)),
        (1.5, lambda t: AratStepSnapshot(
            t=t, hand_contacts=(contact("thumb"), contact("index")), tracked=tracked_held(),
            apertures=CLOSED_APERTURES)),
        (1.0, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(speed=0.0), apertures=OPEN_APERTURES)),
    ])
    result = scorer.finalize()
    assert result.score == 1
    assert any(event.name == "drop" for event in result.events)


# ------------------------------------------------------------------ pour rules


def pour_phases(final_water, cylindrical=True):
    grasp = (
        (contact("thumb"), contact("index"), contact("middle"), contact("ring"))
        if cylindrical
        else (contact("thumb"), contact("index"))
    )
    mid_water = WaterState(n_total=41, n_in_source=41, n_in_dest=0)
    return [
        (0.5, lambda t: AratStepSnapshot(
            t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES, water=mid_water)),
        (1.5, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp, tracked=tracked_held(z=1.05), apertures=CLOSED_APERTURES, water=mid_water)),
        (0.5, lambda t: AratStepSnapshot(
            t=t, hand_contacts=grasp, tracked=tracked_supported(speed=0.02), apertures=CLOSED_APERTURES,
            water=final_water, target=TargetState(at_target=True))),
        (0.8, lambda t: AratStepSnapshot(
            t=t, tracked=tracked_supported(speed=0.0), apertures=OPEN_APERTURES,
            water=final_water, target=TargetState(at_target=True))),
    ]


def test_pour_clean_and_fast_scores_3():
    scorer = scorer_for("arat_grip_pour_water")
    drive(scorer, pour_phases(WaterState(n_total=41, n_in_source=0, n_in_dest=41)))
    assert scorer.finalize().score == 3


def test_pour_with_spill_scores_2():
    scorer = scorer_for("arat_grip_pour_water")
    drive(scorer, pour_phases(WaterState(n_total=41, n_in_source=0, n_in_dest=37)))
    result = scorer.finalize()
    assert result.score == 2
    assert "spilled_water" in result.reasons
    assert result.water["spilled"] == 4


# ------------------------------------------------------------------ gross movement


def gross_snapshot(t, palmar=False, dorsal=False, progress=0.0):
    return AratStepSnapshot(
        t=t,
        target=TargetState(
            at_target=palmar or dorsal,
            palmar_region_contact=palmar,
            dorsal_region_contact=dorsal,
            approach_progress_m=progress,
        ),
    )


def test_gross_palmar_contact_fast_scores_3():
    scorer = scorer_for("arat_gross_movement_hand_mouth")
    drive(scorer, [
        (1.0, lambda t: gross_snapshot(t, progress=0.1 * t)),
        (0.5, lambda t: gross_snapshot(t, palmar=True, progress=0.4)),
    ])
    assert scorer.finalize().score == 3


def test_gross_dorsal_contact_scores_2():
    scorer = scorer_for("arat_gross_movement_hand_top_head")
    drive(scorer, [
        (1.0, lambda t: gross_snapshot(t, progress=0.1 * t)),
        (0.5, lambda t: gross_snapshot(t, dorsal=True, progress=0.4)),
    ])
    result = scorer.finalize()
    assert result.score == 2
    assert "dorsal_or_side_contact" in result.reasons


def test_gross_movement_without_reaching_scores_1():
    scorer = scorer_for("arat_gross_movement_hand_behind_head")
    drive(scorer, [(2.0, lambda t: gross_snapshot(t, progress=0.3))])
    assert scorer.finalize().score == 1


def test_gross_movement_eight_centimeters_of_progress_scores_1():
    scorer = scorer_for("arat_gross_movement_hand_behind_head")
    drive(scorer, [(2.0, lambda t: gross_snapshot(t, progress=0.08))])
    assert scorer.finalize().score == 1


def test_gross_movement_below_eight_centimeters_scores_0():
    scorer = scorer_for("arat_gross_movement_hand_behind_head")
    drive(scorer, [(2.0, lambda t: gross_snapshot(t, progress=0.079))])
    assert scorer.finalize().score == 0


def test_gross_no_movement_scores_0():
    scorer = scorer_for("arat_gross_movement_hand_behind_head")
    drive(scorer, [(2.0, lambda t: gross_snapshot(t, progress=0.01))])
    assert scorer.finalize().score == 0


# ------------------------------------------------------------------ time limit


def test_time_limit_finishes_item():
    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, [(61.0, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=OPEN_APERTURES))])
    assert scorer.finished
    result = scorer.finalize()
    assert result.score == 0
    assert result.elapsed >= 60.0


# ------------------------------------------------------------------ session protocol


def _grasp_session():
    catalog = AratTaskCatalog()
    return AratSessionScorer({"grasp": catalog.subscales["grasp"]})


def test_session_first_item_max_short_circuits():
    catalog = AratTaskCatalog()
    session = _grasp_session()
    first, second = catalog.subscales["grasp"][:2]
    session.record(first, 3)
    assert not session.should_administer(second)
    assert session.subscale_score("grasp") == SUBSCALE_MAX_SCORES["grasp"]
    assert all(score == 3 for score in session.imputed_scores("grasp").values())


def test_session_second_item_zero_short_circuits():
    catalog = AratTaskCatalog()
    session = _grasp_session()
    first, second, third = catalog.subscales["grasp"][:3]
    session.record(first, 2)
    assert session.should_administer(second)
    session.record(second, 0)
    assert not session.should_administer(third)
    assert session.subscale_score("grasp") == 0
    assert session.to_dict()["subscales"]["grasp"]["raw_sum_of_administered"] == 2


def test_session_full_administration_sums_scores():
    catalog = AratTaskCatalog()
    session = _grasp_session()
    for activity in catalog.subscales["grasp"]:
        assert session.should_administer(activity)
        session.record(activity, 2)
    assert session.subscale_score("grasp") == 12


def test_gross_session_first_item_zero_short_circuits():
    catalog = AratTaskCatalog()
    session = AratSessionScorer({"gross_movement": catalog.subscales["gross_movement"]})
    first, second = catalog.subscales["gross_movement"][:2]
    session.record(first, 0)
    assert not session.should_administer(second)
    assert session.subscale_score("gross_movement") == 0


# ------------------------------------------------------------------ classifiers


def summary(pads=(), dorsals=(), palm=False):
    return ContactSummary(pads=frozenset(pads), dorsals=frozenset(dorsals), palm=palm)


def test_lateral_grip_requires_index_side():
    ok = classify_step({"class": "lateral"}, summary(pads=("thumb",), dorsals=("index",)))
    assert ok.appropriate
    pinched = classify_step({"class": "lateral"}, summary(pads=("thumb", "index")))
    assert not pinched.appropriate
    assert "pinch_instead_of_lateral" in pinched.reasons


def test_spherical_needs_thumb_and_two_finger_pads():
    assert classify_step({"class": "spherical"}, summary(pads=("thumb", "index", "middle"))).appropriate
    assert not classify_step({"class": "spherical"}, summary(pads=("thumb", "index"))).appropriate


def test_pincer_rejects_extra_fingers():
    assert classify_step({"class": "pincer"}, summary(pads=("thumb", "index", "middle"))).appropriate
    bad = classify_step({"class": "pincer"}, summary(pads=("thumb", "index", "ring")))
    assert not bad.appropriate


# ------------------------------------------------------------------ rubrics & regions


def test_rubrics_cover_all_catalog_activities():
    catalog = AratTaskCatalog()
    rubrics = load_rubrics()
    assert set(rubrics) == set(catalog.tasks)
    for activity, rubric in rubrics.items():
        task = catalog.tasks[activity]
        assert rubric.subscale == task.subscale
        if rubric.tracked_instance is not None:
            assert rubric.tracked_instance in task.instances
        for key in ("shelf_instance", "tin_instance", "peg_instance", "source_instance", "dest_instance",
                    "region_instance"):
            instance = rubric.target.get(key)
            if instance is not None:
                assert instance in task.instances, f"{activity}: {instance} missing from catalog map"


def test_pinch_rubrics_name_the_tested_finger():
    rubrics = load_rubrics()
    assert rubrics["arat_pinch_marble_index"].grasp["finger"] == "index"
    assert rubrics["arat_pinch_ball_bearing_ring"].grasp["finger"] == "ring"
    assert rubrics["arat_pinch_marble_middle"].grasp["finger"] == "middle"


def test_mannequin_regions_are_disjoint_where_it_matters():
    # Mannequin-like AABB: 1.32 m tall, centered at the origin
    lo, hi = (-0.1, -0.45, 0.0), (0.09, 0.45, 1.322)
    mouth_point = (-0.09, 0.0, 0.905 * 1.322)
    back_point = (0.08, 0.0, 0.91 * 1.322)
    top_point = (0.0, 0.0, 0.99 * 1.322)

    assert MANNEQUIN_REGIONS["mouth"].contains(mouth_point, lo, hi)
    assert not MANNEQUIN_REGIONS["mouth"].contains(back_point, lo, hi)
    assert MANNEQUIN_REGIONS["head_back"].contains(back_point, lo, hi)
    assert not MANNEQUIN_REGIONS["head_back"].contains(mouth_point, lo, hi)
    assert MANNEQUIN_REGIONS["head_top"].contains(top_point, lo, hi)
    # Neck-height contact on the back is below the head_back band
    neck_point = (0.08, 0.0, 0.80 * 1.322)
    assert not MANNEQUIN_REGIONS["head_back"].contains(neck_point, lo, hi)


def test_hand_semantics_classify_sharpa_links():
    hand = HandSemantics.sharpa("right")
    assert hand.classify_link("right_thumb_elastomer") == ("thumb", "pad")
    assert hand.classify_link("right_index_DP") == ("index", "dorsal")
    assert hand.classify_link("right_index_fingertip") == ("index", "dorsal")
    assert hand.classify_link("right_hand_C_MC") == ("palm", "palm")
    assert hand.classify_link("panda_link7") is None


def test_voluntary_opening_gate_blocks_score_1():
    scorer = scorer_for("arat_grasp_block_10cm")
    drive(scorer, [
        (0.5, lambda t: AratStepSnapshot(t=t, tracked=tracked_supported(), apertures=CLOSED_APERTURES)),
        (2.0, lambda t: AratStepSnapshot(
            t=t, hand_contacts=GOOD_BLOCK_GRASP, tracked=tracked_held(), apertures=CLOSED_APERTURES)),
    ])
    result = scorer.finalize()
    assert result.score == 0
    assert "no_voluntary_opening" in result.reasons


def test_scores_are_deterministic_for_the_same_timeline():
    results = []
    for _ in range(2):
        scorer = scorer_for("arat_grasp_block_10cm")
        drive(scorer, block_phases(GOOD_BLOCK_GRASP))
        results.append(scorer.finalize().to_dict())
    assert results[0] == results[1]


def test_evaluation_breakdown_explains_provisional_score_without_finishing_item():
    scorer = scorer_for("arat_gross_movement_hand_behind_head")
    snapshot = gross_snapshot(2.0, progress=0.3)
    scorer.step(snapshot)

    breakdown = scorer.evaluation_breakdown(snapshot)
    conditions = {condition["key"]: condition for condition in breakdown["conditions"]}

    assert breakdown["provisional_score"] == 1
    assert breakdown["score_reasons"] == ["movement_started_target_not_reached"]
    assert conditions["movement_started"]["met"] is True
    assert conditions["movement_started"]["applicable"] is True
    assert conditions["target_contact_confirmed"]["met"] is False
    assert scorer.finished is False


def test_completed_gross_item_keeps_partial_score1_rule_for_stable_overlay():
    scorer = scorer_for("arat_gross_movement_hand_mouth")
    snapshot = None
    for step in range(3):
        snapshot = gross_snapshot((step + 1) / 30.0, dorsal=True, progress=0.1)
        scorer.step(snapshot)

    breakdown = scorer.evaluation_breakdown(snapshot)
    conditions = {condition["key"]: condition for condition in breakdown["conditions"]}

    assert breakdown["provisional_score"] == 2
    assert breakdown["score_reasons"] == ["dorsal_or_side_contact"]
    assert conditions["movement_started"]["met"] is True
    assert conditions["movement_started"]["applicable"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

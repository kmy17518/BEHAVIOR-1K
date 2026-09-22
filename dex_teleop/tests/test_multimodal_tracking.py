import math
from dataclasses import replace

import numpy as np
import pytest

from dex_teleop.tracking import SourceUnavailableError
from dex_teleop.tracking.fusion import ArticulationFrameTransform, HandObservationFuser
from dex_teleop.tracking.hts import (
    HTSBufferOverflowError,
    HTSSource,
    _ClockAlignment,
    _D_TO_OPENXR,
    _quaternion_to_matrix,
)
from dex_teleop.tracking.multimodal import (
    DrainingMultimodalHandTrackingSource,
    MultimodalHandTrackingSource,
)
from dex_teleop.tracking.openxr import (
    articulation_to_mediapipe21,
    hand_frame_to_articulation,
    hand_frame_to_wrist,
    mediapipe21_to_openxr26,
    observation_to_hand_frame,
    openxr26_to_mediapipe21,
)
from dex_teleop.types import (
    FusedHandObservation,
    HandArticulationSample,
    Handedness,
    MEDIAPIPE_JOINT_NAMES,
    OPENXR_ANATOMICAL_WRIST_FRAME,
    OPENXR_HAND_JOINT_NAMES,
    WristPoseSample,
)


def _mediapipe_sample(
    timestamp=10.0,
    *,
    source="quest",
    handedness=Handedness.RIGHT,
    receipt_timestamp=None,
    coordinate_frame="wrist_local",
):
    positions = np.arange(63, dtype=np.float64).reshape(21, 3) / 100.0
    positions -= positions[0]
    orientations = {
        name: np.array([0.0, 0.0, 0.0, 2.0]) for name in MEDIAPIPE_JOINT_NAMES
    }
    return HandArticulationSample(
        timestamp=timestamp,
        receipt_timestamp=receipt_timestamp,
        source_timestamp_ns=round(timestamp * 1e9),
        source_frame_id=round(timestamp * 10),
        handedness=handedness,
        joint_positions=dict(zip(MEDIAPIPE_JOINT_NAMES, positions, strict=True)),
        joint_orientations_xyzw=orientations,
        source=source,
        schema="mediapipe21",
        coordinate_frame=coordinate_frame,
        confidence=0.9,
    )


def _wrist(
    timestamp,
    *,
    source="quest_wrist",
    position=(0.0, 0.0, 0.0),
    quaternion=(0.0, 0.0, 0.0, 1.0),
    receipt_timestamp=None,
    reference_frame="tracking",
    anatomical_frame="wrist_local",
):
    return WristPoseSample(
        timestamp=timestamp,
        receipt_timestamp=receipt_timestamp,
        source_timestamp_ns=round(timestamp * 2e9),
        source_frame_id=round(timestamp * 20),
        handedness=Handedness.RIGHT,
        position=np.asarray(position),
        quaternion_xyzw=np.asarray(quaternion),
        source=source,
        reference_frame=reference_frame,
        anatomical_frame=anatomical_frame,
        confidence=0.8,
    )


def test_rich_samples_copy_normalize_and_freeze_input_arrays():
    position = np.array([1.0, 2.0, 3.0])
    sample = HandArticulationSample(
        timestamp=1.0,
        handedness=Handedness.RIGHT,
        joint_positions={"wrist": position},
        joint_orientations_xyzw={"wrist": np.array([0.0, 0.0, 0.0, 2.0])},
        source="test",
    )
    position[:] = 9.0

    np.testing.assert_array_equal(sample.joint_positions["wrist"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        sample.joint_orientations_xyzw["wrist"], [0.0, 0.0, 0.0, 1.0]
    )
    with pytest.raises(TypeError):
        sample.joint_positions["new"] = np.zeros(3)
    with pytest.raises(ValueError):
        sample.joint_positions["wrist"][0] = 7.0

    wrist = _wrist(1.0, quaternion=(0.0, 0.0, 0.0, 4.0))
    np.testing.assert_array_equal(wrist.quaternion_xyzw, [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        wrist.position[0] = 2.0


def test_invalid_joint_may_retain_nan_but_valid_joint_may_not():
    sample = HandArticulationSample(
        timestamp=1.0,
        handedness="right",
        joint_positions={"missing": np.full(3, np.nan)},
        joint_validity={"missing": False},
        source="test",
    )
    assert not sample.validity()[0]
    assert np.isnan(sample.positions()).all()

    with pytest.raises(ValueError, match="non-finite valid position"):
        HandArticulationSample(
            timestamp=1.0,
            handedness="right",
            joint_positions={"bad": np.full(3, np.nan)},
            source="test",
        )


def test_mediapipe_openxr_round_trip_preserves_observed_joints_and_marks_synthetic_joints():
    mediapipe = _mediapipe_sample()

    openxr = mediapipe21_to_openxr26(mediapipe)

    assert openxr.schema == "openxr26"
    assert openxr.joint_names == OPENXR_HAND_JOINT_NAMES
    assert openxr.joint_validity["thumb_metacarpal"]
    assert not openxr.joint_validity["palm"]
    assert not openxr.joint_validity["index_metacarpal"]
    np.testing.assert_allclose(
        openxr.joint_positions["index_proximal"], mediapipe.joint_positions["index_mcp"]
    )

    round_trip = openxr26_to_mediapipe21(openxr)
    assert round_trip.joint_names == MEDIAPIPE_JOINT_NAMES
    np.testing.assert_allclose(round_trip.positions(), mediapipe.positions())
    np.testing.assert_array_equal(round_trip.validity(), np.ones(21, dtype=np.bool_))
    np.testing.assert_allclose(
        round_trip.orientations_xyzw(), mediapipe.orientations_xyzw()
    )


def test_openxr_conversion_rejects_missing_required_joint():
    openxr = mediapipe21_to_openxr26(_mediapipe_sample())
    incomplete = HandArticulationSample(
        timestamp=openxr.timestamp,
        handedness=openxr.handedness,
        joint_positions={
            name: position
            for name, position in openxr.joint_positions.items()
            if name != "little_tip"
        },
        source=openxr.source,
        schema="openxr26",
    )

    with pytest.raises(ValueError, match="little_tip"):
        articulation_to_mediapipe21(incomplete)


def test_fused_observation_builds_unchanged_world_frame_and_can_split_it_again():
    articulation = _mediapipe_sample(source="manus", receipt_timestamp=10.1)
    half_angle = math.pi / 4.0
    wrist = _wrist(
        10.0,
        source="quest",
        position=(1.0, 2.0, 3.0),
        quaternion=(0.0, 0.0, math.sin(half_angle), math.cos(half_angle)),
        receipt_timestamp=10.2,
    )
    observation = FusedHandObservation(articulation, wrist, 0.002)

    frame = observation_to_hand_frame(observation)

    expected_local = articulation.positions()
    expected_world = np.column_stack(
        (-expected_local[:, 1], expected_local[:, 0], expected_local[:, 2])
    )
    expected_world += wrist.position
    np.testing.assert_allclose(frame.mediapipe_landmarks(), expected_world)
    assert frame.source == "manus+quest"
    assert frame.receipt_timestamp == 10.2
    assert frame.confidence == 0.8

    split_articulation = hand_frame_to_articulation(frame)
    split_wrist = hand_frame_to_wrist(frame)
    np.testing.assert_allclose(split_articulation.positions(), articulation.positions())
    np.testing.assert_allclose(split_wrist.position, wrist.position)
    np.testing.assert_allclose(split_wrist.quaternion_xyzw, wrist.quaternion_xyzw)


def test_fused_observation_rejects_handedness_mismatch():
    left = _mediapipe_sample(handedness=Handedness.LEFT)
    with pytest.raises(ValueError, match="different handedness"):
        FusedHandObservation(left, _wrist(10.0))


def test_fuser_interpolates_translation_orientation_and_source_clock():
    fuser = HandObservationFuser(maximum_skew_seconds=0.1, wrist_source="vive")
    fuser.add_wrist(
        _wrist(
            9.95,
            source="vive",
            position=(0.0, 0.0, 0.0),
            receipt_timestamp=10.01,
        )
    )
    # Use the opposite quaternion hemisphere; SLERP must still take the short path.
    fuser.add_wrist(
        _wrist(
            10.05,
            source="vive",
            position=(2.0, 0.0, 0.0),
            quaternion=(0.0, 0.0, -1.0, 0.0),
            receipt_timestamp=10.10,
        )
    )

    observation = fuser.fuse(_mediapipe_sample(timestamp=10.0, source="manus"))

    assert observation is not None
    assert observation.synchronization_skew_seconds == pytest.approx(0.05)
    assert observation.wrist.timestamp == 10.0
    assert observation.wrist.receipt_timestamp == 10.10
    assert observation.wrist.source_frame_id is None
    assert observation.wrist.provenance == {
        "interpolated": True,
        "before_source_frame_id": 199,
        "after_source_frame_id": 201,
        "before_source_timestamp_ns": 19_900_000_000,
        "after_source_timestamp_ns": 20_100_000_000,
        "before_capture_monotonic_ns": 9_950_000_000,
        "after_capture_monotonic_ns": 10_050_000_000,
    }
    np.testing.assert_allclose(observation.wrist.position, [1.0, 0.0, 0.0])
    assert abs(observation.wrist.quaternion_xyzw[2]) == pytest.approx(math.sqrt(0.5))
    assert abs(observation.wrist.quaternion_xyzw[3]) == pytest.approx(math.sqrt(0.5))


def test_fuser_uses_nearest_in_bound_sample_and_prefers_earlier_on_tie():
    fuser = HandObservationFuser(maximum_skew_seconds=0.1, interpolate_wrist=False)
    before = _wrist(9.95, position=(1.0, 0.0, 0.0))
    after = _wrist(10.05, position=(2.0, 0.0, 0.0))
    fuser.add_wrist(after)
    fuser.add_wrist(before)

    observation = fuser.fuse(_mediapipe_sample(timestamp=10.0))

    assert observation is not None
    assert observation.wrist is before
    assert observation.synchronization_skew_seconds == pytest.approx(0.05)


def test_fuser_returns_none_instead_of_substituting_out_of_bound_wrist():
    fuser = HandObservationFuser(maximum_skew_seconds=0.01)
    fuser.add_wrist(_wrist(9.0))

    assert fuser.fuse(_mediapipe_sample(timestamp=10.0)) is None


def test_fuser_uses_strict_identity_for_declared_same_callback_pair():
    articulation = _mediapipe_sample(timestamp=10.0, source="manus:remote:1")
    wrist = replace(
        _wrist(10.0, source="manus:remote:1"),
        source_timestamp_ns=articulation.source_timestamp_ns,
        source_frame_id=articulation.source_frame_id,
    )
    fuser = HandObservationFuser(
        maximum_skew_seconds=0.0,
        articulation_source="manus:remote:1",
        wrist_source="manus:remote:1",
    )

    observation = fuser.fuse_paired(articulation, wrist)

    assert observation.articulation is articulation
    assert observation.wrist is wrist
    assert observation.synchronization_skew_seconds == 0.0
    assert fuser.buffered_wrist_count() == 0


def test_fuser_rejects_a_false_same_callback_claim():
    articulation = _mediapipe_sample(timestamp=10.0, source="manus")
    wrist = replace(
        _wrist(10.0, source="manus"),
        source_timestamp_ns=articulation.source_timestamp_ns,
        source_frame_id=articulation.source_frame_id + 1,
    )

    with pytest.raises(ValueError, match="frame IDs must be identical"):
        HandObservationFuser().fuse_paired(articulation, wrist)


def test_fuser_requires_explicit_selection_when_several_wrist_sources_are_buffered():
    fuser = HandObservationFuser(maximum_skew_seconds=0.1)
    fuser.add_wrist(_wrist(10.0, source="quest"))
    fuser.add_wrist(_wrist(10.0, source="vive"))
    articulation = _mediapipe_sample(timestamp=10.0)

    with pytest.raises(ValueError, match="Several wrist sources"):
        fuser.fuse(articulation)
    assert fuser.fuse(articulation, wrist_source="vive").wrist.source == "vive"


def test_fuser_rejects_source_and_reference_frame_changes_until_reset():
    fixed = HandObservationFuser(articulation_source="manus", wrist_source="vive")
    with pytest.raises(ValueError, match="expects wrist source"):
        fixed.add_wrist(_wrist(10.0, source="quest"))
    fixed.add_wrist(_wrist(10.0, source="vive", reference_frame="lighthouse"))
    with pytest.raises(ValueError, match="changed reference frame"):
        fixed.add_wrist(_wrist(10.1, source="vive", reference_frame="world"))
    with pytest.raises(ValueError, match="expects articulation source"):
        fixed.fuse(_mediapipe_sample(timestamp=10.0, source="quest"))

    fixed.reset()
    fixed.add_wrist(_wrist(10.1, source="vive", reference_frame="world"))
    assert fixed.buffered_wrist_count() == 1


def test_fuser_rejects_unrelated_anatomical_basis_without_transform():
    fuser = HandObservationFuser(maximum_skew_seconds=0.1)
    fuser.add_wrist(_wrist(10.0, anatomical_frame="quest_wrist"))

    with pytest.raises(ValueError, match="articulation-to-wrist transform"):
        fuser.fuse(_mediapipe_sample(timestamp=10.0, coordinate_frame="manus_wrist"))


def test_fuser_applies_explicit_articulation_to_wrist_transform():
    half_angle = math.pi / 4.0
    transform = ArticulationFrameTransform(
        source_frame="manus_wrist",
        target_frame="quest_wrist",
        translation=np.array([1.0, 2.0, 3.0]),
        quaternion_xyzw=np.array(
            [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
        ),
    )
    fuser = HandObservationFuser(
        maximum_skew_seconds=0.1,
        articulation_to_wrist=transform,
    )
    fuser.add_wrist(_wrist(10.0, anatomical_frame="quest_wrist"))
    articulation = _mediapipe_sample(timestamp=10.0, coordinate_frame="manus_wrist")

    observation = fuser.fuse(articulation)

    assert observation is not None
    assert observation.articulation.coordinate_frame == "quest_wrist"
    first = articulation.joint_positions["thumb_cmc"]
    expected = np.array([-first[1], first[0], first[2]]) + transform.translation
    np.testing.assert_allclose(
        observation.articulation.joint_positions["thumb_cmc"], expected
    )
    np.testing.assert_allclose(
        observation.articulation.joint_orientations_xyzw["wrist"],
        transform.quaternion_xyzw,
    )


def test_fuser_bounds_buffer_and_replaces_duplicate_capture():
    fuser = HandObservationFuser(maximum_buffered_samples=2)
    fuser.add_wrist(_wrist(3.0, receipt_timestamp=3.1))
    fuser.add_wrist(_wrist(1.0, receipt_timestamp=1.1))
    fuser.add_wrist(_wrist(2.0, receipt_timestamp=2.1))
    assert fuser.buffered_wrist_count() == 2

    replacement = _wrist(2.0, position=(7.0, 0.0, 0.0), receipt_timestamp=2.2)
    fuser.add_wrist(replacement)
    assert fuser.buffered_wrist_count() == 2
    assert fuser.fuse(_mediapipe_sample(timestamp=2.0)).wrist is replacement


def test_hts_one_receiver_exposes_legacy_and_both_rich_views_with_parity():
    source = HTSSource()
    landmarks = np.arange(63, dtype=np.float64) / 1000.0
    payload = (
        "Right wrist | f = 7 | t = 5000000000:, 1, 2, 3, 0, 0, 0, 1\n"
        "Right landmarks | f = 7 | t = 5000000000:, "
        + ", ".join(str(value) for value in landmarks)
    )
    source._handle_payload(payload, receipt_timestamp=100.0)

    legacy = source.read(Handedness.RIGHT)
    articulation = source.read_articulation(Handedness.RIGHT)
    wrist = source.read_wrist(Handedness.RIGHT)

    assert isinstance(source, MultimodalHandTrackingSource)
    assert legacy is not None and articulation is not None and wrist is not None
    assert articulation.schema == "mediapipe21"
    assert articulation.coordinate_frame == OPENXR_ANATOMICAL_WRIST_FRAME
    assert wrist.anatomical_frame == OPENXR_ANATOMICAL_WRIST_FRAME
    assert (
        articulation.source_frame_id
        == wrist.source_frame_id
        == legacy.source_frame_id
        == 7
    )
    reconstructed = observation_to_hand_frame(FusedHandObservation(articulation, wrist))
    np.testing.assert_allclose(
        reconstructed.mediapipe_landmarks(), legacy.mediapipe_landmarks()
    )
    np.testing.assert_allclose(reconstructed.wrist_position, legacy.wrist_position)
    # Rich wrist orientation maps the canonical OpenXR local basis into the
    # unchanged legacy tracking/world basis.  Converting it back to the legacy
    # anatomical basis recovers the legacy wrist rotation exactly.
    np.testing.assert_allclose(
        _quaternion_to_matrix(reconstructed.wrist_quaternion_xyzw) @ _D_TO_OPENXR,
        _quaternion_to_matrix(legacy.wrist_quaternion_xyzw),
        atol=1e-12,
    )
    assert reconstructed.timestamp == legacy.timestamp
    assert reconstructed.receipt_timestamp == legacy.receipt_timestamp


def _hts_payload(frame_id: int, source_timestamp_ns: int) -> str:
    landmarks = np.arange(63, dtype=np.float64) / 1000.0
    return (
        f"Right wrist | f = {frame_id} | t = {source_timestamp_ns}:, "
        "1, 2, 3, 0, 0, 0, 1\n"
        f"Right landmarks | f = {frame_id} | t = {source_timestamp_ns}:, "
        + ", ".join(str(value) for value in landmarks)
    )


def test_hts_atomic_drain_preserves_every_pair_and_legacy_reads_do_not_activate_it():
    legacy_source = HTSSource(maximum_buffered_samples=1)
    for frame_id in range(1, 4):
        legacy_source._handle_payload(
            _hts_payload(frame_id, frame_id * 1_000_000),
            receipt_timestamp=100.0 + frame_id * 0.01,
        )
    assert legacy_source.read(Handedness.RIGHT).source_frame_id == 3
    legacy_source.check_health()

    source = HTSSource(maximum_buffered_samples=3)
    assert isinstance(source, DrainingMultimodalHandTrackingSource)
    assert source.drain_hand_tracking(Handedness.RIGHT).articulations == ()
    for frame_id in range(1, 4):
        source._handle_payload(
            _hts_payload(frame_id, frame_id * 1_000_000),
            receipt_timestamp=200.0 + frame_id * 0.01,
        )

    batch = source.drain_hand_tracking(Handedness.RIGHT)
    assert [sample.source_frame_id for sample in batch.articulations] == [1, 2, 3]
    assert [sample.source_frame_id for sample in batch.wrists] == [1, 2, 3]
    assert source.drain_hand_tracking(Handedness.RIGHT).articulations == ()


def test_hts_lossless_drain_overflow_is_an_explicit_health_failure():
    source = HTSSource(maximum_buffered_samples=1)
    source.drain_hand_tracking(Handedness.RIGHT)
    source._handle_payload(_hts_payload(1, 1_000_000), receipt_timestamp=300.01)
    with pytest.raises(HTSBufferOverflowError, match="exceeded 1 unread frames"):
        source._handle_payload(_hts_payload(2, 2_000_000), receipt_timestamp=300.02)
    with pytest.raises(SourceUnavailableError, match="native sample buffer"):
        source.check_health()


def test_hts_clock_alignment_tracks_drift_without_chasing_arrival_jitter():
    rng = np.random.default_rng(7)
    alignment = _ClockAlignment()
    true_elapsed = np.arange(6000, dtype=np.float64) * 0.02
    drift = 350e-6
    source_origin_ns = 9_000_000_000_000
    source_timestamps = source_origin_ns + np.rint(
        true_elapsed * (1.0 + drift) * 1e9
    ).astype(np.int64)
    jitter = 0.006 + rng.uniform(0.0, 0.004, len(true_elapsed))
    jitter[::173] += 0.010
    receipts = 1000.0 + true_elapsed + jitter

    mapped = np.asarray(
        [
            alignment.to_desktop(int(source_timestamp), float(receipt))
            for source_timestamp, receipt in zip(
                source_timestamps, receipts, strict=True
            )
        ]
    )

    assert alignment.rate == pytest.approx(1.0 / (1.0 + drift), abs=5e-5)
    assert np.all(mapped <= receipts)
    assert np.all(np.diff(mapped) > 0.0)
    mapped_window = mapped[-1] - mapped[-1001]
    true_window = true_elapsed[-1] - true_elapsed[-1001]
    assert mapped_window == pytest.approx(true_window, abs=0.002)


def test_hts_stream_clamps_an_affine_refit_without_reversing_capture_time():
    source = HTSSource(maximum_buffered_samples=2)
    source.drain_hand_tracking(Handedness.RIGHT)
    source._handle_payload(_hts_payload(1, 1_000_000_000), receipt_timestamp=500.0)
    # Force the same adverse offset step that a noisy online refit could
    # produce; stream publication must remain strictly ordered.
    source._states[Handedness.RIGHT].clock_alignment.desktop_offset -= 1.0
    source._handle_payload(_hts_payload(2, 1_010_000_000), receipt_timestamp=500.02)
    timestamps = [
        sample.timestamp
        for sample in source.drain_hand_tracking(Handedness.RIGHT).articulations
    ]
    assert timestamps[1] > timestamps[0]
    assert timestamps[1] <= 500.02


def test_hts_bimanual_streams_accept_the_same_headset_timestamp():
    source = HTSSource(maximum_buffered_samples=1)
    source.drain_hand_tracking(Handedness.LEFT)
    source.drain_hand_tracking(Handedness.RIGHT)
    source._handle_payload(_hts_payload(1, 2_000_000_000), receipt_timestamp=600.0)
    source._handle_payload(
        _hts_payload(1, 2_000_000_000).replace("Right", "Left"),
        receipt_timestamp=600.0,
    )

    right = source.drain_hand_tracking(Handedness.RIGHT).articulations
    left = source.drain_hand_tracking(Handedness.LEFT).articulations
    assert len(right) == len(left) == 1
    assert right[0].source_timestamp_ns == left[0].source_timestamp_ns
    assert right[0].timestamp == left[0].timestamp

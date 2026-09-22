from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

from dex_teleop.omnigibson.hand_tracking_recording import _write_wrist_stream
from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.retargeting import LandmarkRetargeter
from dex_teleop.tracking.manus import (
    MANUS_ARTICULATION_SCHEMA,
    MANUS_JOINT_NAMES,
    ManusIntegratedSource,
    _native_bridge_directory,
    build_manus_bridge,
    discover_manus_sdk,
)
from dex_teleop.tracking.multimodal import HandArticulationSource
from dex_teleop.tracking.manus_calibration import (
    ManusCoreCalibration,
    ManusWristCalibration,
)
from dex_teleop.tracking.openxr import (
    articulation_to_mediapipe21,
    observation_to_hand_frame,
)
from dex_teleop.types import (
    FusedHandObservation,
    Handedness,
    MEDIAPIPE_JOINT_NAMES,
    OPENXR_ANATOMICAL_WRIST_FRAME,
    WristPoseSample,
)
from dex_teleop.tracking.transforms import RigidTransform, compose_transforms


FAKE_BRIDGE = Path(__file__).with_name("fake_manus_bridge.py")
STRICT_V1_BRIDGE = Path(__file__).with_name("fake_manus_v1_bridge.py")


def _source(mode="normal", side="right", *, bridge_mode="integrated", **kwargs):
    startup_timeout = kwargs.pop("startup_timeout", 2.0)
    required_handedness = kwargs.pop("required_handedness", side)
    kwargs.setdefault("mode", bridge_mode)
    return ManusIntegratedSource(
        bridge_command=(
            sys.executable,
            str(FAKE_BRIDGE),
            "--scenario",
            mode,
            "--mode",
            bridge_mode,
            "--side",
            side,
        ),
        startup_timeout=startup_timeout,
        required_handedness=required_handedness,
        **kwargs,
    )


def _core_calibration(
    *,
    tracker_id: str = "ultimate-right",
    wrist_transform: RigidTransform | None = None,
) -> ManusCoreCalibration:
    return ManusCoreCalibration(
        reference_frame="robot_world",
        reference_from_core_world=RigidTransform(
            np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        ),
        wrists=(
            ManusWristCalibration(
                handedness=Handedness.RIGHT,
                tracker_id=tracker_id,
                skeleton_wrist_to_anatomical_wrist=(
                    wrist_transform
                    if wrist_transform is not None
                    else RigidTransform(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
                ),
            ),
        ),
    )


def test_manus_fake_bridge_publishes_rich_wrist_local_articulation():
    source = _source()
    assert isinstance(source, HandArticulationSource)
    try:
        source.start()
        source.check_health()
        sample = source.read_articulation(Handedness.RIGHT)
        assert sample is not None
        assert sample.schema == MANUS_ARTICULATION_SCHEMA
        assert sample.coordinate_frame == OPENXR_ANATOMICAL_WRIST_FRAME
        assert sample.joint_names == MANUS_JOINT_NAMES
        assert "thumb_distal" in sample.joint_positions
        assert "thumb_intermediate" not in sample.joint_positions
        assert "palm" not in sample.joint_positions
        assert np.allclose(sample.joint_positions["wrist"], 0.0, atol=1e-12)
        assert np.allclose(
            sample.joint_positions["index_metacarpal"],
            [0.005, 0.010, -0.005],
            atol=1e-9,
        )
        assert np.allclose(
            sample.joint_orientations_xyzw["little_tip"], [0.0, 0.0, 0.0, 1.0]
        )
        assert sample.timestamp == pytest.approx(12.5)
        assert sample.source_timestamp_ns == 12_500_000_000
        assert sample.source_frame_id == 7
        assert sample.provenance["glove_id"] == 1417281806
        assert sample.provenance["manus_publish_time"] == 987654321
        assert sample.provenance["callback_capture_monotonic_ns"] == 12_500_000_000
        assert sample.provenance["mode"] == "integrated"
        assert sample.provenance["core_host_name"] == "fake-integrated"
        assert sample.provenance["core_version"] == "3.1.1"
        assert sample.provenance["connection_generation"] == 1

        diagnostics = source.frame_diagnostics(Handedness.RIGHT)
        assert diagnostics is not None
        assert diagnostics.glove_id == 1417281806
        assert diagnostics.manus_publish_time == 987654321
        assert diagnostics.capture_monotonic_ns == 12_500_000_000
        assert diagnostics.connection_generation == 1
        assert source.recording_metadata()["loopback_only"] is True

        drained = source.drain_articulations(Handedness.RIGHT)
        assert tuple(item.source_frame_id for item in drained) == (7,)
        assert source.drain_articulations(Handedness.RIGHT) == ()
        assert source.read_articulation(Handedness.RIGHT) is sample
    finally:
        source.close()


def test_manus_remote_coemits_exact_callback_wrist_and_articulation():
    source = _source(
        bridge_mode="remote",
        core_host="192.0.2.10",
        hand_motion="tracker",
        tracker_diagnostics=True,
    )
    try:
        source.start()
        batch = source.drain_hand_tracking(Handedness.RIGHT)

        assert len(batch.articulations) == len(batch.wrists) == 1
        articulation = batch.articulations[0]
        wrist = batch.wrists[0]
        assert articulation.source == wrist.source == "manus_remote"
        assert articulation.timestamp == wrist.timestamp == pytest.approx(12.5)
        assert (
            articulation.source_timestamp_ns
            == wrist.source_timestamp_ns
            == 12_500_000_000
        )
        assert articulation.source_frame_id == wrist.source_frame_id == 7
        assert wrist.reference_frame == "manus_core_world_y_up_rh_z_to_viewer_m"
        np.testing.assert_allclose(wrist.position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            wrist.quaternion_xyzw,
            [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        )

        deadline = time.monotonic() + 1.0
        while source.tracker_diagnostics() is None and time.monotonic() < deadline:
            time.sleep(0.005)
        tracker = source.tracker_diagnostics()
        assert tracker is not None
        assert tracker.trackers[0]["id"] == "ultimate-right"
        assert tracker.trackers[0]["quality"] == "trackable"
        assert tracker.connection_generation == 1

        metadata = source.recording_metadata()
        assert metadata["provider"] == "MANUS SDK Remote"
        assert metadata["mode"] == "remote"
        assert metadata["loopback_only"] is False
        assert metadata["core_host_name"] == "fake-windows-core"
        assert metadata["core_host_ip"] == "192.0.2.10"
        assert metadata["hand_motion"] == "tracker"
        assert metadata["discovered_hosts"][0]["ip"] == "192.0.2.10"
        assert metadata["landscape"]["license_sdk"] is True
    finally:
        source.close()


def test_manus_remote_wrist_requires_exact_healthy_advancing_ultimate():
    source = _source(
        bridge_mode="remote",
        hand_motion="tracker",
        core_calibration=_core_calibration(),
        require_tracker_for_wrist=True,
        expected_tracker_id="ultimate-right",
        expected_tracker_user_id=1,
        tracker_stale_timeout=0.5,
    )
    try:
        source.start()
        wrist = source.read_wrist(Handedness.RIGHT)
        assert wrist is not None
        assert wrist.confidence == 1.0
        assert wrist.provenance["tracker_health"] == "healthy"
        assert wrist.provenance["validated_tracker_id"] == "ultimate-right"
        source.check_health()
    finally:
        source.close()


def test_manus_remote_wrist_rejects_wrong_tracker_identity():
    source = _source(
        bridge_mode="remote",
        hand_motion="tracker",
        core_calibration=_core_calibration(tracker_id="different-ultimate"),
        require_tracker_for_wrist=True,
        expected_tracker_id="different-ultimate",
        expected_tracker_user_id=1,
        startup_timeout=0.15,
    )
    with pytest.raises(SourceUnavailableError, match="different-ultimate.*absent"):
        source.start()
    assert source.recording_metadata()["tracker_healthy"] is False
    assert "absent" in source.recording_metadata()["tracker_health_reason"]
    source.close()


@pytest.mark.parametrize(
    "missing",
    ("expected_tracker_id", "expected_tracker_user_id"),
)
def test_manus_remote_wrist_requires_complete_runtime_identity_policy(missing):
    policy = {
        "expected_tracker_id": "ultimate-right",
        "expected_tracker_user_id": 1,
    }
    policy[missing] = None
    with pytest.raises(ValueError, match=f"explicit {missing}"):
        _source(
            bridge_mode="remote",
            hand_motion="tracker",
            core_calibration=_core_calibration(),
            require_tracker_for_wrist=True,
            **policy,
        )


def test_manus_remote_wrist_rejects_wrong_core_user_assignment():
    source = _source(
        bridge_mode="remote",
        hand_motion="tracker",
        core_calibration=_core_calibration(),
        require_tracker_for_wrist=True,
        expected_tracker_id="ultimate-right",
        expected_tracker_user_id=2,
        startup_timeout=0.15,
    )
    with pytest.raises(SourceUnavailableError, match="tracker user 1 != 2"):
        source.start()
    source.close()


def test_manus_remote_wrist_fails_health_on_quality_degradation():
    source = _source(
        "tracker_degrades",
        bridge_mode="remote",
        hand_motion="tracker",
        core_calibration=_core_calibration(),
        require_tracker_for_wrist=True,
        expected_tracker_id="ultimate-right",
        expected_tracker_user_id=1,
        tracker_stale_timeout=0.5,
    )
    try:
        source.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                source.check_health()
            except SourceUnavailableError as error:
                assert "untrackable" in str(error)
                break
            time.sleep(0.01)
        else:
            pytest.fail("MANUS wrist did not reject degraded tracker quality")
        assert source.read_wrist(Handedness.RIGHT) is None
    finally:
        source.close()


def test_manus_remote_wrist_fails_health_when_tracker_update_stalls():
    source = _source(
        bridge_mode="remote",
        hand_motion="tracker",
        core_calibration=_core_calibration(),
        require_tracker_for_wrist=True,
        expected_tracker_id="ultimate-right",
        expected_tracker_user_id=1,
        tracker_stale_timeout=0.05,
    )
    try:
        source.start()
        time.sleep(0.08)
        with pytest.raises(SourceUnavailableError, match="update is stale"):
            source.check_health()
        assert source.read_wrist(Handedness.RIGHT) is None
    finally:
        source.close()


def test_manus_legacy_articulation_message_still_produces_a_paired_wrist():
    with _source("legacy_protocol") as source:
        batch = source.drain_hand_tracking(Handedness.RIGHT)

    assert len(batch.articulations) == len(batch.wrists) == 1
    assert (
        batch.articulations[0].source_frame_id == batch.wrists[0].source_frame_id == 7
    )
    assert batch.articulations[0].timestamp == batch.wrists[0].timestamp
    assert batch.wrists[0].provenance["wrist_node_id"] == 0


def test_actual_v1_cli_omits_v2_flags_and_records_negotiated_version(tmp_path):
    source = ManusIntegratedSource(
        bridge_command=(sys.executable, str(STRICT_V1_BRIDGE)),
        startup_timeout=2.0,
        required_handedness=Handedness.RIGHT,
    )
    with source:
        batch = source.drain_hand_tracking(Handedness.RIGHT)
        metadata = source.recording_metadata()

    assert metadata["bridge_protocol_version"] == 1
    assert metadata["supported_protocol_version"] == 2
    assert metadata["protocol_version_history"] == [1]
    assert metadata["protocol_version_changes_allowed"] is False
    assert metadata["bridge_capabilities"] == [
        "integrated_wire_v1",
        "python_required_hand_filter",
        "same_callback_pairing",
    ]
    assert "--required-hand" not in metadata["resolved_bridge_command"]
    articulation = batch.articulations[0]
    wrist = batch.wrists[0]
    assert articulation.provenance["bridge_protocol_version"] == 1
    assert wrist.provenance["bridge_protocol_version"] == 1
    assert articulation.provenance["supported_protocol_version"] == 2
    assert wrist.provenance["supported_protocol_version"] == 2
    assert json.loads(articulation.provenance["protocol_version_history"]) == [1]
    assert json.loads(wrist.provenance["protocol_version_history"]) == [1]
    assert (
        articulation.provenance["protocol_version_history"]
        == wrist.provenance["protocol_version_history"]
    )

    result = subprocess.run(
        (
            sys.executable,
            str(STRICT_V1_BRIDGE),
            "--required-hand",
            "right",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --required-hand right" in result.stderr

    recording = tmp_path / "legacy-protocol.hdf5"
    with h5py.File(recording, "w") as stream:
        group = stream.create_group("wrist_streams")
        _write_wrist_stream(group, "manus_legacy", (wrist,), {"manus_legacy": metadata})
    with h5py.File(recording, "r") as stream:
        stored = stream["wrist_streams/manus_legacy"]
        assert json.loads(stored.attrs["metadata"])["bridge_protocol_version"] == 1
        provenance = json.loads(stored["provenance_json"][0])
        assert provenance["bridge_protocol_version"] == 1
        assert provenance["supported_protocol_version"] == 2
        assert json.loads(provenance["protocol_version_history"]) == [1]


def test_manus_remote_applies_core_world_and_anatomical_wrist_calibration(tmp_path):
    calibration = tmp_path / "manus-core-calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "core_world_frame": "manus_core_world_y_up_rh_z_to_viewer_m",
                "reference_frame": "robot_world",
                "reference_from_core_world": {
                    "translation": [10.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "wrists": [
                    {
                        "handedness": "right",
                        "source_pose": "core_baked_skeleton_wrist",
                        "core_tracker_offset_applied": True,
                        "tracker_id": "ultimate-right",
                        "tracker_offset_preset": "MANUS Ultimate right wrist",
                        "tracker_to_anatomical_wrist": None,
                        "skeleton_wrist_to_anatomical_wrist": {
                            "translation": [0.0, 1.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with _source(
        bridge_mode="remote",
        hand_motion="tracker",
        core_calibration=calibration,
    ) as source:
        batch = source.drain_hand_tracking(Handedness.RIGHT)
        wrist = batch.wrists[0]

        assert wrist.reference_frame == "robot_world"
        np.testing.assert_allclose(wrist.position, [10.0, 2.0, 3.0], atol=1e-12)
        assert wrist.provenance["calibration_schema_version"] == 1
        assert len(wrist.provenance["calibration_sha256"]) == 64
        assert wrist.provenance["wrist_pose_semantics"] == "core_baked_skeleton_wrist"


def test_manus_nonidentity_wrist_frame_preserves_every_global_joint():
    root_orientation = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    residual = RigidTransform(
        np.array([0.03, -0.04, 0.05]),
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    )
    with _source(
        bridge_mode="remote",
        hand_motion="tracker",
        core_calibration=_core_calibration(wrist_transform=residual),
    ) as source:
        batch = source.drain_hand_tracking(Handedness.RIGHT)

    articulation = batch.articulations[0]
    wrist = batch.wrists[0]
    reference_from_anatomical = RigidTransform(wrist.position, wrist.quaternion_xyzw)
    for node_id, name in enumerate(MANUS_JOINT_NAMES):
        reconstructed = compose_transforms(
            reference_from_anatomical,
            RigidTransform(
                articulation.joint_positions[name],
                articulation.joint_orientations_xyzw[name],
            ),
        )
        local = np.array([node_id * 0.001, node_id * 0.002, -node_id * 0.001])
        expected_position = np.array([1.0 - local[1], 2.0 + local[0], 3.0 + local[2]])
        np.testing.assert_allclose(
            reconstructed.translation, expected_position, atol=1e-9
        )
        assert abs(float(np.dot(reconstructed.quaternion_xyzw, root_orientation))) == (
            pytest.approx(1.0, abs=1e-9)
        )


def test_manus_combined_drain_is_lossless_and_cannot_mix_with_split_drains():
    source = _source("overflow", max_pending_samples=4)
    try:
        source.start()
        first = source.drain_hand_tracking(Handedness.RIGHT)
        assert [sample.source_frame_id for sample in first.articulations] == [7]
        time.sleep(0.2)
        source.check_health()
        second = source.drain_hand_tracking(Handedness.RIGHT)

        assert [sample.source_frame_id for sample in second.articulations] == [8, 9, 10]
        assert [sample.source_frame_id for sample in second.wrists] == [8, 9, 10]
        with pytest.raises(RuntimeError, match="combined and articulation-only"):
            source.drain_articulations(Handedness.RIGHT)
        with pytest.raises(RuntimeError, match="combined and wrist-only"):
            source.drain_wrists(Handedness.RIGHT)
    finally:
        source.close()


def test_manus_articulation_converts_to_existing_mediapipe21_solver_input():
    with _source() as source:
        sample = source.read_articulation(Handedness.RIGHT)
        assert sample is not None
        mediapipe = articulation_to_mediapipe21(sample)

    assert mediapipe.schema == "mediapipe21"
    assert mediapipe.joint_names == MEDIAPIPE_JOINT_NAMES
    assert np.allclose(mediapipe.joint_positions["index_mcp"], [0.006, 0.012, -0.006])
    assert np.all(mediapipe.validity())


def test_manus_articulation_runs_through_existing_adaptive_sharpa_solver():
    with _source() as source:
        articulation = source.read_articulation(Handedness.RIGHT)
    assert articulation is not None
    wrist = WristPoseSample(
        timestamp=articulation.timestamp,
        handedness=Handedness.RIGHT,
        position=np.zeros(3),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        source="quest",
        anatomical_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
    )
    frame = observation_to_hand_frame(FusedHandObservation(articulation, wrist))

    command = LandmarkRetargeter.from_hand_model("sharpa", "right").retarget(frame)

    assert command.hand_model == "sharpa"
    assert len(command.joint_positions) == 22
    assert np.isfinite(command.joint_positions).all()


def test_manus_source_preserves_left_handedness():
    with _source(side="left") as source:
        sample = source.read_articulation(Handedness.LEFT)

    assert sample is not None
    assert sample.handedness == Handedness.LEFT
    assert source.read_articulation(Handedness.RIGHT) is None


def test_manus_source_accepts_documented_thumb_intermediate_alias():
    with _source(mode="documented_thumb_intermediate") as source:
        sample = source.read_articulation(Handedness.RIGHT)

    assert sample is not None
    assert "thumb_distal" in sample.joint_positions
    assert "thumb_intermediate" not in sample.joint_positions


def test_manus_source_reports_no_glove_without_hanging():
    source = _source("no_glove")
    started = time.monotonic()
    with pytest.raises(
        SourceUnavailableError, match="no_glove: No MANUS glove connected"
    ):
        source.start()
    assert time.monotonic() - started >= 0.04
    source.close()


def test_manus_native_empty_landscape_waits_for_no_glove_timeout(tmp_path):
    if os.environ.get("DEX_TELEOP_TEST_MANUS_NO_DEVICE") != "1":
        pytest.skip(
            "Set DEX_TELEOP_TEST_MANUS_NO_DEVICE=1 only on a host with no connected MANUS glove"
        )
    setup_root = Path.home() / "manus_setup"
    if not setup_root.is_dir():
        pytest.skip("Official MANUS SDK installation is not present on this host")

    bridge = build_manus_bridge(sdk_root=setup_root, output=tmp_path / "manus_bridge")
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        (
            str(bridge),
            "--connect-timeout",
            "3",
            "--glove-timeout",
            "1",
            "--protocol-fd",
            str(write_fd),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    try:
        exit_code = process.wait(timeout=15.0)
        protocol = os.fdopen(read_fd, "r", encoding="utf-8")
        read_fd = -1
        with protocol:
            events = [json.loads(line) for line in protocol if line.strip()]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3.0)
        if read_fd >= 0:
            os.close(read_fd)

    connected_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "status" and event.get("state") == "sdk_connected"
    )
    no_glove_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "error" and event.get("code") == "no_glove"
    )
    assert connected_index < no_glove_index
    assert not any(
        event.get("code") == "integrated_license_unavailable" for event in events
    )
    assert exit_code == 15


def test_native_manus_epoch_guard_rejects_unqualified_and_stale_callbacks(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    executable = tmp_path / "manus_lifecycle_harness"
    subprocess.run(
        (
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(_native_bridge_directory()),
            str(Path(__file__).with_name("manus_lifecycle_harness.cpp")),
            "-o",
            str(executable),
        ),
        check=True,
    )
    subprocess.run((str(executable),), check=True)


def test_manus_source_rejects_unvalidated_topology():
    source = _source("bad_topology")
    with pytest.raises(SourceUnavailableError, match="must contain 25 nodes"):
        source.start()
    source.close()


def test_manus_source_rejects_both_thumb_ip_aliases_in_one_topology():
    source = _source("duplicate_thumb_ip_alias")
    with pytest.raises(SourceUnavailableError, match="duplicate joint 'thumb_distal'"):
        source.start()
    source.close()


def test_manus_source_rejects_multiple_glove_ids_for_one_hand():
    source = _source("duplicate_side_topology")
    with pytest.raises(SourceUnavailableError, match="multiple right-hand glove IDs"):
        source.start()
    source.close()


def test_manus_topology_status_alone_does_not_make_source_ready():
    source = _source("topology_only", startup_timeout=0.2)
    with pytest.raises(
        SourceUnavailableError, match="validated right-hand articulation"
    ):
        source.start()
    source.close()


def test_manus_wrong_hand_articulation_does_not_satisfy_required_hand():
    source = _source(
        side="left", required_handedness=Handedness.RIGHT, startup_timeout=0.2
    )
    with pytest.raises(
        SourceUnavailableError, match="validated right-hand articulation"
    ):
        source.start()
    source.close()


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("hand_motion_error", "hand_motion_configuration_failed"),
        ("integrated_license_error", "integrated_license_unavailable"),
    ),
)
def test_manus_startup_configuration_errors_are_actionable(mode, message):
    source = _source(mode)
    with pytest.raises(SourceUnavailableError, match=message):
        source.start()
    source.close()


def test_manus_remote_sdk_license_error_is_actionable():
    source = _source("sdk_license_error", bridge_mode="remote")
    with pytest.raises(SourceUnavailableError, match="sdk_license_unavailable"):
        source.start()
    source.close()


def test_manus_remote_reconnect_clears_stale_state_and_resumes_pairs():
    source = _source("reconnect_after_sample", bridge_mode="remote")
    try:
        source.start()
        source.drain_hand_tracking(Handedness.RIGHT)

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            source.check_health()
            sample = source.read_articulation(Handedness.RIGHT)
            if (
                sample is not None
                and sample.source_frame_id == 8
                and source.recording_metadata()["reconnect_count"] == 1
            ):
                break
            time.sleep(0.005)
        else:
            pytest.fail("MANUS Remote source did not resume after reconnect")

        resumed = source.drain_hand_tracking(Handedness.RIGHT)
        assert [item.source_frame_id for item in resumed.articulations] == [8]
        assert [item.source_frame_id for item in resumed.wrists] == [8]
        articulation = resumed.articulations[0]
        wrist = resumed.wrists[0]
        assert wrist.provenance["connection_generation"] == 2
        assert articulation.provenance["supported_protocol_version"] == 2
        assert wrist.provenance["supported_protocol_version"] == 2
        assert json.loads(articulation.provenance["protocol_version_history"]) == [2]
        assert json.loads(wrist.provenance["protocol_version_history"]) == [2]
        assert (
            articulation.provenance["protocol_version_history"]
            == wrist.provenance["protocol_version_history"]
        )
        metadata = source.recording_metadata()
        assert metadata["protocol_version_history"] == [2]
        assert source._bridge_metadata["protocol_version_history"] == [2]
    finally:
        source.close()


def test_manus_discards_old_generation_sample_after_disconnect():
    source = _source("sample_after_disconnect", bridge_mode="remote")
    try:
        source.start()
        deadline = time.monotonic() + 1.0
        while (
            source.recording_metadata()["connection_state"] != "recovering"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert source.recording_metadata()["connection_generation"] == 2
        assert source.read_articulation(Handedness.RIGHT) is None
        assert source.read_wrist(Handedness.RIGHT) is None
    finally:
        source.close()


def test_manus_discards_new_generation_sample_before_requalification():
    source = _source("sample_before_reconnected", bridge_mode="remote")
    try:
        source.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            sample = source.read_articulation(Handedness.RIGHT)
            if sample is not None and sample.source_frame_id == 9:
                break
            time.sleep(0.005)
        else:
            pytest.fail("MANUS source did not accept post-requalification sample")
        assert sample.source_frame_id == 9
        assert sample.provenance["connection_generation"] == 2
    finally:
        source.close()


def test_manus_reconnect_fails_when_required_glove_never_requalifies():
    source = _source("reconnect_no_glove", bridge_mode="remote")
    try:
        source.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                source.check_health()
            except SourceUnavailableError as error:
                assert "reconnect_stream_not_qualified" in str(error)
                break
            time.sleep(0.005)
        else:
            pytest.fail("MANUS reconnect failure was not reported")
    finally:
        source.close()


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("remove_after_sample", "Required MANUS right-hand glove .* disconnected"),
        ("disconnect_after_sample", "MANUS Integrated host disconnected"),
    ),
)
def test_manus_disconnect_invalidates_required_hand_state(mode, message):
    source = _source(mode)
    try:
        source.start()
        assert source.read_articulation(Handedness.RIGHT) is not None

        deadline = time.monotonic() + 1.0
        while True:
            try:
                source.check_health()
            except SourceUnavailableError as error:
                assert re.search(message, str(error))
                break
            if time.monotonic() >= deadline:
                pytest.fail("MANUS source did not report the required-hand disconnect")
            time.sleep(0.01)

        assert source.read_articulation(Handedness.RIGHT) is None
        assert source.frame_diagnostics(Handedness.RIGHT) is None
        assert not source._topologies
    finally:
        source.close()


def test_manus_source_reports_pending_queue_overflow_without_silent_sample_loss():
    source = _source("overflow", max_pending_samples=2)
    try:
        source.start()
        initial = source.drain_articulations(Handedness.RIGHT)
        assert tuple(sample.source_frame_id for sample in initial) == (7,)
        deadline = time.monotonic() + 1.0
        while True:
            try:
                source.check_health()
            except SourceUnavailableError as error:
                assert "overflowed its 2-sample queue" in str(error)
                break
            if time.monotonic() >= deadline:
                pytest.fail("MANUS source did not report its pending-queue overflow")
            time.sleep(0.01)
    finally:
        source.close()


def test_manus_latest_only_reader_does_not_activate_lossless_queue_or_overflow():
    source = _source("overflow", max_pending_samples=2)
    try:
        source.start()
        time.sleep(0.2)
        source.check_health()

        sample = source.read_articulation(Handedness.RIGHT)
        assert sample is not None
        assert sample.source_frame_id == 10
    finally:
        source.close()


def test_manus_source_restart_discards_undrained_samples_from_the_previous_process():
    source = _source()
    try:
        source.start()
        assert source.read_articulation(Handedness.RIGHT) is not None
        source.close()

        source.start()
        drained = source.drain_articulations(Handedness.RIGHT)

        assert tuple(sample.source_frame_id for sample in drained) == (7,)
    finally:
        source.close()


def test_manus_native_build_resources_live_inside_the_installed_package():
    package_root = Path(__file__).parents[1] / "src/dex_teleop"
    native_directory = _native_bridge_directory()

    assert native_directory.is_relative_to(package_root.resolve())
    assert (native_directory / "manus_bridge.cpp").is_file()
    assert (native_directory / "build_manus_bridge.sh").is_file()
    assert (native_directory / "build_manus_remote_runtime.sh").is_file()


def test_manus_remote_runtime_helper_rejects_unsafe_and_unowned_prefixes(tmp_path):
    repository = Path(__file__).parents[1].resolve()
    helper = _native_bridge_directory() / "build_manus_remote_runtime.sh"
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment.pop("DEX_TELEOP_MANUS_REMOTE_RUNTIME", None)
    approved_prefix = (
        tmp_path / "cache/dex_teleop/manus_sdk_v3.1.1_remote_runtime"
    ).resolve()

    approved = subprocess.run(
        (str(helper), "--validate-prefix-only", "--prefix", str(approved_prefix)),
        cwd=repository,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert approved.returncode == 0, approved.stderr

    protected_prefixes = [
        "/",
        str(Path.home()),
        ".",
        "/usr/local",
        str(repository),
        str(Path.home() / "manus_setup"),
        str(tmp_path / "cache/dex_teleop"),
    ]
    if conda_prefix := environment.get("CONDA_PREFIX"):
        protected_prefixes.append(conda_prefix)
    for protected_prefix in protected_prefixes:
        rejected = subprocess.run(
            (
                str(helper),
                "--force",
                "--validate-prefix-only",
                "--prefix",
                protected_prefix,
            ),
            cwd=repository,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        assert rejected.returncode != 0, protected_prefix

    approved_prefix.mkdir(parents=True)
    sentinel = approved_prefix / "do-not-delete"
    sentinel.write_text("unowned\n")
    unowned = subprocess.run(
        (
            str(helper),
            "--force",
            "--validate-prefix-only",
            "--prefix",
            str(approved_prefix),
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert unowned.returncode == 4
    assert sentinel.read_text() == "unowned\n"

    alias_parent = tmp_path / "cache/dex_teleop/manus_remote_runtimes"
    alias_parent.mkdir(parents=True)
    repository_alias = alias_parent / "repository-alias"
    repository_alias.symlink_to(repository, target_is_directory=True)
    aliased = subprocess.run(
        (
            str(helper),
            "--force",
            "--validate-prefix-only",
            "--prefix",
            str(repository_alias),
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert aliased.returncode != 0


def test_manus_remote_runtime_helper_requires_marker_for_outside_cache(tmp_path):
    helper = _native_bridge_directory() / "build_manus_remote_runtime.sh"
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment.pop("DEX_TELEOP_MANUS_REMOTE_RUNTIME", None)
    owned_prefix = (tmp_path / "explicit-owned-runtime").resolve()
    owned_prefix.mkdir()
    marker = owned_prefix / ".dex-teleop-manus-remote-runtime"
    marker.write_text(
        "kind=dex_teleop_manus_remote_runtime\n"
        "schema=2\n"
        f"canonical_prefix={owned_prefix}\n"
    )

    without_override = subprocess.run(
        (str(helper), "--validate-prefix-only", "--prefix", str(owned_prefix)),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert without_override.returncode != 0

    with_override = subprocess.run(
        (
            str(helper),
            "--allow-owned-prefix-outside-cache",
            "--validate-prefix-only",
            "--prefix",
            str(owned_prefix),
        ),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert with_override.returncode == 0, with_override.stderr

    marker.write_text(
        "kind=dex_teleop_manus_remote_runtime\n"
        "schema=2\n"
        "canonical_prefix=/different/path\n"
    )
    mismatched_marker = subprocess.run(
        (
            str(helper),
            "--allow-owned-prefix-outside-cache",
            "--force",
            "--validate-prefix-only",
            "--prefix",
            str(owned_prefix),
        ),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert mismatched_marker.returncode != 0
    assert owned_prefix.is_dir()


def test_manus_remote_runtime_helper_records_strict_provenance_and_manifest():
    helper = _native_bridge_directory() / "build_manus_remote_runtime.sh"
    source = helper.read_text()

    assert 'worktree add --detach "$build_source" "$grpc_commit"' in source
    assert "source_status_snapshot" in source
    assert "submodule foreach --quiet --recursive" in source
    assert "--untracked-files=all" in source
    assert "submodule status --recursive" in source
    assert "source_prebuild_dirty=false" in source
    assert "source_postbuild_dirty=$postbuild_dirty" in source
    assert "source_submodule_manifest_sha256=$submodule_manifest_sha256" in source
    assert "MANIFEST.sha256" in source
    assert "validate_load_closure" in source


def test_manus_sdk_discovery_finds_installed_3_1_1_layout():
    setup_root = Path.home() / "manus_setup"
    if not setup_root.is_dir():
        pytest.skip("Official MANUS SDK installation is not present on this host")

    layout = discover_manus_sdk(setup_root)

    assert (layout.include_dir / "ManusSDK.h").is_file()
    assert layout.integrated_library.name == "libManusSDK_Integrated.so"
    assert layout.integrated_library.is_file()
    assert layout.remote_library.name == "libManusSDK.so"
    assert layout.remote_library.is_file()


def test_manus_bridge_uses_mode_specific_rpath_to_prefer_integrated_runtime(tmp_path):
    setup_root = Path.home() / "manus_setup"
    if not setup_root.is_dir():
        pytest.skip("Official MANUS SDK installation is not present on this host")
    readelf = shutil.which("readelf")
    if readelf is None:
        pytest.skip("readelf is not installed")

    bridge = build_manus_bridge(sdk_root=setup_root, output=tmp_path / "manus_bridge")
    dynamic_section = subprocess.run(
        (readelf, "-d", str(bridge)),
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "(RPATH)" in dynamic_section
    assert "$ORIGIN/manus_runtime_integrated" in dynamic_section
    assert "(RUNPATH)" not in dynamic_section


def test_failed_manus_bridge_rebuild_preserves_published_pair(tmp_path):
    setup_root = Path.home() / "manus_setup"
    if not setup_root.is_dir():
        pytest.skip("Official MANUS SDK installation is not present on this host")
    helper = _native_bridge_directory() / "build_manus_bridge.sh"
    output = tmp_path / "manus_bridge"
    command = (
        str(helper),
        "--mode",
        "integrated",
        "--sdk-root",
        str(setup_root),
        "--output",
        str(output),
    )

    subprocess.run(command, capture_output=True, text=True, check=True)
    runtime = tmp_path / "manus_runtime_integrated"
    original_artifact = output.read_bytes()
    original_runtime = runtime.resolve()

    environment = os.environ.copy()
    environment["CXX"] = shutil.which("false") or "/bin/false"
    failed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert failed.returncode != 0
    assert output.read_bytes() == original_artifact
    assert runtime.resolve() == original_runtime
    assert (
        subprocess.run(
            (str(output), "--protocol-version"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == "dex_teleop.manus 2"
    )


@pytest.mark.parametrize(
    "failure_phase",
    (
        "before-runtime-publish",
        "after-runtime-publish",
        "after-executable-publish",
        "final-verification",
    ),
)
def test_manus_bridge_publication_failure_restores_exact_owned_pair(
    tmp_path, failure_phase
):
    setup_root = Path.home() / "manus_setup"
    if not setup_root.is_dir():
        pytest.skip("Official MANUS SDK installation is not present on this host")
    helper = _native_bridge_directory() / "build_manus_bridge.sh"
    output = tmp_path / "manus_bridge"
    runtime = tmp_path / "manus_runtime_integrated"
    bundle_root = tmp_path / ".dex_teleop_manus_bridge_bundles"
    command = (
        str(helper),
        "--mode",
        "integrated",
        "--sdk-root",
        str(setup_root),
        "--output",
        str(output),
    )

    subprocess.run(command, capture_output=True, text=True, check=True)
    original_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    original_link_value = os.readlink(runtime)
    original_runtime_target = runtime.resolve()
    original_bundles = {path.name for path in bundle_root.iterdir()}
    marker = original_runtime_target.parent / ".dex-teleop-manus-bridge-bundle"
    marker_contents = marker.read_text()
    assert "kind=dex_teleop_manus_bridge_bundle" in marker_contents
    assert "mode=integrated" in marker_contents
    assert f"executable_sha256={original_hash}" in marker_contents

    environment = os.environ.copy()
    environment["DEX_TELEOP_MANUS_BRIDGE_TESTING"] = "1"
    environment["DEX_TELEOP_MANUS_BRIDGE_TEST_FAIL_PHASE"] = failure_phase
    failed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert failed.returncode == 5, failed.stderr
    assert f"failure at phase: {failure_phase}" in failed.stderr
    assert hashlib.sha256(output.read_bytes()).hexdigest() == original_hash
    assert os.readlink(runtime) == original_link_value
    assert runtime.resolve() == original_runtime_target
    assert marker.read_text() == marker_contents
    assert {path.name for path in bundle_root.iterdir()} == original_bundles

    loader = subprocess.run(
        ("ldd", "-r", str(output)),
        capture_output=True,
        text=True,
        check=True,
        env={
            key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
        },
    )
    assert "not found" not in loader.stdout
    assert "undefined symbol" not in loader.stdout
    assert (
        subprocess.run(
            (str(output), "--protocol-version"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == "dex_teleop.manus 2"
    )
    transient_markers = (".publish.", ".previous.", ".rollback.", ".stage.")
    assert not any(
        any(marker in path.name for marker in transient_markers)
        for path in tmp_path.iterdir()
    )


def test_manus_bridge_refuses_unowned_runtime_link_without_mutation(tmp_path):
    setup_root = Path.home() / "manus_setup"
    if not setup_root.is_dir():
        pytest.skip("Official MANUS SDK installation is not present on this host")
    helper = _native_bridge_directory() / "build_manus_bridge.sh"
    output = tmp_path / "manus_bridge"
    runtime = tmp_path / "manus_runtime_integrated"
    bundle_root = tmp_path / ".dex_teleop_manus_bridge_bundles"
    command = (
        str(helper),
        "--mode",
        "integrated",
        "--sdk-root",
        str(setup_root),
        "--output",
        str(output),
    )

    subprocess.run(command, capture_output=True, text=True, check=True)
    original_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    original_bundles = {path.name for path in bundle_root.iterdir()}
    unowned_runtime = tmp_path / "unowned-runtime"
    unowned_runtime.mkdir()
    (unowned_runtime / "libManusSDK.so").symlink_to(
        setup_root
        / "vendor/ManusSDK_v3.1.1/ROS2/ManusSDK/lib/libManusSDK_Integrated.so"
    )
    runtime.unlink()
    runtime.symlink_to(unowned_runtime)
    unowned_link_value = os.readlink(runtime)

    failed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert failed.returncode == 5
    assert "outside the managed bridge bundles" in failed.stderr
    assert hashlib.sha256(output.read_bytes()).hexdigest() == original_hash
    assert os.readlink(runtime) == unowned_link_value
    assert runtime.resolve() == unowned_runtime
    assert {path.name for path in bundle_root.iterdir()} == original_bundles


def test_remote_manus_bridge_uses_isolated_runtime_and_protocol_probe(tmp_path):
    setup_root = Path.home() / "manus_setup"
    if not setup_root.is_dir():
        pytest.skip("Official MANUS SDK installation is not present on this host")
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    runtime_prefix = (
        Path(
            os.environ.get(
                "DEX_TELEOP_MANUS_REMOTE_RUNTIME",
                cache_root / "dex_teleop/manus_sdk_v3.1.1_remote_runtime",
            )
        )
        .expanduser()
        .resolve()
    )
    required_libraries = (
        runtime_prefix / "lib/libgrpc.so.9",
        runtime_prefix / "lib/libgrpc++.so.1",
        runtime_prefix / "lib/libprotobuf.so.22",
    )
    if not all(library.is_file() for library in required_libraries):
        pytest.skip("Isolated MANUS Remote runtime is not installed on this host")

    bridge = build_manus_bridge(
        sdk_root=setup_root,
        output=tmp_path / "manus_bridge_remote",
        mode="remote",
    )
    probe = subprocess.run(
        (str(bridge), "--protocol-version"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "dex_teleop.manus 2"

    dynamic_section = subprocess.run(
        ("readelf", "-d", str(bridge)),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "(RPATH)" in dynamic_section
    assert "$ORIGIN/manus_runtime_remote" in dynamic_section
    assert str(runtime_prefix / "lib") in dynamic_section
    assert "(RUNPATH)" not in dynamic_section

    loader_environment = os.environ.copy()
    loader_environment.pop("LD_LIBRARY_PATH", None)
    loader = subprocess.run(
        ("ldd", str(bridge)),
        capture_output=True,
        text=True,
        check=True,
        env=loader_environment,
    )
    assert "not found" not in loader.stdout
    for soname in ("libgrpc.so.9", "libgrpc++.so.1", "libprotobuf.so.22"):
        assert f"{soname} => {runtime_prefix}/lib/{soname}" in loader.stdout


def test_manus_source_does_not_retain_multiline_license_data_in_errors():
    source = _source()

    class _Process:
        stderr = io.StringIO(
            "normal diagnostic\n"
            "dongle license data: {\n"
            '  "Cust": "private",\n'
            '  "Key": "secret"\n'
            "}\n"
            "safe tail\n"
        )

    source._process = _Process()
    source._stderr_loop()

    assert tuple(source._stderr_tail) == ("normal diagnostic", "safe tail")

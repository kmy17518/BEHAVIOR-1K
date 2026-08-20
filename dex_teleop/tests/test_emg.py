from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dex_teleop.emg import (
    ACTION_TIMING_GROUP,
    EMG_GROUP,
    SYNC_GROUP,
    ActionTimingSample,
    EmgBatch,
    EmgSession,
    emg_staging_path,
    merge_emg_recording,
    write_action_timing_episodes,
)
from dex_teleop.emg.decoder import (
    DecoderRunner,
    MODEL_CHANNEL_COUNT,
    MODEL_WINDOW_SAMPLES,
    preprocess_emg,
    required_source_samples,
)
from dex_teleop.emg.sidecar import (
    EmgHdf5Writer,
    _batch_from_sensor_data,
    _configure_hardware_filters,
    _select_device,
)
from dex_teleop.emg.ui import (
    DecodedHandMonitor,
    IMPEDANCE_FAIR_COLOR,
    IMPEDANCE_GOOD_COLOR,
    IMPEDANCE_POOR_COLOR,
    IMPEDANCE_UNKNOWN_COLOR,
    impedance_display,
)
from dex_teleop.arat import AratTaskCatalog
from dex_teleop.omnigibson.launcher import _parser


def _batch(indices, receive_ns, *, channel_count=2):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.arange(len(indices) * channel_count, dtype=np.float32).reshape(len(indices), channel_count)
    return EmgBatch(
        receive_monotonic_ns=receive_ns,
        sample_index=indices,
        sdk_timestamp_ms=indices * 2,
        signal_uv=values,
        raw_adc=values.astype(np.int32),
        is_lost=np.zeros(len(indices), dtype=np.bool_),
        impedance_ohm=np.full(channel_count, 1000.0, dtype=np.float32),
        saturation=np.zeros(channel_count, dtype=np.float32),
    )


def _write_emg(path):
    writer = EmgHdf5Writer(
        path,
        sample_rate_hz=500.0,
        channel_count=2,
        metadata={"device_name": "OYTest"},
    )
    writer.append(_batch([100, 101, 102], 1_010_000_000))
    writer.append(_batch([103, 104], 1_014_000_000))
    writer.close()


def _write_trajectory(path, lengths):
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        for episode_id, length in lengths:
            episode = data.create_group(f"demo_{episode_id}")
            episode.attrs["num_samples"] = length
            episode.create_dataset("action", data=np.zeros((length, 3), dtype=np.float32))


def test_emg_writer_preserves_native_samples_and_estimates_monotonic_time(tmp_path):
    path = tmp_path / "emg.hdf5"
    _write_emg(path)

    with h5py.File(path, "r") as recording:
        emg = recording[EMG_GROUP]
        assert emg.attrs["sample_rate_hz"] == 500.0
        assert emg.attrs["channel_count"] == 2
        assert emg.attrs["device_name"] == "OYTest"
        np.testing.assert_array_equal(emg["samples/sample_index"][:], [100, 101, 102, 103, 104])
        np.testing.assert_array_equal(
            emg["samples/estimated_monotonic_ns"][:],
            [1_006_000_000, 1_008_000_000, 1_010_000_000, 1_012_000_000, 1_014_000_000],
        )
        np.testing.assert_array_equal(emg["batches/first_sample_row"][:], [0, 3])
        np.testing.assert_array_equal(emg["batches/sample_count"][:], [3, 2])


def test_emg_writer_rejects_nonadvancing_sample_index(tmp_path):
    writer = EmgHdf5Writer(
        tmp_path / "emg.hdf5",
        sample_rate_hz=500.0,
        channel_count=2,
        metadata={},
    )
    try:
        writer.append(_batch([10, 11], 100_000_000))
        with pytest.raises(ValueError, match="did not advance"):
            writer.append(_batch([11, 12], 104_000_000))
    finally:
        writer.close()


def test_action_timing_and_emg_are_merged_with_per_action_ranges(tmp_path):
    recording_path = tmp_path / "trajectory.hdf5"
    emg_path = tmp_path / "emg.hdf5"
    _write_trajectory(recording_path, [(0, 2)])
    _write_emg(emg_path)
    timings = [
        ActionTimingSample(1_008_000_000, 1_010_000_000, 0.0, 1.0 / 30.0),
        ActionTimingSample(1_012_000_000, 1_014_000_000, 1.0 / 30.0, 2.0 / 30.0),
    ]

    write_action_timing_episodes(recording_path, [timings])
    merge_emg_recording(recording_path, emg_path)

    with h5py.File(recording_path, "r") as recording:
        assert ACTION_TIMING_GROUP in recording
        assert EMG_GROUP in recording
        episode = recording[f"{SYNC_GROUP}/demo_0"]
        assert episode["episode_emg_row_start"][()] == 1
        assert episode["episode_emg_row_end"][()] == 5
        np.testing.assert_array_equal(episode["action_emg_row_start"][:], [1, 3])
        np.testing.assert_array_equal(episode["action_emg_row_end"][:], [3, 5])


def test_action_timing_rejects_trajectory_length_mismatch(tmp_path):
    recording_path = tmp_path / "trajectory.hdf5"
    _write_trajectory(recording_path, [(0, 2)])
    sample = ActionTimingSample(10, 11, 0.0, 0.1)

    with pytest.raises(RuntimeError, match="has 1 steps; trajectory has 2"):
        write_action_timing_episodes(recording_path, [[sample]])


def test_sdk_callback_is_copied_into_channel_major_samples():
    def sample(index, channel):
        return SimpleNamespace(
            sampleIndex=index,
            timeStampInMs=index * 2,
            data=float(index + channel),
            rawData=index * 10 + channel,
            isLost=index == 2,
            impedance=1000.0 + channel,
            saturation=float(channel),
        )

    data = SimpleNamespace(channelSamples=[[sample(i, channel) for i in (1, 2)] for channel in (0, 1)])
    batch = _batch_from_sensor_data(data, 123)

    np.testing.assert_array_equal(batch.sample_index, [1, 2])
    np.testing.assert_allclose(batch.signal_uv, [[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_array_equal(batch.raw_adc, [[10, 11], [20, 21]])
    np.testing.assert_array_equal(batch.is_lost, [False, True])


def test_device_selection_accepts_unique_name_or_address():
    first = SimpleNamespace(Name="OYMotion A", Address="AA", RSSI=-40)
    second = SimpleNamespace(Name="unrelated", Address="BB", RSSI=-50)

    assert _select_device([first, second], None) is first
    assert _select_device([first, second], "aa") is first
    assert _select_device([first, second], "bb") is second
    with pytest.raises(RuntimeError, match="matching"):
        _select_device([first], "missing")


def test_hardware_filters_are_explicitly_set_and_verified():
    class Profile:
        def __init__(self):
            self.states = {
                "FILTER_HPF": "OFF",
                "FILTER_LPF": "OFF",
                "FILTER_50HZ": "ON",
                "FILTER_60HZ": "OFF",
            }

        def setParam(self, key, value):
            self.states[key] = value
            return "OK"

        def getParam(self, key):
            assert key == "FILTER"
            return "|".join(item for pair in self.states.items() for item in pair)

    profile = Profile()
    configuration = _configure_hardware_filters(profile, hpf=True, lpf=True, notch="60")

    assert profile.states == {
        "FILTER_HPF": "ON",
        "FILTER_LPF": "ON",
        "FILTER_50HZ": "OFF",
        "FILTER_60HZ": "ON",
    }
    assert configuration == "FILTER_HPF|ON|FILTER_LPF|ON|FILTER_50HZ|OFF|FILTER_60HZ|ON"


def test_session_keeps_a_bounded_preview_and_staging_name(tmp_path):
    path = tmp_path / "recording.hdf5"
    session = EmgSession(emg_staging_path(path, process_id=42), preview_capacity=3)
    session._handle_message(
        {"type": "ready", "status": "streaming", "sample_rate_hz": 500, "channel_count": 2}
    )
    session._handle_message(
        {
            "type": "samples",
            "receive_monotonic_ns": 100,
            "sample_index": [1, 2],
            "signal_uv": [[1.0, 2.0], [3.0, 4.0]],
            "is_lost": [False, False],
            "impedance_ohm": [100_000.0, 200_000.0],
        }
    )
    session._handle_message(
        {
            "type": "samples",
            "receive_monotonic_ns": 200,
            "sample_index": [3, 4],
            "signal_uv": [[5.0, 6.0], [7.0, 8.0]],
            "is_lost": [False, True],
            "impedance_ohm": [300_000.0, 400_000.0],
        }
    )

    preview = session.preview()
    assert session.output_path.name == ".recording.hdf5.42.emg.in_progress.hdf5"
    np.testing.assert_array_equal(preview.sample_index, [3, 4])
    np.testing.assert_array_equal(preview.is_lost, [False, True])
    np.testing.assert_array_equal(preview.impedance_ohm, [300_000.0, 400_000.0])


def test_session_keeps_latest_decoded_hand_mesh(tmp_path):
    session = EmgSession(tmp_path / "emg.hdf5", visualize_decoder=True)
    session._handle_message(
        {
            "type": "decoder_geometry",
            "triangles": [[0, 1, 2]],
            "status": "buffering model window",
        }
    )
    session._handle_message(
        {
            "type": "decoder_frame",
            "sequence": 3,
            "source_sample_index": 9876,
            "joint_angles": np.arange(20, dtype=np.float32).tolist(),
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "inference_ms": 12.5,
            "status": "live",
        }
    )

    decoded = session.decoded_hand()
    assert decoded.sequence == 3
    assert decoded.source_sample_index == 9876
    assert decoded.inference_ms == 12.5
    assert decoded.status == "live"
    np.testing.assert_array_equal(decoded.triangles, [[0, 1, 2]])
    np.testing.assert_array_equal(decoded.joint_angles, np.arange(20, dtype=np.float32))


def test_emg2pose_preprocessing_matches_checkpoint_shape_and_end_alignment():
    source_rate_hz = 500.0
    count = required_source_samples(source_rate_hz)
    samples = np.column_stack(
        [np.arange(count, dtype=np.float32) + 10.0 * channel for channel in range(8)]
    )

    model_input = preprocess_emg(samples, source_rate_hz)

    assert model_input.shape == (MODEL_CHANNEL_COUNT, MODEL_WINDOW_SAMPLES)
    assert model_input.dtype == np.float32
    np.testing.assert_allclose(model_input[[0, -1], -1], samples[-1, [0, -1]])


def test_decoder_target_rate_advances_on_the_source_sample_clock():
    class Decoder:
        @staticmethod
        def decode(_emg):
            return np.zeros(20, dtype=np.float32), 1.0

        @staticmethod
        def skin(_pose):
            return np.zeros((3, 3), dtype=np.float32)

    runner = DecoderRunner(Decoder(), source_rate_hz=500.0, inference_hz=10.0)
    try:
        initial_last_index = runner.required_samples - 1
        runner.append(np.zeros((runner.required_samples, 8), dtype=np.float32), initial_last_index)
        assert runner.poll() is None
        runner._future.result(timeout=1.0)
        assert runner.poll().source_sample_index == initial_last_index

        runner.append(np.zeros((32, 8), dtype=np.float32), initial_last_index + 32)
        assert runner.poll() is None
        assert runner._future is None

        runner.append(np.zeros((32, 8), dtype=np.float32), initial_last_index + 64)
        assert runner.poll() is None
        assert runner._future is not None
    finally:
        runner.close()


def test_impedance_display_matches_sdk_example_thresholds():
    assert impedance_display(float("nan")) == ("— kOhm", IMPEDANCE_UNKNOWN_COLOR)
    assert impedance_display(0.0) == ("— kOhm", IMPEDANCE_UNKNOWN_COLOR)
    assert impedance_display(500_000.0) == ("500 kOhm", IMPEDANCE_GOOD_COLOR)
    assert impedance_display(750_000.0) == ("750 kOhm", IMPEDANCE_FAIR_COLOR)
    assert impedance_display(1_000_000.0) == ("1000 kOhm", IMPEDANCE_POOR_COLOR)


def test_decoder_mesh_expands_colors_per_triangle_corner():
    vertices = np.asarray(
        [[-1.0, -1.0, -1.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 0.5]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    colors = np.asarray(DecodedHandMonitor._surface_colors(vertices, triangles))

    assert colors.shape == (triangles.size, 4)
    np.testing.assert_array_equal(colors[1], colors[3])
    np.testing.assert_array_equal(colors[2], colors[5])


def test_launcher_exposes_emg_acquisition_options():
    args = _parser(AratTaskCatalog()).parse_args(
        [
            "--task",
            "arat_grasp_block_10cm",
            "--emg",
            "--emg-device",
            "OYMotion",
            "--emg-adapter",
            "hci1",
            "--emg-sdk-path",
            "../synchroni-sensor-sdk",
            "--emg-hpf",
            "--no-emg-lpf",
            "--emg-notch",
            "60",
            "--no-emg-display",
            "--visualize-decoder",
            "--emg2pose-root",
            "../emg2pose",
            "--emg2pose-checkpoint",
            "../emg2pose/model.ckpt",
            "--emg2pose-device",
            "cpu",
            "--emg2pose-inference-hz",
            "10",
        ]
    )

    assert args.emg is True
    assert args.emg_device == "OYMotion"
    assert args.emg_adapter == "hci1"
    assert args.emg_sdk_path == "../synchroni-sensor-sdk"
    assert args.emg_hpf is True
    assert args.emg_lpf is False
    assert args.emg_notch == "60"
    assert args.no_emg_display is True
    assert args.visualize_decoder is True
    assert args.emg2pose_root == "../emg2pose"
    assert args.emg2pose_checkpoint == "../emg2pose/model.ckpt"
    assert args.emg2pose_device == "cpu"
    assert args.emg2pose_inference_hz == 10.0

"""Standalone Synchroni SDK process that records EMG and publishes previews."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time
import traceback

import h5py
import numpy as np

from dex_teleop.emg.recording import EMG_GROUP, EMG_SCHEMA_VERSION
from dex_teleop.emg.types import EmgBatch


PACKAGE_COUNT = 32
# profile.connect() uses a 10-second wrapper around an SDK worker that itself
# allows up to 25 seconds.  On a marginal BLE link the wrapper can return False
# while the worker subsequently connects, leaving BlueZ with a stale connection.
SDK_COMMAND_TIMEOUT_S = 20
SDK_CONNECT_TIMEOUT_S = 30
# The SDK performs an initial battery read during profile.init().  Periodic
# refreshes issue GET_BATTERY_LEVEL on the command characteristic while EMG
# notifications are active; some OYWW1000 firmware drops the GATT connection
# when that write overlaps a busy stream.  Battery state is not part of the
# synchronized dataset, so keep the initial read and disable in-stream polling.
POWER_REFRESH_PERIOD_MS = 0
PREVIEW_SAMPLE_LIMIT = 64


class EmgHdf5Writer:
    """Incrementally write copied SDK batches and finalize their clock mapping."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        sample_rate_hz: float,
        channel_count: int,
        metadata: dict,
        flush_interval_s: float = 1.0,
    ) -> None:
        if sample_rate_hz <= 0 or channel_count <= 0:
            raise ValueError("EMG sample rate and channel count must be positive")
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate_hz = float(sample_rate_hz)
        self.channel_count = int(channel_count)
        self.flush_interval_s = float(flush_interval_s)
        self._last_flush = time.monotonic()
        self._sample_count = 0
        self._batch_count = 0
        self._last_sample_index: int | None = None

        self._file = h5py.File(self.output_path, "w")
        self._group = self._file.create_group(EMG_GROUP)
        self._group.attrs["schema_version"] = EMG_SCHEMA_VERSION
        self._group.attrs["sample_rate_hz"] = self.sample_rate_hz
        self._group.attrs["channel_count"] = self.channel_count
        self._group.attrs["sample_clock"] = "SDK sampleIndex at the configured sample rate"
        self._group.attrs["receive_clock"] = "time.monotonic_ns in the EMG sidecar process"
        self._group.attrs["receive_timestamp_semantics"] = "public SensorData callback entry"
        self._group.attrs["estimated_timestamp_semantics"] = (
            "nominal sample period anchored to the callback with minimum observed receive delay"
        )
        monotonic_anchor_ns = time.monotonic_ns()
        self._group.attrs["session_monotonic_anchor_ns"] = monotonic_anchor_ns
        self._group.attrs["session_unix_anchor_ns"] = time.time_ns()
        for key, value in metadata.items():
            if value is not None:
                self._group.attrs[str(key)] = value

        samples = self._group.create_group("samples")
        batches = self._group.create_group("batches")
        sample_chunks = (2048, self.channel_count)
        vector_chunks = (2048,)
        batch_chunks = (256,)
        batch_channel_chunks = (256, self.channel_count)
        self._datasets = {
            "signal_uv": samples.create_dataset(
                "signal_uv",
                shape=(0, self.channel_count),
                maxshape=(None, self.channel_count),
                dtype=np.float32,
                chunks=sample_chunks,
            ),
            "raw_adc": samples.create_dataset(
                "raw_adc",
                shape=(0, self.channel_count),
                maxshape=(None, self.channel_count),
                dtype=np.int32,
                chunks=sample_chunks,
            ),
            "sample_index": samples.create_dataset(
                "sample_index", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=vector_chunks
            ),
            "sdk_timestamp_ms": samples.create_dataset(
                "sdk_timestamp_ms", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=vector_chunks
            ),
            "is_lost": samples.create_dataset(
                "is_lost", shape=(0,), maxshape=(None,), dtype=np.bool_, chunks=vector_chunks
            ),
            "batch_id": samples.create_dataset(
                "batch_id", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=vector_chunks
            ),
            "receive_monotonic_ns": batches.create_dataset(
                "receive_monotonic_ns", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=batch_chunks
            ),
            "first_sample_row": batches.create_dataset(
                "first_sample_row", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=batch_chunks
            ),
            "sample_count": batches.create_dataset(
                "sample_count", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=batch_chunks
            ),
            "impedance_ohm": batches.create_dataset(
                "impedance_ohm",
                shape=(0, self.channel_count),
                maxshape=(None, self.channel_count),
                dtype=np.float32,
                chunks=batch_channel_chunks,
            ),
            "saturation": batches.create_dataset(
                "saturation",
                shape=(0, self.channel_count),
                maxshape=(None, self.channel_count),
                dtype=np.float32,
                chunks=batch_channel_chunks,
            ),
        }

    def append(self, batch: EmgBatch) -> None:
        if batch.channel_count != self.channel_count:
            raise ValueError(
                f"EMG batch has {batch.channel_count} channels; expected {self.channel_count}"
            )
        if self._last_sample_index is not None and int(batch.sample_index[0]) <= self._last_sample_index:
            raise ValueError(
                f"EMG sample index did not advance across callbacks: {int(batch.sample_index[0])} "
                f"after {self._last_sample_index}"
            )
        sample_start = self._sample_count
        sample_end = sample_start + batch.sample_count
        for name in ("signal_uv", "raw_adc", "sample_index", "sdk_timestamp_ms", "is_lost", "batch_id"):
            dataset = self._datasets[name]
            dataset.resize(sample_end, axis=0)
        self._datasets["signal_uv"][sample_start:sample_end] = batch.signal_uv
        self._datasets["raw_adc"][sample_start:sample_end] = batch.raw_adc
        self._datasets["sample_index"][sample_start:sample_end] = batch.sample_index
        self._datasets["sdk_timestamp_ms"][sample_start:sample_end] = batch.sdk_timestamp_ms
        self._datasets["is_lost"][sample_start:sample_end] = batch.is_lost
        self._datasets["batch_id"][sample_start:sample_end] = self._batch_count

        batch_end = self._batch_count + 1
        for name in ("receive_monotonic_ns", "first_sample_row", "sample_count", "impedance_ohm", "saturation"):
            self._datasets[name].resize(batch_end, axis=0)
        self._datasets["receive_monotonic_ns"][self._batch_count] = batch.receive_monotonic_ns
        self._datasets["first_sample_row"][self._batch_count] = sample_start
        self._datasets["sample_count"][self._batch_count] = batch.sample_count
        self._datasets["impedance_ohm"][self._batch_count] = batch.impedance_ohm
        self._datasets["saturation"][self._batch_count] = batch.saturation

        self._sample_count = sample_end
        self._batch_count = batch_end
        self._last_sample_index = int(batch.sample_index[-1])
        now = time.monotonic()
        if now - self._last_flush >= self.flush_interval_s:
            self._file.flush()
            self._last_flush = now

    def close(self) -> None:
        if self._file is None:
            return
        samples = self._group["samples"]
        indices = np.asarray(samples["sample_index"], dtype=np.int64)
        estimated = np.empty(len(indices), dtype=np.int64)
        nominal_period_ns = int(round(1e9 / self.sample_rate_hz))
        self._group.attrs["nominal_sample_period_ns"] = nominal_period_ns
        self._group.attrs["sample_count"] = len(indices)
        self._group.attrs["batch_count"] = self._batch_count
        self._group.attrs["stream_end_monotonic_ns"] = time.monotonic_ns()

        if len(indices):
            first_index = int(indices[0])
            first_rows = np.asarray(self._group["batches/first_sample_row"], dtype=np.int64)
            batch_counts = np.asarray(self._group["batches/sample_count"], dtype=np.int64)
            receive_times = np.asarray(self._group["batches/receive_monotonic_ns"], dtype=np.int64)
            last_rows = first_rows + batch_counts - 1
            last_indices = indices[last_rows]
            candidate_origins = receive_times - (last_indices - first_index) * nominal_period_ns
            origin_ns = int(candidate_origins.min())
            estimated[:] = origin_ns + (indices - first_index) * nominal_period_ns
            residuals = receive_times - estimated[last_rows]
            self._group.attrs["first_sample_index"] = first_index
            self._group.attrs["estimated_first_sample_monotonic_ns"] = origin_ns
            self._group.attrs["receive_residual_min_ns"] = int(residuals.min())
            self._group.attrs["receive_residual_max_ns"] = int(residuals.max())
        else:
            self._group.attrs["first_sample_index"] = -1
            self._group.attrs["estimated_first_sample_monotonic_ns"] = -1
            self._group.attrs["receive_residual_min_ns"] = -1
            self._group.attrs["receive_residual_max_ns"] = -1
        samples.create_dataset("estimated_monotonic_ns", data=estimated, dtype=np.int64, chunks=True)
        self._file.flush()
        self._file.close()
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()


class _UdpPublisher:
    def __init__(self, host: str, port: int) -> None:
        self._address = (host, int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message_type: str, **values) -> None:
        payload = json.dumps({"type": message_type, **values}, separators=(",", ":")).encode("utf-8")
        self._socket.sendto(payload, self._address)

    def close(self) -> None:
        self._socket.close()


def _batch_from_sensor_data(data, receive_monotonic_ns: int) -> EmgBatch:
    channels = data.channelSamples
    if not channels or not channels[0]:
        raise ValueError("Received an empty EMG callback")
    sample_count = len(channels[0])
    if any(len(channel) != sample_count for channel in channels):
        raise ValueError("EMG channels contain different sample counts")

    sample_index = np.asarray([sample.sampleIndex for sample in channels[0]], dtype=np.int64)
    sdk_timestamp_ms = np.asarray([sample.timeStampInMs for sample in channels[0]], dtype=np.int64)
    channel_indices = [np.asarray([sample.sampleIndex for sample in channel], dtype=np.int64) for channel in channels]
    if any(not np.array_equal(indices, sample_index) for indices in channel_indices[1:]):
        raise ValueError("EMG channels contain different sample indices")
    signal_uv = np.stack(
        [np.asarray([sample.data for sample in channel], dtype=np.float32) for channel in channels], axis=1
    )
    raw_adc = np.stack(
        [np.asarray([sample.rawData for sample in channel], dtype=np.int32) for channel in channels], axis=1
    )
    is_lost = np.logical_or.reduce(
        [np.asarray([sample.isLost for sample in channel], dtype=np.bool_) for channel in channels]
    )
    impedance = np.asarray([channel[-1].impedance for channel in channels], dtype=np.float32)
    saturation = np.asarray([channel[-1].saturation for channel in channels], dtype=np.float32)
    return EmgBatch(
        receive_monotonic_ns=receive_monotonic_ns,
        sample_index=sample_index,
        sdk_timestamp_ms=sdk_timestamp_ms,
        signal_uv=signal_uv,
        raw_adc=raw_adc,
        is_lost=is_lost,
        impedance_ohm=impedance,
        saturation=saturation,
    )


def _select_device(devices, requested: str | None):
    def describe(items) -> str:
        return ", ".join(
            f"{item.Name or '<unnamed>'} ({item.Address}, RSSI={item.RSSI})" for item in items
        ) or "none"

    supported = [
        device
        for device in devices
        if (device.Name or "").startswith(("OY", "Sync", "gForce"))
    ]
    if requested:
        needle = requested.casefold()
        matches = [
            device
            for device in devices
            if device.Address.casefold() == needle or needle in (device.Name or "").casefold()
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one EMG device matching {requested!r}, found {len(matches)}; "
                f"scanned devices: {describe(devices)}"
            )
        return matches[0]
    if len(supported) != 1:
        raise RuntimeError(
            f"Expected exactly one supported EMG device, found {len(supported)}; "
            f"scanned devices: {describe(devices)}"
        )
    return supported[0]


def _configure_hardware_filters(profile, *, hpf: bool, lpf: bool, notch: str) -> str:
    """Set and verify every firmware filter before notifications start."""

    if notch not in {"off", "50", "60", "both"}:
        raise ValueError(f"Unsupported EMG notch filter selection: {notch}")
    settings = {
        "FILTER_HPF": "ON" if hpf else "OFF",
        "FILTER_LPF": "ON" if lpf else "OFF",
        "FILTER_50HZ": "ON" if notch in {"50", "both"} else "OFF",
        "FILTER_60HZ": "ON" if notch in {"60", "both"} else "OFF",
    }
    for key, value in settings.items():
        result = profile.setParam(key, value)
        if str(result).strip().upper() != "OK":
            raise RuntimeError(f"Failed to configure {key}={value}: {result}")

    configuration = str(profile.getParam("FILTER"))
    if configuration.startswith("Error"):
        raise RuntimeError(f"Failed to read configured EMG filters: {configuration}")
    items = configuration.split("|")
    reported = dict(zip(items[0::2], items[1::2]))
    mismatches = [
        f"{key}: requested {value}, reported {reported.get(key, '<missing>')}"
        for key, value in settings.items()
        if reported.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(f"EMG firmware filter verification failed: {'; '.join(mismatches)}")
    return configuration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--preview-port", required=True, type=int)
    parser.add_argument("--device")
    parser.add_argument("--adapter")
    parser.add_argument("--sdk-path")
    parser.add_argument("--scan-ms", type=int, default=5000)
    parser.add_argument("--filter-hpf", choices=("on", "off"), default="on")
    parser.add_argument("--filter-lpf", choices=("on", "off"), default="on")
    parser.add_argument("--filter-notch", choices=("off", "50", "60", "both"), default="60")
    parser.add_argument("--visualize-decoder", action="store_true")
    parser.add_argument("--emg2pose-root")
    parser.add_argument("--emg2pose-checkpoint")
    parser.add_argument("--decoder-device", default="auto")
    parser.add_argument("--decoder-hand", choices=("left", "right"), default="right")
    parser.add_argument("--decoder-inference-hz", type=float, default=5.0)
    return parser


def run(args) -> int:
    publisher = _UdpPublisher(args.preview_host, args.preview_port)
    stop = threading.Event()

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.sdk_path:
        sdk_path = Path(args.sdk_path).expanduser().resolve()
        if not sdk_path.is_dir():
            publisher.send("error", message=f"Synchroni SDK path does not exist: {sdk_path}")
            publisher.close()
            return 2
        sys.path.insert(0, str(sdk_path))
    if args.adapter:
        os.environ["SYNCHRONI_BLE_ADAPTER"] = args.adapter

    try:
        import sensor as sensor_package
        from sensor import DataType, DeviceStateEx, SensorController
        from sensor import sensor_utils
    except Exception as error:
        publisher.send(
            "error",
            message=(
                "Synchroni SDK is unavailable in this Python environment. Install synchroni-sensor-sdk, bleak, "
                f"and flatbuffers, or pass --emg-sdk-path. Import failed: {error}"
            ),
        )
        publisher.close()
        return 2

    # Set this before SensorController starts its Linux BLE worker so both the
    # SDK process and its worker inherit the same command timeout.
    sensor_utils._TIMEOUT = max(sensor_utils._TIMEOUT, SDK_COMMAND_TIMEOUT_S)
    batches: queue.Queue[EmgBatch] = queue.Queue(maxsize=512)
    asynchronous_errors: queue.Queue[str] = queue.Queue()
    controller = SensorController()
    profile = None
    writer = None
    decoder = None
    decoder_runner = None
    first_batch = True
    last_decoder_status_time = 0.0

    def publish_decoder_result() -> None:
        nonlocal last_decoder_status_time
        if decoder_runner is None:
            return
        try:
            frame = decoder_runner.poll()
        except Exception as error:
            publisher.send("decoder_error", message=f"{type(error).__name__}: {error}")
            return
        if frame is not None:
            publisher.send(
                "decoder_frame",
                sequence=frame.sequence,
                source_sample_index=frame.source_sample_index,
                joint_angles=frame.joint_angles.tolist(),
                vertices=frame.vertices.tolist(),
                inference_ms=frame.inference_ms,
                status="live",
            )
        now = time.monotonic()
        if decoder_runner.buffered_samples < decoder_runner.required_samples and now - last_decoder_status_time >= 0.5:
            publisher.send(
                "decoder_status",
                status=(
                    f"buffering model window: {decoder_runner.buffered_samples}/"
                    f"{decoder_runner.required_samples} samples"
                ),
            )
            last_decoder_status_time = now

    try:
        if args.visualize_decoder:
            from dex_teleop.emg.decoder import DecoderRunner, Emg2PoseDecoder, default_emg2pose_root

            emg2pose_root = (
                Path(args.emg2pose_root).expanduser().resolve()
                if args.emg2pose_root
                else default_emg2pose_root()
            )
            publisher.send("status", status="loading EMG2Pose decoder", emg2pose_root=str(emg2pose_root))
            decoder = Emg2PoseDecoder(
                emg2pose_root,
                args.emg2pose_checkpoint,
                device_name=args.decoder_device,
                hand=args.decoder_hand,
            )
            publisher.send(
                "decoder_geometry",
                triangles=decoder.triangles.tolist(),
                status="waiting for EMG stream",
            )
            publisher.send(
                "decoder_frame",
                sequence=0,
                source_sample_index=None,
                joint_angles=np.zeros(20, dtype=np.float32).tolist(),
                vertices=decoder.neutral_vertices.tolist(),
                inference_ms=None,
                status="waiting for EMG stream",
            )

        publisher.send("status", status="scanning")
        devices = controller.scan(args.scan_ms)
        device = _select_device(devices, args.device)
        profile = controller.requireSensor(device)
        if profile is None:
            raise RuntimeError(f"Failed to create a sensor profile for {device.Name}")

        def on_data(_profile, data) -> None:
            if data is None or data.dataType != DataType.NTF_EMG or not data.channelSamples:
                return
            try:
                batches.put_nowait(_batch_from_sensor_data(data, time.monotonic_ns()))
            except queue.Full:
                asynchronous_errors.put("EMG writer queue is full; acquisition cannot remain lossless")
            except Exception as error:
                asynchronous_errors.put(f"Failed to copy EMG callback: {error}")

        def on_error(_profile, reason) -> None:
            asynchronous_errors.put(f"Synchroni SDK error: {reason}")

        def on_state_changed(_profile, state) -> None:
            if state == DeviceStateEx.Disconnected and not stop.is_set():
                asynchronous_errors.put("EMG device disconnected")

        profile.onDataCallback = on_data
        profile.onErrorCallback = on_error
        profile.onStateChanged = on_state_changed
        publisher.send("status", status="connecting", device_name=device.Name, device_address=device.Address)
        # The SDK's public connect() has a fixed 10-second outer timeout even
        # though its BLE worker may legitimately run for 25 seconds.  Use the
        # same SDK coroutine with a matching outer timeout until upstream fixes
        # that mismatch.
        if not sensor_utils.sync_call(profile._connect(), _timeout=SDK_CONNECT_TIMEOUT_S):
            raise RuntimeError(f"Failed to connect to {device.Name}")
        if not profile.hasInited:
            profile.setParam("NTF_IMU", "OFF")
            if not profile.init(PACKAGE_COUNT, POWER_REFRESH_PERIOD_MS):
                raise RuntimeError(f"Failed to initialize {device.Name}")
        info = profile.getDeviceInfo()
        if info.EmgChannelCount <= 0 or info.EmgSampleRate <= 0:
            raise RuntimeError(f"{device.Name} does not expose an EMG stream")

        filter_configuration = _configure_hardware_filters(
            profile,
            hpf=args.filter_hpf == "on",
            lpf=args.filter_lpf == "on",
            notch=args.filter_notch,
        )
        metadata = {
            "sdk_version": getattr(sensor_package, "__version__", "unknown"),
            "device_name": device.Name,
            "device_address": device.Address,
            "ble_adapter": args.adapter or "SDK default",
            "model_name": info.ModelName,
            "firmware_version": info.FirmwareVersion,
            "hardware_version": info.HardwareVersion,
            "filter_configuration": str(filter_configuration),
            "configured_filter_hpf": args.filter_hpf,
            "configured_filter_lpf": args.filter_lpf,
            "configured_filter_notch": args.filter_notch,
            "battery_refresh_period_ms": POWER_REFRESH_PERIOD_MS,
            "sdk_command_timeout_s": SDK_COMMAND_TIMEOUT_S,
            "decoder_enabled": bool(decoder is not None),
        }
        if decoder is not None:
            metadata.update(
                {
                    "decoder_checkpoint": str(decoder.checkpoint),
                    "decoder_device": decoder.device_name,
                    "decoder_left_context": decoder.left_context,
                    "decoder_inference_hz_target": args.decoder_inference_hz,
                }
            )
            decoder_runner = DecoderRunner(
                decoder,
                info.EmgSampleRate,
                inference_hz=args.decoder_inference_hz,
            )
            publisher.send(
                "decoder_status",
                status=f"buffering model window: 0/{decoder_runner.required_samples} samples",
            )
        writer = EmgHdf5Writer(
            args.output,
            sample_rate_hz=info.EmgSampleRate,
            channel_count=info.EmgChannelCount,
            metadata=metadata,
        )
        publisher.send(
            "status",
            status="streaming",
            sample_rate_hz=info.EmgSampleRate,
            channel_count=info.EmgChannelCount,
            **metadata,
        )
        if not profile.startDataNotification():
            raise RuntimeError("Failed to start EMG data notifications")

        while not stop.is_set():
            try:
                error = asynchronous_errors.get_nowait()
            except queue.Empty:
                error = None
            if error is not None:
                raise RuntimeError(error)
            try:
                batch = batches.get(timeout=0.1)
            except queue.Empty:
                publish_decoder_result()
                continue
            writer.append(batch)
            if decoder_runner is not None:
                decoder_runner.append(batch.signal_uv, int(batch.sample_index[-1]))
                publish_decoder_result()
            preview_start = max(0, batch.sample_count - PREVIEW_SAMPLE_LIMIT)
            publisher.send(
                "samples",
                receive_monotonic_ns=batch.receive_monotonic_ns,
                sample_index=batch.sample_index[preview_start:].tolist(),
                signal_uv=batch.signal_uv[preview_start:].tolist(),
                is_lost=batch.is_lost[preview_start:].tolist(),
                impedance_ohm=batch.impedance_ohm.tolist(),
            )
            if first_batch:
                first_batch = False
                publisher.send(
                    "ready",
                    sample_rate_hz=info.EmgSampleRate,
                    channel_count=info.EmgChannelCount,
                    **metadata,
                )

        publisher.send("status", status="stopping")
        return 0
    except Exception as error:
        publisher.send("error", message=str(error))
        traceback.print_exc()
        return 1
    finally:
        stop.set()
        if decoder_runner is not None:
            try:
                decoder_runner.close()
            except Exception:
                traceback.print_exc()
        if profile is not None:
            try:
                profile.disconnect()
            except Exception:
                traceback.print_exc()
        if writer is not None:
            try:
                while True:
                    writer.append(batches.get_nowait())
            except queue.Empty:
                pass
            except Exception:
                traceback.print_exc()
        try:
            controller.terminate()
        except Exception:
            traceback.print_exc()
        if writer is not None:
            try:
                writer.close()
            except Exception:
                traceback.print_exc()
        publisher.close()


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(_parser().parse_args(argv)))


if __name__ == "__main__":
    main()

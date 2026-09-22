"""Deterministic MANUS bridge protocol producer used by offline tests."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time


PROTOCOL = "dex_teleop.manus"
VERSION = 2
BRIDGE_MODE = "integrated"
CONNECTION_GENERATION = 1


def _event(event_type: str, **values):
    event = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "type": event_type,
        "mode": BRIDGE_MODE,
        "linked_mode": BRIDGE_MODE,
        **values,
    }
    if VERSION >= 2:
        event.setdefault("connection_generation", CONNECTION_GENERATION)
    return event


def _topology_nodes(side, *, thumb_ip_joint="distal"):
    nodes = [
        {"id": 0, "parent_id": 0, "side": side, "chain": "hand", "joint": "invalid"},
    ]
    node_id = 1
    for finger in ("thumb", "index", "middle", "ring", "little"):
        if finger == "thumb":
            # Live SDK 3.1.1 reports Distal, while its header documents the
            # same thumb IP node as Intermediate. Tests exercise both labels.
            joints = (
                ("metacarpal", 0),
                ("proximal", node_id),
                (thumb_ip_joint, node_id + 1),
                ("tip", node_id + 2),
            )
        else:
            joints = (
                ("metacarpal", 0),
                ("proximal", node_id),
                ("intermediate", node_id + 1),
                ("distal", node_id + 2),
                ("tip", node_id + 3),
            )
        for joint, parent_id in joints:
            nodes.append(
                {
                    "id": node_id,
                    "parent_id": parent_id,
                    "side": side,
                    "chain": finger,
                    "joint": joint,
                }
            )
            node_id += 1
    return nodes


def _sample_nodes():
    root = [1.0, 2.0, 3.0]
    root_quaternion = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    nodes = []
    for node_id in range(25):
        local = [node_id * 0.001, node_id * 0.002, -node_id * 0.001]
        # Global is root + a +90-degree rotation around Z.
        position = [root[0] - local[1], root[1] + local[0], root[2] + local[2]]
        nodes.append(
            {
                "id": node_id,
                "position": position,
                "orientation_xyzw": root_quaternion,
            }
        )
    return nodes


def _write_sample(protocol, side, sequence, *, legacy=False, generation=None):
    offset = sequence - 7
    event = _event(
        "articulation",
        sequence=sequence,
        glove_id=1417281806,
        side=side,
        manus_publish_time=987654321 + offset,
        capture_monotonic_ns=12_500_000_000 + offset,
        callback_capture_monotonic_ns=12_500_000_000 + offset,
        wrist_node_id=0,
        core_host_name=(
            "fake-windows-core" if BRIDGE_MODE == "remote" else "fake-integrated"
        ),
        core_host_ip="192.0.2.10" if BRIDGE_MODE == "remote" else "127.0.0.1",
        sdk_version="3.1.1",
        core_version="3.1.1",
        versions_compatible=True,
        session_id=42,
        hand_motion="tracker" if BRIDGE_MODE == "remote" else "auto",
        core_world_frame="manus_core_world_y_up_rh_z_to_viewer_m",
        nodes=_sample_nodes(),
    )
    if legacy:
        for key in (
            "mode",
            "linked_mode",
            "callback_capture_monotonic_ns",
            "wrist_node_id",
            "core_host_name",
            "core_host_ip",
            "sdk_version",
            "core_version",
            "versions_compatible",
            "session_id",
            "hand_motion",
            "core_world_frame",
        ):
            event.pop(key, None)
    if generation is not None:
        event["connection_generation"] = generation
    protocol.write(json.dumps(event) + "\n")


def main():
    global BRIDGE_MODE, CONNECTION_GENERATION, VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-fd", type=int)
    parser.add_argument("--protocol-version", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=(
            "normal",
            "legacy_protocol",
            "no_glove",
            "bad_topology",
            "documented_thumb_intermediate",
            "duplicate_thumb_ip_alias",
            "duplicate_side_topology",
            "overflow",
            "topology_only",
            "remove_after_sample",
            "disconnect_after_sample",
            "reconnect_after_sample",
            "hand_motion_error",
            "integrated_license_error",
            "sdk_license_error",
            "sample_after_disconnect",
            "sample_before_reconnected",
            "reconnect_no_glove",
            "tracker_degrades",
        ),
        default="normal",
    )
    parser.add_argument(
        "--mode", choices=("integrated", "remote"), default="integrated"
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--connect-timeout")
    parser.add_argument("--glove-timeout")
    parser.add_argument("--reconnect-timeout")
    parser.add_argument("--discovery-wait")
    parser.add_argument("--core-host")
    parser.add_argument("--loopback-only", action="store_true")
    parser.add_argument("--hand-motion")
    parser.add_argument("--required-hand")
    parser.add_argument("--tracker-diagnostics", action="store_true")
    parser.add_argument("--no-tracker-diagnostics", action="store_true")
    parser.add_argument("--settings-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--left-calibration")
    parser.add_argument("--right-calibration")
    args = parser.parse_args()
    BRIDGE_MODE = args.mode
    VERSION = 1 if args.scenario == "legacy_protocol" else 2
    CONNECTION_GENERATION = 1
    if args.protocol_version:
        print(f"{PROTOCOL} {VERSION}")
        return
    if args.protocol_fd is None:
        parser.error("--protocol-fd is required")
    protocol = os.fdopen(args.protocol_fd, "w", encoding="utf-8", buffering=1)

    # This must never contaminate the dedicated protocol descriptor.
    print(
        'vendor console noise; license data: {"Key":"must-not-be-parsed"}', flush=True
    )
    protocol.write(
        json.dumps(_event("status", state="starting", message="fake")) + "\n"
    )
    if args.scenario == "no_glove":
        protocol.write(
            json.dumps(
                _event("status", state="sdk_connected", message="fake connected")
            )
            + "\n"
        )
        # Model an empty early Landscape followed by the native glove timeout.
        time.sleep(0.05)
        protocol.write(
            json.dumps(
                _event("error", code="no_glove", message="No MANUS glove connected")
            )
            + "\n"
        )
        return
    if args.scenario == "hand_motion_error":
        protocol.write(
            json.dumps(
                _event(
                    "error",
                    code="hand_motion_configuration_failed",
                    message="CoreSdk_SetRawSkeletonHandMotion failed",
                    sdk_return_code=7,
                )
            )
            + "\n"
        )
        return
    if args.scenario in {"integrated_license_error", "sdk_license_error"}:
        code = (
            "sdk_license_unavailable"
            if args.scenario == "sdk_license_error"
            else "integrated_license_unavailable"
        )
        protocol.write(
            json.dumps(
                _event(
                    "error",
                    code=code,
                    message=f"The connected MANUS license does not enable the {args.mode} SDK feature",
                )
            )
            + "\n"
        )
        return
    if args.mode == "remote":
        protocol.write(
            json.dumps(
                _event(
                    "hosts",
                    selector=args.core_host,
                    selected={
                        "name": "fake-windows-core",
                        "ip": "192.0.2.10",
                    },
                    hosts=[
                        {
                            "index": 0,
                            "name": "fake-windows-core",
                            "ip": "192.0.2.10",
                        }
                    ],
                )
            )
            + "\n"
        )
    protocol.write(
        json.dumps(
            _event(
                "status",
                state="sdk_connected",
                core_host_name=(
                    "fake-windows-core" if args.mode == "remote" else "fake-integrated"
                ),
                core_host_ip="192.0.2.10" if args.mode == "remote" else "127.0.0.1",
                sdk_version="3.1.1",
                core_version="3.1.1",
                versions_compatible=True,
                session_id=42,
                hand_motion="tracker" if args.mode == "remote" else "auto",
                core_world_frame="manus_core_world_y_up_rh_z_to_viewer_m",
            )
        )
        + "\n"
    )
    protocol.write(
        json.dumps(
            _event(
                "landscape",
                core_version="3.1.1",
                license_sdk=args.mode == "remote",
                license_integrated=args.mode == "integrated",
                trackers=[
                    {
                        "id": "ultimate-right",
                        "type": "right_hand",
                        "user_id": 1,
                    }
                ],
            )
        )
        + "\n"
    )

    thumb_ip_joint = (
        "intermediate" if args.scenario == "documented_thumb_intermediate" else "distal"
    )
    nodes = _topology_nodes(args.side, thumb_ip_joint=thumb_ip_joint)
    if args.scenario == "bad_topology":
        nodes.pop()
    elif args.scenario == "duplicate_thumb_ip_alias":
        nodes[-1].update(chain="thumb", joint="intermediate")
    protocol.write(
        json.dumps(
            _event(
                "topology",
                glove_id=1417281806,
                side=args.side,
                transform_space="sdk_global",
                nodes=nodes,
            )
        )
        + "\n"
    )
    if args.scenario == "duplicate_side_topology":
        protocol.write(
            json.dumps(
                _event(
                    "topology",
                    glove_id=1417281807,
                    side=args.side,
                    transform_space="sdk_global",
                    nodes=nodes,
                )
            )
            + "\n"
        )
        time.sleep(0.1)
        return
    if args.scenario in {"bad_topology", "duplicate_thumb_ip_alias"}:
        time.sleep(0.1)
        return

    if args.scenario != "topology_only":
        if args.tracker_diagnostics:
            protocol.write(
                json.dumps(
                    _event(
                        "trackers",
                        sequence=1,
                        manus_publish_time=987654321,
                        capture_monotonic_ns=12_500_000_000,
                        session_id=42,
                        trackers=[
                            {
                                "id": "ultimate-right",
                                "type": "right_hand",
                                "user_id": 1,
                                "is_hmd": False,
                                "quality": "trackable",
                                "tracking_system": "openvr",
                                "last_update_time": 100,
                                "pose_valid": True,
                                "position": [1.0, 2.0, 3.0],
                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            }
                        ],
                    )
                )
                + "\n"
            )
    protocol.write(
        json.dumps(_event("status", state="topology_ready", message="fake ready"))
        + "\n"
    )
    if args.scenario != "topology_only":
        _write_sample(
            protocol,
            args.side,
            7,
            legacy=args.scenario == "legacy_protocol",
        )
        if args.scenario == "tracker_degrades" and args.tracker_diagnostics:
            time.sleep(0.05)
            protocol.write(
                json.dumps(
                    _event(
                        "trackers",
                        sequence=2,
                        manus_publish_time=987654322,
                        capture_monotonic_ns=12_500_000_001,
                        session_id=42,
                        trackers=[
                            {
                                "id": "ultimate-right",
                                "type": "right_hand",
                                "user_id": 1,
                                "is_hmd": False,
                                "quality": "untrackable",
                                "tracking_system": "openvr",
                                "last_update_time": 101,
                                "pose_valid": True,
                                "position": [1.0, 2.0, 3.0],
                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            }
                        ],
                    )
                )
                + "\n"
            )

    if args.scenario == "overflow":
        # Give the test consumer time to opt into lossless draining, then
        # overrun its deliberately tiny queue.
        time.sleep(0.1)
        for sequence in range(8, 11):
            _write_sample(protocol, args.side, sequence)
    elif args.scenario == "remove_after_sample":
        time.sleep(0.05)
        protocol.write(
            json.dumps(
                _event(
                    "glove_removed",
                    glove_id=1417281806,
                    side=args.side,
                    reason="landscape_absent",
                )
            )
            + "\n"
        )
    elif args.scenario in {
        "disconnect_after_sample",
        "sample_after_disconnect",
        "sample_before_reconnected",
        "reconnect_no_glove",
    }:
        time.sleep(0.05)
        CONNECTION_GENERATION = 2
        protocol.write(
            json.dumps(
                _event(
                    "status",
                    state="disconnected",
                    message="fake host disconnected",
                    recoverable=args.scenario != "disconnect_after_sample",
                )
            )
            + "\n"
        )
        if args.scenario == "sample_after_disconnect":
            _write_sample(protocol, args.side, 8, generation=1)
        elif args.scenario in {"sample_before_reconnected", "reconnect_no_glove"}:
            protocol.write(
                json.dumps(
                    _event("status", state="reconnecting", message="fake reconnecting")
                )
                + "\n"
            )
            if args.scenario == "reconnect_no_glove":
                protocol.write(
                    json.dumps(
                        _event(
                            "error",
                            code="reconnect_stream_not_qualified",
                            message="required glove did not recover",
                        )
                    )
                    + "\n"
                )
            else:
                protocol.write(
                    json.dumps(
                        _event(
                            "topology",
                            glove_id=1417281806,
                            side=args.side,
                            transform_space="sdk_global",
                            nodes=nodes,
                        )
                    )
                    + "\n"
                )
                _write_sample(protocol, args.side, 8)
                protocol.write(
                    json.dumps(
                        _event(
                            "status", state="reconnected", message="fake reconnected"
                        )
                    )
                    + "\n"
                )
                _write_sample(protocol, args.side, 9)
    elif args.scenario == "reconnect_after_sample":
        time.sleep(0.05)
        CONNECTION_GENERATION = 2
        protocol.write(
            json.dumps(
                _event(
                    "status",
                    state="disconnected",
                    message="fake recoverable disconnect",
                    recoverable=True,
                )
            )
            + "\n"
        )
        protocol.write(
            json.dumps(
                _event("status", state="reconnecting", message="fake reconnecting")
            )
            + "\n"
        )
        protocol.write(
            json.dumps(
                _event(
                    "topology",
                    glove_id=1417281806,
                    side=args.side,
                    transform_space="sdk_global",
                    nodes=nodes,
                )
            )
            + "\n"
        )
        protocol.write(
            json.dumps(
                _event("status", state="reconnected", message="fake reconnected")
            )
            + "\n"
        )
        _write_sample(protocol, args.side, 8)

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        time.sleep(0.02)


if __name__ == "__main__":
    main()

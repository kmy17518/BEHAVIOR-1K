"""MANUS SDK articulation and same-callback global-wrist source.

The proprietary SDK runs in a small native sidecar.  This module consumes its
versioned JSON-lines protocol and publishes the validated 25-node MANUS hand as
wrist-local named articulation plus the preserved global wrist. Integrated
remains the default gloves-only deployment; Remote connects from Linux to
MANUS Core on Windows and is the production glove + tracker path.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dex_teleop.tracking.base import SourceUnavailableError
from dex_teleop.tracking.manus_calibration import (
    MANUS_BAKED_WRIST_SEMANTICS,
    MANUS_CORE_CALIBRATION_SCHEMA_VERSION,
    MANUS_CORE_WORLD_FRAME,
    ManusCoreCalibration,
)
from dex_teleop.tracking.multimodal import HandTrackingSampleBatch
from dex_teleop.tracking.transforms import RigidTransform, inverse_transform
from dex_teleop.types import (
    HandArticulationSample,
    Handedness,
    OPENXR_ANATOMICAL_WRIST_FRAME,
    OPENXR_HAND_JOINT_NAMES,
    WristPoseSample,
)


LOGGER = logging.getLogger(__name__)

MANUS_PROTOCOL = "dex_teleop.manus"
MANUS_PROTOCOL_VERSION = 2
MANUS_LEGACY_PROTOCOL_VERSION = 1
MANUS_ARTICULATION_SCHEMA = "openxr25_no_palm"
MANUS_JOINT_NAMES = tuple(name for name in OPENXR_HAND_JOINT_NAMES if name != "palm")
MANUS_MODES = ("integrated", "remote")
MANUS_HAND_MOTIONS = ("auto", "tracker", "tracker_rotation_only", "imu", "none")
MANUS_ULTIMATE_TRACKER_SYSTEM = "openvr"


class ManusProtocolError(ValueError):
    """Raised when the native bridge violates the versioned wire contract."""


@dataclass(frozen=True)
class ManusSdkLayout:
    """One matching MANUS header plus both available SDK variants."""

    root: Path
    include_dir: Path
    integrated_library: Path
    remote_library: Path | None = None

    def library_for(self, mode: str) -> Path:
        mode = _validate_mode(mode)
        if mode == "integrated":
            return self.integrated_library
        if self.remote_library is None:
            raise SourceUnavailableError(
                f"MANUS SDK layout {self.root} has no Remote libManusSDK.so"
            )
        return self.remote_library


@dataclass(frozen=True)
class ManusFrameDiagnostics:
    """Native timing and device identity retained outside the canonical sample."""

    sequence: int
    glove_id: int
    manus_publish_time: int
    capture_monotonic_ns: int
    receipt_monotonic_ns: int
    mode: str = "integrated"
    core_host_name: str = ""
    core_host_ip: str = ""
    core_version: str = ""
    sdk_version: str = ""
    hand_motion: str = "auto"
    connection_generation: int = 0


@dataclass(frozen=True)
class ManusTrackerDiagnostics:
    """Latest optional Core tracker-stream packet for hardware diagnostics."""

    sequence: int
    manus_publish_time: int
    capture_monotonic_ns: int
    receipt_monotonic_ns: int
    connection_generation: int
    session_id: int
    trackers: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _TopologyNode:
    node_id: int
    parent_id: int
    name: str


@dataclass(frozen=True)
class _GloveTopology:
    glove_id: int
    handedness: Handedness
    nodes_by_id: Mapping[int, _TopologyNode]


def _validate_mode(value: str) -> str:
    mode = str(value).casefold()
    if mode not in MANUS_MODES:
        raise ValueError(f"MANUS mode must be one of {MANUS_MODES}, got {value!r}")
    return mode


def _validate_hand_motion(value: str) -> str:
    motion = str(value).casefold()
    if motion not in MANUS_HAND_MOTIONS:
        raise ValueError(
            f"MANUS hand motion must be one of {MANUS_HAND_MOTIONS}, got {value!r}"
        )
    return motion


def _protocol_capabilities(version: int) -> tuple[str, ...]:
    if version == MANUS_LEGACY_PROTOCOL_VERSION:
        return (
            "integrated_wire_v1",
            "python_required_hand_filter",
            "same_callback_pairing",
        )
    if version == MANUS_PROTOCOL_VERSION:
        return (
            "connection_generation",
            "native_required_hand",
            "reconnect_requalification",
            "same_callback_pairing",
            "tracker_health_metadata",
        )
    raise ValueError(f"Unsupported MANUS protocol version {version}")


def _candidate_manus_roots(explicit_root: str | Path | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root).expanduser())
    environment_root = os.environ.get("MANUS_SDK_ROOT")
    if environment_root:
        candidates.append(Path(environment_root).expanduser())
    candidates.extend(
        (
            Path.home() / "Desktop/emg/manus",
            Path.home() / "manus_setup",
        )
    )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _sdk_directories(root: Path) -> tuple[Path, ...]:
    candidates = [
        root,
        root / "ManusSDK",
        root / "SDKClient_Linux/ManusSDK",
        root / "SDKMinimalClient_Linux/ManusSDK",
    ]
    vendor = root / "vendor"
    if vendor.is_dir():
        candidates.extend(sorted(vendor.glob("ManusSDK_v*/SDKClient_Linux/ManusSDK")))
        candidates.extend(
            sorted(vendor.glob("ManusSDK_v*/SDKMinimalClient_Linux/ManusSDK"))
        )
        candidates.extend(sorted(vendor.glob("ManusSDK_v*/ROS2/ManusSDK")))
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def discover_manus_sdk(
    explicit_root: str | Path | None = None,
    *,
    mode: str = "integrated",
) -> ManusSdkLayout:
    """Find a matching official header and requested SDK library variant."""

    mode = _validate_mode(mode)
    searched: list[Path] = []
    for root in _candidate_manus_roots(explicit_root):
        if not root.is_dir():
            searched.append(root)
            continue
        for candidate in _sdk_directories(root):
            searched.append(candidate)
            header = candidate / "include/ManusSDK.h"
            integrated_library = candidate / "lib/libManusSDK_Integrated.so"
            remote_library = candidate / "lib/libManusSDK.so"
            selected_library = (
                integrated_library if mode == "integrated" else remote_library
            )
            if header.is_file() and selected_library.is_file():
                return ManusSdkLayout(
                    root=candidate,
                    include_dir=header.parent,
                    integrated_library=integrated_library,
                    remote_library=remote_library if remote_library.is_file() else None,
                )
    locations = "\n".join(f"  - {path}" for path in searched)
    expected = (
        "libManusSDK_Integrated.so" if mode == "integrated" else "Remote libManusSDK.so"
    )
    raise SourceUnavailableError(
        f"Could not find a matching MANUS SDK header and {expected}. "
        "Set MANUS_SDK_ROOT or pass sdk_root. Searched:\n" + locations
    )


def _native_bridge_directory() -> Path:
    # Keep native sources inside the Python package so wheel installs retain
    # the optional build helper without bundling the proprietary SDK itself.
    return Path(__file__).resolve().parents[1] / "native/manus_bridge"


def default_manus_bridge_path(mode: str = "integrated") -> Path:
    mode = _validate_mode(mode)
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    filename = "manus_bridge" if mode == "integrated" else "manus_bridge_remote"
    return cache_root / "dex_teleop/manus_bridge" / filename


def build_manus_bridge(
    *,
    sdk_root: str | Path | None = None,
    output: str | Path | None = None,
    mode: str = "integrated",
) -> Path:
    """Build the optional native bridge against the separately installed SDK."""

    mode = _validate_mode(mode)
    layout = discover_manus_sdk(sdk_root, mode=mode)
    output_path = (
        default_manus_bridge_path(mode)
        if output is None
        else Path(output).expanduser().resolve()
    )
    build_script = _native_bridge_directory() / "build_manus_bridge.sh"
    if not build_script.is_file():
        raise SourceUnavailableError(
            f"MANUS bridge build helper is missing: {build_script}"
        )
    command = [
        "bash",
        str(build_script),
        "--mode",
        mode,
        "--sdk-root",
        str(layout.root),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SourceUnavailableError(
            f"Failed to build the optional MANUS {mode} bridge (exit {result.returncode}): {detail}"
        )
    if not output_path.is_file():
        raise SourceUnavailableError(
            f"MANUS bridge build reported success but did not create {output_path}"
        )
    return output_path


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManusProtocolError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManusProtocolError(f"{context} must be a JSON array")
    return value


def _require_int(value: Any, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManusProtocolError(f"{context} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ManusProtocolError(f"{context} must be {qualifier}")
    return value


def _vector(value: Any, size: int, context: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ManusProtocolError(f"{context} must contain numeric values") from error
    if vector.shape != (size,):
        raise ManusProtocolError(
            f"{context} must contain {size} values, got shape {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ManusProtocolError(f"{context} contains a non-finite value")
    return vector


def _normalize_quaternion(quaternion_xyzw: np.ndarray, context: str) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm <= 0.0:
        raise ManusProtocolError(f"{context} has zero norm")
    return quaternion_xyzw / norm


def _quaternion_multiply(left_xyzw: np.ndarray, right_xyzw: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left_xyzw
    rx, ry, rz, rw = right_xyzw
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _rotate(vector: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    xyz = quaternion_xyzw[:3]
    w = quaternion_xyzw[3]
    cross = 2.0 * np.cross(xyz, vector)
    return vector + w * cross + np.cross(xyz, cross)


def _semantic_joint_name(chain: Any, joint: Any) -> str:
    if not isinstance(chain, str) or not isinstance(joint, str):
        raise ManusProtocolError("MANUS topology chain and joint must be strings")
    if chain == "hand" and joint == "invalid":
        return "wrist"
    if chain not in {"thumb", "index", "middle", "ring", "little"}:
        raise ManusProtocolError(f"Unsupported MANUS topology chain: {chain!r}")
    allowed = {"metacarpal", "proximal", "intermediate", "distal", "tip"}
    if joint not in allowed:
        raise ManusProtocolError(f"Unsupported MANUS topology joint: {joint!r}")
    if chain == "thumb" and joint in {"intermediate", "distal"}:
        # The SDK 3.1.1 header documents the thumb IP node as ``Intermediate``,
        # while a live Metaglove Pro Haptics reports the same node as
        # ``Distal``. OpenXR calls this single anatomical node
        # ``thumb_distal``. If a malformed topology contains both aliases, the
        # duplicate semantic-name check in _parse_topology rejects it.
        return "thumb_distal"
    return f"{chain}_{joint}"


def _validate_parent_graph(nodes_by_id: Mapping[int, _TopologyNode]) -> None:
    ids_by_name = {node.name: node.node_id for node in nodes_by_id.values()}
    expected_parent: dict[str, str] = {"wrist": "wrist"}
    expected_parent.update(
        {
            "thumb_metacarpal": "wrist",
            "thumb_proximal": "thumb_metacarpal",
            "thumb_distal": "thumb_proximal",
            "thumb_tip": "thumb_distal",
        }
    )
    for finger in ("index", "middle", "ring", "little"):
        expected_parent.update(
            {
                f"{finger}_metacarpal": "wrist",
                f"{finger}_proximal": f"{finger}_metacarpal",
                f"{finger}_intermediate": f"{finger}_proximal",
                f"{finger}_distal": f"{finger}_intermediate",
                f"{finger}_tip": f"{finger}_distal",
            }
        )
    for name, parent_name in expected_parent.items():
        node = nodes_by_id[ids_by_name[name]]
        if node.parent_id != ids_by_name[parent_name]:
            raise ManusProtocolError(
                f"MANUS topology parent mismatch for {name}: got {node.parent_id}, "
                f"expected node {ids_by_name[parent_name]} ({parent_name})"
            )


def _parse_topology(payload: Mapping[str, Any]) -> _GloveTopology:
    glove_id = _require_int(payload.get("glove_id"), "topology.glove_id", positive=True)
    try:
        handedness = Handedness(payload.get("side"))
    except ValueError as error:
        raise ManusProtocolError(
            f"Invalid MANUS topology side: {payload.get('side')!r}"
        ) from error
    if payload.get("transform_space") != "sdk_global":
        raise ManusProtocolError(
            "MANUS topology must declare transform_space='sdk_global'"
        )

    raw_nodes = _require_list(payload.get("nodes"), "topology.nodes")
    if len(raw_nodes) != len(MANUS_JOINT_NAMES):
        raise ManusProtocolError(
            f"MANUS topology must contain {len(MANUS_JOINT_NAMES)} nodes, got {len(raw_nodes)}"
        )
    nodes_by_id: dict[int, _TopologyNode] = {}
    names: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        node = _require_mapping(raw_node, f"topology.nodes[{index}]")
        node_id = _require_int(node.get("id"), f"topology.nodes[{index}].id")
        parent_id = _require_int(
            node.get("parent_id"), f"topology.nodes[{index}].parent_id"
        )
        if node.get("side") != handedness.value:
            raise ManusProtocolError(
                f"MANUS topology node {node_id} has a mismatched side"
            )
        name = _semantic_joint_name(node.get("chain"), node.get("joint"))
        if node_id in nodes_by_id:
            raise ManusProtocolError(
                f"MANUS topology contains duplicate node ID {node_id}"
            )
        if name in names:
            raise ManusProtocolError(
                f"MANUS topology contains duplicate joint {name!r}"
            )
        nodes_by_id[node_id] = _TopologyNode(
            node_id=node_id, parent_id=parent_id, name=name
        )
        names.add(name)
    if names != set(MANUS_JOINT_NAMES):
        raise ManusProtocolError(
            "MANUS topology does not match the validated 25-node Metaglove layout; "
            f"missing={sorted(set(MANUS_JOINT_NAMES) - names)}, extra={sorted(names - set(MANUS_JOINT_NAMES))}"
        )
    _validate_parent_graph(nodes_by_id)
    return _GloveTopology(
        glove_id=glove_id, handedness=handedness, nodes_by_id=nodes_by_id
    )


def _parse_articulation(
    payload: Mapping[str, Any],
    topology: _GloveTopology,
    *,
    receipt_monotonic_ns: int,
    mode: str = "integrated",
    calibration: ManusCoreCalibration | None = None,
    calibration_sha256: str | None = None,
    bridge_metadata: Mapping[str, Any] | None = None,
) -> tuple[HandArticulationSample, WristPoseSample, ManusFrameDiagnostics]:
    mode = _validate_mode(mode)
    glove_id = _require_int(
        payload.get("glove_id"), "articulation.glove_id", positive=True
    )
    if glove_id != topology.glove_id:
        raise ManusProtocolError(
            "MANUS articulation glove ID does not match its topology"
        )
    if payload.get("side") != topology.handedness.value:
        raise ManusProtocolError("MANUS articulation side does not match its topology")
    sequence = _require_int(payload.get("sequence"), "articulation.sequence")
    connection_generation = (
        0
        if "connection_generation" not in payload
        else _require_int(
            payload.get("connection_generation"),
            "articulation.connection_generation",
            positive=True,
        )
    )
    manus_publish_time = _require_int(
        payload.get("manus_publish_time"), "articulation.manus_publish_time"
    )
    capture_monotonic_ns = _require_int(
        payload.get("capture_monotonic_ns"),
        "articulation.capture_monotonic_ns",
        positive=True,
    )
    callback_capture = payload.get("callback_capture_monotonic_ns")
    if (
        callback_capture is not None
        and _require_int(
            callback_capture,
            "articulation.callback_capture_monotonic_ns",
            positive=True,
        )
        != capture_monotonic_ns
    ):
        raise ManusProtocolError(
            "MANUS callback capture time differs from articulation capture time"
        )
    payload_mode = payload.get("mode")
    if payload_mode is not None and payload_mode != mode:
        raise ManusProtocolError(
            f"MANUS bridge mode {payload_mode!r} does not match requested mode {mode!r}"
        )
    core_world_frame = payload.get("core_world_frame", MANUS_CORE_WORLD_FRAME)
    if core_world_frame != MANUS_CORE_WORLD_FRAME:
        raise ManusProtocolError(
            f"Unsupported MANUS Core world frame {core_world_frame!r}; "
            f"expected {MANUS_CORE_WORLD_FRAME!r}"
        )
    raw_nodes = _require_list(payload.get("nodes"), "articulation.nodes")
    if len(raw_nodes) != len(topology.nodes_by_id):
        raise ManusProtocolError(
            f"MANUS articulation contains {len(raw_nodes)} nodes for a {len(topology.nodes_by_id)}-node topology"
        )

    global_positions: dict[str, np.ndarray] = {}
    global_orientations: dict[str, np.ndarray] = {}
    seen_ids: set[int] = set()
    for index, raw_node in enumerate(raw_nodes):
        node = _require_mapping(raw_node, f"articulation.nodes[{index}]")
        node_id = _require_int(node.get("id"), f"articulation.nodes[{index}].id")
        if node_id in seen_ids:
            raise ManusProtocolError(
                f"MANUS articulation contains duplicate node ID {node_id}"
            )
        topology_node = topology.nodes_by_id.get(node_id)
        if topology_node is None:
            raise ManusProtocolError(
                f"MANUS articulation contains unknown node ID {node_id}"
            )
        seen_ids.add(node_id)
        global_positions[topology_node.name] = _vector(
            node.get("position"), 3, f"articulation.nodes[{index}].position"
        )
        global_orientations[topology_node.name] = _normalize_quaternion(
            _vector(
                node.get("orientation_xyzw"),
                4,
                f"articulation.nodes[{index}].orientation_xyzw",
            ),
            f"articulation.nodes[{index}].orientation_xyzw",
        )
    if seen_ids != set(topology.nodes_by_id):
        raise ManusProtocolError("MANUS articulation is missing topology node IDs")

    wrist_position = global_positions["wrist"]
    wrist_orientation = global_orientations["wrist"]
    wrist_node_id = next(
        node.node_id for node in topology.nodes_by_id.values() if node.name == "wrist"
    )
    declared_wrist_node = payload.get("wrist_node_id")
    if (
        declared_wrist_node is not None
        and _require_int(declared_wrist_node, "articulation.wrist_node_id")
        != wrist_node_id
    ):
        raise ManusProtocolError(
            "MANUS articulation wrist_node_id does not match the validated topology"
        )
    inverse_wrist = wrist_orientation * np.array([-1.0, -1.0, -1.0, 1.0])
    local_positions = {
        name: _rotate(global_positions[name] - wrist_position, inverse_wrist)
        for name in MANUS_JOINT_NAMES
    }
    local_orientations = {
        name: _normalize_quaternion(
            _quaternion_multiply(inverse_wrist, global_orientations[name]),
            f"local orientation for {name}",
        )
        for name in MANUS_JOINT_NAMES
    }
    core_wrist = RigidTransform(wrist_position, wrist_orientation)
    if calibration is None:
        reference_wrist = core_wrist
        reference_frame = MANUS_CORE_WORLD_FRAME
        wrist_calibration = None
    else:
        try:
            wrist_calibration = calibration.wrist(topology.handedness)
            reference_wrist = calibration.apply(topology.handedness, core_wrist)
        except KeyError as error:
            raise ManusProtocolError(str(error)) from error
        reference_frame = calibration.reference_frame
        anatomical_from_skeleton = inverse_transform(
            wrist_calibration.skeleton_wrist_to_anatomical_wrist
        )
        local_positions = {
            name: (
                _rotate(
                    position,
                    anatomical_from_skeleton.quaternion_xyzw,
                )
                + anatomical_from_skeleton.translation
            )
            for name, position in local_positions.items()
        }
        local_orientations = {
            name: _normalize_quaternion(
                _quaternion_multiply(
                    anatomical_from_skeleton.quaternion_xyzw,
                    orientation,
                ),
                f"anatomical local orientation for {name}",
            )
            for name, orientation in local_orientations.items()
        }
    metadata = dict(bridge_metadata or {})
    for key in (
        "linked_mode",
        "core_host_name",
        "core_host_ip",
        "sdk_version",
        "core_version",
        "versions_compatible",
        "session_id",
        "hand_motion",
    ):
        if key in payload:
            metadata[key] = payload[key]
    hand_motion = metadata.get("hand_motion", "auto")
    if hand_motion not in MANUS_HAND_MOTIONS:
        raise ManusProtocolError(f"Invalid MANUS hand_motion metadata: {hand_motion!r}")
    protocol_version = metadata.get("bridge_protocol_version")
    protocol_version_history = metadata.get(
        "protocol_version_history",
        () if protocol_version is None else (protocol_version,),
    )
    common_provenance: dict[str, Any] = {
        "glove_id": glove_id,
        "manus_publish_time": manus_publish_time,
        "callback_capture_monotonic_ns": capture_monotonic_ns,
        "sequence": sequence,
        "connection_generation": connection_generation,
        "reconnect_count": metadata.get("reconnect_count", 0),
        "bridge_protocol_version": protocol_version,
        "supported_protocol_version": metadata.get(
            "supported_protocol_version", MANUS_PROTOCOL_VERSION
        ),
        # Sample provenance intentionally contains JSON scalars. Preserve the
        # immutable history snapshot as canonical JSON rather than sharing the
        # source's mutable list.
        "protocol_version_history": json.dumps(
            list(protocol_version_history), separators=(",", ":")
        ),
        "bridge_capabilities": ",".join(metadata.get("bridge_capabilities", ())),
        "mode": mode,
        "linked_mode": metadata.get("linked_mode"),
        "core_host_name": metadata.get("core_host_name", ""),
        "core_host_ip": metadata.get("core_host_ip", ""),
        "sdk_version": metadata.get("sdk_version", ""),
        "core_version": metadata.get("core_version", ""),
        "versions_compatible": metadata.get("versions_compatible"),
        "session_id": metadata.get("session_id"),
        "hand_motion": hand_motion,
        "core_world_frame": core_world_frame,
        "calibration_sha256": calibration_sha256,
        "calibration_schema_version": (
            None if calibration is None else MANUS_CORE_CALIBRATION_SCHEMA_VERSION
        ),
        "calibration_reference_frame": (
            None if calibration is None else calibration.reference_frame
        ),
    }
    source_id = f"manus_{mode}"
    articulation = HandArticulationSample(
        timestamp=capture_monotonic_ns / 1e9,
        receipt_timestamp=receipt_monotonic_ns / 1e9,
        # The native sidecar and Python share Linux CLOCK_MONOTONIC. Preserve
        # the exact callback-entry value in addition to its seconds form.
        source_timestamp_ns=capture_monotonic_ns,
        source_frame_id=sequence,
        handedness=topology.handedness,
        joint_positions=local_positions,
        joint_orientations_xyzw=local_orientations,
        joint_validity={name: True for name in MANUS_JOINT_NAMES},
        source=source_id,
        schema=MANUS_ARTICULATION_SCHEMA,
        coordinate_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
        provenance=common_provenance,
        confidence=None,
    )

    wrist_provenance = {
        **common_provenance,
        "wrist_node_id": wrist_node_id,
        "wrist_pose_semantics": MANUS_BAKED_WRIST_SEMANTICS,
        "core_tracker_offset_applied": (
            None
            if wrist_calibration is None
            else wrist_calibration.core_tracker_offset_applied
        ),
        "tracker_id": None
        if wrist_calibration is None
        else wrist_calibration.tracker_id,
        "tracker_offset_preset": (
            None
            if wrist_calibration is None
            else wrist_calibration.tracker_offset_preset
        ),
    }
    wrist = WristPoseSample(
        timestamp=capture_monotonic_ns / 1e9,
        receipt_timestamp=receipt_monotonic_ns / 1e9,
        source_timestamp_ns=capture_monotonic_ns,
        source_frame_id=sequence,
        handedness=topology.handedness,
        position=reference_wrist.translation,
        quaternion_xyzw=reference_wrist.quaternion_xyzw,
        source=source_id,
        reference_frame=reference_frame,
        anatomical_frame=OPENXR_ANATOMICAL_WRIST_FRAME,
        provenance=wrist_provenance,
        confidence=None,
    )
    diagnostics = ManusFrameDiagnostics(
        sequence=sequence,
        glove_id=glove_id,
        manus_publish_time=manus_publish_time,
        capture_monotonic_ns=capture_monotonic_ns,
        receipt_monotonic_ns=receipt_monotonic_ns,
        mode=mode,
        core_host_name=str(metadata.get("core_host_name", "")),
        core_host_ip=str(metadata.get("core_host_ip", "")),
        core_version=str(metadata.get("core_version", "")),
        sdk_version=str(metadata.get("sdk_version", "")),
        hand_motion=str(hand_motion),
        connection_generation=connection_generation,
    )
    return articulation, wrist, diagnostics


class ManusIntegratedSource:
    """Lifecycle-managed Integrated/Remote MANUS source.

    The historical class name is retained for import compatibility. ``mode``
    defaults to ``integrated``; ``ManusSource`` below is the preferred alias.
    """

    def __init__(
        self,
        *,
        bridge_executable: str | Path | None = None,
        bridge_command: Sequence[str] | None = None,
        sdk_root: str | Path | None = None,
        auto_build: bool = True,
        mode: str = "integrated",
        core_host: str | None = None,
        loopback_only: bool = False,
        hand_motion: str | None = None,
        tracker_diagnostics: bool | None = None,
        startup_timeout: float = 35.0,
        connect_timeout: int = 15,
        glove_timeout: int = 15,
        reconnect_timeout: int = 15,
        discovery_wait: int = 1,
        settings_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        left_calibration: str | Path | None = None,
        right_calibration: str | Path | None = None,
        core_calibration: ManusCoreCalibration | str | Path | None = None,
        max_pending_samples: int = 256,
        required_handedness: Handedness | str = Handedness.RIGHT,
        require_tracker_for_wrist: bool = False,
        tracker_stale_timeout: float = 0.25,
        expected_tracker_id: str | None = None,
        expected_tracker_user_id: int | None = None,
    ) -> None:
        if bridge_command is not None and bridge_executable is not None:
            raise ValueError("Specify bridge_command or bridge_executable, not both")
        if startup_timeout <= 0.0:
            raise ValueError("startup_timeout must be positive")
        if connect_timeout < 0 or glove_timeout < 0 or reconnect_timeout < 0:
            raise ValueError("MANUS bridge timeouts must be non-negative")
        if (
            isinstance(discovery_wait, bool)
            or not isinstance(discovery_wait, int)
            or discovery_wait <= 0
        ):
            raise ValueError("MANUS discovery_wait must be a positive integer")
        if (
            isinstance(max_pending_samples, bool)
            or not isinstance(max_pending_samples, int)
            or max_pending_samples <= 0
        ):
            raise ValueError("max_pending_samples must be a positive integer")
        if not np.isfinite(tracker_stale_timeout) or tracker_stale_timeout <= 0.0:
            raise ValueError("tracker_stale_timeout must be positive and finite")
        try:
            self._required_handedness = Handedness(required_handedness)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid required MANUS handedness: {required_handedness!r}"
            ) from error
        self.mode = _validate_mode(mode)
        self.hand_motion = _validate_hand_motion(
            hand_motion
            if hand_motion is not None
            else ("tracker" if self.mode == "remote" else "auto")
        )
        if core_host is not None and (
            not isinstance(core_host, str) or not core_host.strip()
        ):
            raise ValueError("core_host must be a non-empty IP/name/index or None")
        if self.mode == "integrated" and core_host is not None:
            raise ValueError("core_host is only valid for MANUS Remote mode")
        self.core_host = None if core_host is None else core_host.strip()
        # Integrated embeds Core in this process and must retain the official
        # local-only discovery behavior. Remote may discover the Windows host.
        self.loopback_only = self.mode == "integrated" or bool(loopback_only)
        self.tracker_diagnostics_enabled = (
            self.mode == "remote"
            if tracker_diagnostics is None
            else bool(tracker_diagnostics)
        )
        self.require_tracker_for_wrist = bool(require_tracker_for_wrist)
        self._tracker_stale_timeout = float(tracker_stale_timeout)
        if expected_tracker_id is not None and (
            not isinstance(expected_tracker_id, str) or not expected_tracker_id.strip()
        ):
            raise ValueError("expected_tracker_id must be a non-empty string or None")
        if expected_tracker_user_id is not None and (
            isinstance(expected_tracker_user_id, bool)
            or not isinstance(expected_tracker_user_id, int)
            or expected_tracker_user_id < 0
        ):
            raise ValueError(
                "expected_tracker_user_id must be a non-negative integer or None"
            )
        self._bridge_executable = (
            None if bridge_executable is None else Path(bridge_executable).expanduser()
        )
        self._bridge_command = (
            None
            if bridge_command is None
            else tuple(str(part) for part in bridge_command)
        )
        self._sdk_root = sdk_root
        self._auto_build = bool(auto_build)
        self._startup_timeout = float(startup_timeout)
        self._connect_timeout = int(connect_timeout)
        self._glove_timeout = int(glove_timeout)
        self._reconnect_timeout = int(reconnect_timeout)
        self._discovery_wait = int(discovery_wait)
        self._max_pending_samples = int(max_pending_samples)
        self._settings_dir = (
            None if settings_dir is None else Path(settings_dir).expanduser()
        )
        self._log_dir = None if log_dir is None else Path(log_dir).expanduser()
        self._calibrations = {
            Handedness.LEFT: None
            if left_calibration is None
            else Path(left_calibration).expanduser(),
            Handedness.RIGHT: None
            if right_calibration is None
            else Path(right_calibration).expanduser(),
        }
        if isinstance(core_calibration, ManusCoreCalibration):
            self.core_calibration = core_calibration
            self._core_calibration_path = None
            self._core_calibration_sha256 = None
        elif core_calibration is None:
            self.core_calibration = None
            self._core_calibration_path = None
            self._core_calibration_sha256 = None
        else:
            calibration_path = Path(core_calibration).expanduser()
            try:
                calibration_bytes = calibration_path.read_bytes()
                self.core_calibration = ManusCoreCalibration.load(calibration_path)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Invalid MANUS Core calibration {calibration_path}: {error}"
                ) from error
            self._core_calibration_path = calibration_path
            self._core_calibration_sha256 = hashlib.sha256(
                calibration_bytes
            ).hexdigest()

        self._expected_tracker_id = (
            None if expected_tracker_id is None else expected_tracker_id.strip()
        )
        self._expected_tracker_type = f"{self._required_handedness.value}_hand"
        self._expected_tracker_system = MANUS_ULTIMATE_TRACKER_SYSTEM
        self._expected_tracker_user_id = expected_tracker_user_id
        if self.require_tracker_for_wrist:
            if self.mode != "remote":
                raise ValueError(
                    "Tracker-gated MANUS wrist control requires Remote mode"
                )
            if self.hand_motion != "tracker":
                raise ValueError(
                    "Tracker-gated MANUS wrist control requires hand_motion='tracker'"
                )
            if self.core_calibration is None:
                raise ValueError(
                    "Tracker-gated MANUS wrist control requires a Core calibration"
                )
            try:
                wrist_calibration = self.core_calibration.wrist(
                    self._required_handedness
                )
            except KeyError as error:
                raise ValueError(str(error)) from error
            if self._expected_tracker_id is None:
                raise ValueError(
                    "Tracker-gated MANUS wrist control requires an explicit "
                    "expected_tracker_id runtime policy"
                )
            if self._expected_tracker_user_id is None:
                raise ValueError(
                    "Tracker-gated MANUS wrist control requires an explicit "
                    "expected_tracker_user_id runtime policy"
                )
            if (
                wrist_calibration.tracker_id is not None
                and wrist_calibration.tracker_id != self._expected_tracker_id
            ):
                raise ValueError(
                    "Core calibration tracker_id audit metadata does not match "
                    "the mandatory runtime expected_tracker_id"
                )
            self.tracker_diagnostics_enabled = True

        self._condition = threading.Condition()
        self._latest_articulations: dict[Handedness, HandArticulationSample] = {}
        self._latest_wrists: dict[Handedness, WristPoseSample] = {}
        # Historical private alias retained for local diagnostics/tests.
        self._latest = self._latest_articulations
        self._pending_articulations = {side: deque() for side in Handedness}
        self._pending_wrists = {side: deque() for side in Handedness}
        self._pending_pairs = {side: deque() for side in Handedness}
        self._articulation_draining_sides: set[Handedness] = set()
        self._wrist_draining_sides: set[Handedness] = set()
        self._combined_draining_sides: set[Handedness] = set()
        self._diagnostics: dict[Handedness, ManusFrameDiagnostics] = {}
        self._tracker_diagnostics: ManusTrackerDiagnostics | None = None
        self._landscape: Mapping[str, Any] | None = None
        self._available_hosts: tuple[Mapping[str, Any], ...] = ()
        self._bridge_metadata: dict[str, Any] = {
            "mode": self.mode,
            "hand_motion": self.hand_motion,
        }
        self._reconnect_count = 0
        self._protocol_version: int | None = None
        self._probed_protocol_version: int | None = None
        self._negotiated_capabilities: tuple[str, ...] = ()
        self._protocol_version_history: list[int] = []
        self._connection_generation: int | None = None
        self._connection_state = "starting"
        self._tracker_health_reason = "tracker packet not received"
        self._tracker_last_update_time: int | None = None
        self._tracker_last_receipt_monotonic_ns: int | None = None
        self._validated_tracker: Mapping[str, Any] | None = None
        self._topologies: dict[int, _GloveTopology] = {}
        self._last_sequence: dict[int, int] = {}
        self._error: BaseException | None = None
        self._ready = False
        self._closing = False
        self._process: subprocess.Popen[str] | None = None
        self._protocol_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)

    def _probe_bridge_protocol(self, command: Sequence[str]) -> int:
        try:
            result = subprocess.run(
                [*command, "--protocol-version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SourceUnavailableError(
                f"Could not probe MANUS bridge protocol capabilities: {error}"
            ) from error
        expected_prefix = f"{MANUS_PROTOCOL} "
        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith(expected_prefix)
        ]
        if result.returncode != 0 or len(lines) != 1:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise SourceUnavailableError(
                "MANUS bridge must support the side-effect-free --protocol-version "
                f"capability probe; exit {result.returncode}: {detail}"
            )
        try:
            version = int(lines[0][len(expected_prefix) :])
            _protocol_capabilities(version)
        except ValueError as error:
            raise SourceUnavailableError(
                f"Unsupported MANUS bridge protocol probe response: {lines[0]!r}"
            ) from error
        if version == MANUS_LEGACY_PROTOCOL_VERSION and self.mode != "integrated":
            raise SourceUnavailableError(
                "Protocol-v1 MANUS bridges are supported only for Integrated fallback"
            )
        return version

    def _resolve_command(self) -> list[str]:
        if self._bridge_command is not None:
            command = list(self._bridge_command)
        else:
            executable = self._bridge_executable
            if executable is None:
                mode_environment = os.environ.get(
                    f"DEX_TELEOP_MANUS_BRIDGE_{self.mode.upper()}"
                )
                environment_bridge = mode_environment or os.environ.get(
                    "DEX_TELEOP_MANUS_BRIDGE"
                )
                executable = (
                    Path(environment_bridge).expanduser()
                    if environment_bridge
                    else default_manus_bridge_path(self.mode)
                )
            if not executable.is_file():
                if not self._auto_build:
                    raise SourceUnavailableError(
                        f"MANUS {self.mode} bridge is not built at {executable}; run "
                        f"bash {_native_bridge_directory() / 'build_manus_bridge.sh'} "
                        f"--mode {self.mode}"
                    )
                executable = build_manus_bridge(
                    sdk_root=self._sdk_root,
                    output=executable,
                    mode=self.mode,
                )
            if not os.access(executable, os.X_OK):
                raise SourceUnavailableError(
                    f"MANUS bridge is not executable: {executable}"
                )
            command = [str(executable)]

        protocol_version = self._probe_bridge_protocol(command)
        self._probed_protocol_version = protocol_version
        self._negotiated_capabilities = _protocol_capabilities(protocol_version)

        command.extend(
            [
                "--mode",
                self.mode,
                "--connect-timeout",
                str(self._connect_timeout),
                "--glove-timeout",
                str(self._glove_timeout),
                "--reconnect-timeout",
                str(self._reconnect_timeout),
                "--discovery-wait",
                str(self._discovery_wait),
                "--hand-motion",
                self.hand_motion,
                "--tracker-diagnostics"
                if self.tracker_diagnostics_enabled
                else "--no-tracker-diagnostics",
            ]
        )
        if protocol_version >= MANUS_PROTOCOL_VERSION:
            command.extend(("--required-hand", self._required_handedness.value))
        if self.core_host is not None:
            command.extend(("--core-host", self.core_host))
        if self.loopback_only:
            command.append("--loopback-only")
        if self._settings_dir is not None:
            command.extend(("--settings-dir", str(self._settings_dir)))
        if self._log_dir is not None:
            command.extend(("--log-dir", str(self._log_dir)))
        for side, option in (
            (Handedness.LEFT, "--left-calibration"),
            (Handedness.RIGHT, "--right-calibration"),
        ):
            path = self._calibrations[side]
            if path is not None:
                if not path.is_file():
                    raise SourceUnavailableError(
                        f"MANUS {side.value} calibration file does not exist: {path}"
                    )
                command.extend((option, str(path)))
        return command

    def start(self) -> None:
        """Start and wait for a validated articulation from the required hand."""

        with self._condition:
            if self._process is not None and self._process.poll() is None:
                return
            self._latest_articulations.clear()
            self._latest_wrists.clear()
            for queues in (
                self._pending_articulations,
                self._pending_wrists,
                self._pending_pairs,
            ):
                for queue in queues.values():
                    queue.clear()
            self._articulation_draining_sides.clear()
            self._wrist_draining_sides.clear()
            self._combined_draining_sides.clear()
            self._diagnostics.clear()
            self._tracker_diagnostics = None
            self._landscape = None
            self._available_hosts = ()
            self._bridge_metadata = {
                "mode": self.mode,
                "hand_motion": self.hand_motion,
            }
            self._reconnect_count = 0
            self._protocol_version = None
            self._probed_protocol_version = None
            self._negotiated_capabilities = ()
            self._protocol_version_history.clear()
            self._connection_generation = None
            self._connection_state = "starting"
            self._tracker_health_reason = "tracker packet not received"
            self._tracker_last_update_time = None
            self._tracker_last_receipt_monotonic_ns = None
            self._validated_tracker = None
            self._topologies.clear()
            self._last_sequence.clear()
            self._error = None
            self._ready = False
            self._closing = False

        command = self._resolve_command()
        executable = Path(command[0]).expanduser()
        bridge_sha256 = None
        if executable.is_file():
            try:
                bridge_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
            except OSError:
                bridge_sha256 = None
        with self._condition:
            self._bridge_metadata.update(
                {
                    "resolved_bridge_executable": str(executable.resolve()),
                    "resolved_bridge_sha256": bridge_sha256,
                    "resolved_bridge_command": list(command),
                }
            )
        read_fd, write_fd = os.pipe()
        try:
            process = subprocess.Popen(
                [*command, "--protocol-fd", str(write_fd)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                close_fds=True,
                pass_fds=(write_fd,),
            )
        except OSError as error:
            os.close(read_fd)
            os.close(write_fd)
            raise SourceUnavailableError(
                f"Could not start MANUS bridge: {error}"
            ) from error
        os.close(write_fd)
        with self._condition:
            self._process = process
        self._protocol_thread = threading.Thread(
            target=self._protocol_loop,
            args=(read_fd,),
            name="manus-protocol",
            daemon=True,
        )
        self._protocol_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name="manus-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        deadline = time.monotonic() + self._startup_timeout
        with self._condition:
            while not self._ready and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    tracker_detail = (
                        ""
                        if not self.require_tracker_for_wrist
                        else f"; tracker health: {self._tracker_health_reason}"
                    )
                    self._error = TimeoutError(
                        f"MANUS bridge did not publish a validated "
                        f"{self._required_handedness.value}-hand "
                        f"{'tracker-qualified pair' if self.require_tracker_for_wrist else 'articulation'} "
                        f"within {self._startup_timeout:.1f} seconds{tracker_detail}"
                    )
                    break
                self._condition.wait(timeout=remaining)
            error = self._error
        if error is not None:
            self.close()
            raise SourceUnavailableError(
                f"MANUS {self.mode} source could not start: {error}"
            ) from error

    def _stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        suppress_license_block = False
        for line in process.stderr:
            stripped = line.rstrip()
            lowered = stripped.casefold()
            if "license data" in lowered:
                suppress_license_block = True
                continue
            if suppress_license_block:
                if stripped.strip() == "}":
                    suppress_license_block = False
                continue
            if any(secret in lowered for secret in ('"key"', '"custid"', '"feat"')):
                continue
            with self._condition:
                self._stderr_tail.append(stripped)

    def _protocol_loop(self, read_fd: int) -> None:
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8", errors="strict") as protocol:
                for line in protocol:
                    if not line.strip():
                        continue
                    self._handle_line(line, receipt_monotonic_ns=time.monotonic_ns())
        except BaseException as error:
            self._set_error(error)
            return
        with self._condition:
            if not self._closing and self._error is None:
                process = self._process
                exit_code = None if process is None else process.poll()
                tail = "; ".join(self._stderr_tail)
                detail = f" (exit {exit_code})" if exit_code is not None else ""
                if tail:
                    detail += f": {tail}"
                self._error = RuntimeError(
                    "MANUS bridge protocol stream closed" + detail
                )
                self._condition.notify_all()

    def _set_error(self, error: BaseException) -> None:
        with self._condition:
            if self._error is None and not self._closing:
                self._error = error
            self._condition.notify_all()

    def _update_tracker_health_locked(
        self,
        trackers: Sequence[Mapping[str, Any]],
        *,
        receipt_monotonic_ns: int,
    ) -> None:
        if not self.require_tracker_for_wrist:
            return
        matches = [
            tracker
            for tracker in trackers
            if tracker.get("id") == self._expected_tracker_id
        ]
        reason: str | None = None
        tracker: Mapping[str, Any] | None = None
        if len(matches) != 1:
            reason = (
                f"expected tracker {self._expected_tracker_id!r} is "
                f"{'absent' if not matches else 'duplicated'}"
            )
        else:
            tracker = matches[0]
            checks = (
                (tracker.get("is_hmd") is False, "selected device is an HMD"),
                (
                    tracker.get("type") == self._expected_tracker_type,
                    f"tracker type {tracker.get('type')!r} != {self._expected_tracker_type!r}",
                ),
                (
                    tracker.get("tracking_system") == self._expected_tracker_system,
                    f"tracking system {tracker.get('tracking_system')!r} != "
                    f"{self._expected_tracker_system!r}",
                ),
                (tracker.get("pose_valid") is True, "tracker pose is invalid"),
                (
                    tracker.get("quality") == "trackable",
                    f"tracker quality is {tracker.get('quality')!r}",
                ),
                (
                    self._expected_tracker_user_id is None
                    or tracker.get("user_id") == self._expected_tracker_user_id,
                    f"tracker user {tracker.get('user_id')!r} != "
                    f"{self._expected_tracker_user_id!r}",
                ),
            )
            reason = next((message for ok, message in checks if not ok), None)
        if reason is None and tracker is not None:
            update_time = _require_int(
                tracker.get("last_update_time"),
                "validated tracker.last_update_time",
                positive=True,
            )
            if (
                self._tracker_last_update_time is not None
                and update_time < self._tracker_last_update_time
            ):
                reason = (
                    f"tracker update identity regressed from "
                    f"{self._tracker_last_update_time} to {update_time}"
                )
            elif update_time != self._tracker_last_update_time:
                self._tracker_last_update_time = update_time
                self._tracker_last_receipt_monotonic_ns = receipt_monotonic_ns
        if reason is None:
            self._validated_tracker = dict(tracker) if tracker is not None else None
            self._tracker_health_reason = ""
            return

        self._validated_tracker = None
        self._tracker_health_reason = reason
        self._latest_wrists.clear()
        for queue in self._pending_wrists.values():
            queue.clear()
        for queue in self._pending_pairs.values():
            queue.clear()
        self._ready = False

    def _tracker_health_locked(
        self, now_monotonic_ns: int
    ) -> tuple[bool, str, Mapping[str, Any] | None]:
        if not self.require_tracker_for_wrist:
            return True, "", None
        if (
            self._validated_tracker is None
            or self._tracker_last_receipt_monotonic_ns is None
        ):
            return False, self._tracker_health_reason, None
        age_seconds = (now_monotonic_ns - self._tracker_last_receipt_monotonic_ns) / 1e9
        if age_seconds > self._tracker_stale_timeout:
            self._tracker_health_reason = (
                f"tracker {self._expected_tracker_id!r} update is stale "
                f"({age_seconds:.3f}s > {self._tracker_stale_timeout:.3f}s)"
            )
            self._validated_tracker = None
            self._latest_wrists.clear()
            self._ready = False
            return False, self._tracker_health_reason, None
        return True, "", dict(self._validated_tracker)

    def _handle_line(self, line: str, *, receipt_monotonic_ns: int) -> None:
        def reject_constant(value: str) -> None:
            raise ManusProtocolError(f"Invalid non-finite JSON number: {value}")

        try:
            raw_payload = json.loads(line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ManusProtocolError) as error:
            raise ManusProtocolError(
                f"Invalid MANUS bridge JSON line: {error}"
            ) from error
        payload = _require_mapping(raw_payload, "protocol event")
        if payload.get("protocol") != MANUS_PROTOCOL:
            raise ManusProtocolError(
                f"Unexpected MANUS protocol: {payload.get('protocol')!r}"
            )
        protocol_version = payload.get("version")
        if protocol_version not in (
            MANUS_LEGACY_PROTOCOL_VERSION,
            MANUS_PROTOCOL_VERSION,
        ):
            raise ManusProtocolError(
                f"Unsupported MANUS protocol version {protocol_version!r}; "
                f"expected {MANUS_LEGACY_PROTOCOL_VERSION} or {MANUS_PROTOCOL_VERSION}"
            )
        if (
            protocol_version == MANUS_LEGACY_PROTOCOL_VERSION
            and self.mode != "integrated"
        ):
            raise ManusProtocolError(
                "Legacy MANUS protocol v1 is accepted only for Integrated fallback; "
                "Remote requires connection-generation ordering from protocol v2"
            )
        event_generation = None
        if protocol_version == MANUS_PROTOCOL_VERSION:
            event_generation = _require_int(
                payload.get("connection_generation"),
                "event.connection_generation",
                positive=True,
            )
        with self._condition:
            if self._protocol_version is None:
                if (
                    self._probed_protocol_version is not None
                    and protocol_version != self._probed_protocol_version
                ):
                    raise ManusProtocolError(
                        "MANUS bridge wire protocol does not match its "
                        "--protocol-version capability probe"
                    )
                self._protocol_version = protocol_version
                self._protocol_version_history.append(protocol_version)
                self._negotiated_capabilities = _protocol_capabilities(protocol_version)
                self._bridge_metadata.update(
                    {
                        "bridge_protocol_version": protocol_version,
                        "supported_protocol_version": MANUS_PROTOCOL_VERSION,
                        "bridge_capabilities": list(self._negotiated_capabilities),
                        "protocol_version_history": list(
                            self._protocol_version_history
                        ),
                    }
                )
            elif self._protocol_version != protocol_version:
                raise ManusProtocolError(
                    "MANUS protocol version changed during one process"
                )
            if event_generation is not None:
                current_generation = self._connection_generation
                if (
                    current_generation is not None
                    and event_generation < current_generation
                ):
                    return
                if current_generation is None:
                    self._connection_generation = event_generation
                elif event_generation > current_generation:
                    if not (
                        payload.get("type") == "status"
                        and payload.get("state") == "disconnected"
                        and event_generation == current_generation + 1
                    ):
                        raise ManusProtocolError(
                            "MANUS event advanced connection_generation outside "
                            "a disconnected transition"
                        )
                    self._connection_generation = event_generation
        payload_mode = payload.get("mode")
        if payload_mode is not None and payload_mode != self.mode:
            raise ManusProtocolError(
                f"MANUS bridge mode {payload_mode!r} does not match requested mode {self.mode!r}"
            )
        event_type = payload.get("type")
        if event_type == "status":
            state = payload.get("state")
            with self._condition:
                for key in (
                    "mode",
                    "linked_mode",
                    "core_host_name",
                    "core_host_ip",
                    "sdk_version",
                    "core_version",
                    "versions_compatible",
                    "session_id",
                    "hand_motion",
                    "core_world_frame",
                ):
                    if key in payload:
                        self._bridge_metadata[key] = payload[key]
                if event_generation is not None:
                    self._bridge_metadata["connection_generation"] = event_generation
                if state in {"starting"}:
                    self._connection_state = "starting"
                elif state in {"sdk_connected"}:
                    self._connection_state = "qualifying"
                elif state in {"ready", "topology_ready", "reconnected"}:
                    self._connection_state = "streaming"
                elif state == "reconnecting":
                    self._connection_state = "recovering"
                if state == "reconnected":
                    self._reconnect_count += 1
                    self._bridge_metadata["reconnect_count"] = self._reconnect_count
                    self._bridge_metadata["supported_protocol_version"] = (
                        MANUS_PROTOCOL_VERSION
                    )
                    self._bridge_metadata["protocol_version_history"] = list(
                        self._protocol_version_history
                    )
            if state in {"ready", "topology_ready", "reconnected"}:
                with self._condition:
                    if not self._topologies:
                        raise ManusProtocolError(
                            "MANUS bridge reported topology readiness before publishing a topology"
                        )
                    self._condition.notify_all()
            elif state == "disconnected":
                with self._condition:
                    self._latest_articulations.clear()
                    self._latest_wrists.clear()
                    for queues in (
                        self._pending_articulations,
                        self._pending_wrists,
                        self._pending_pairs,
                    ):
                        for queue in queues.values():
                            queue.clear()
                    self._diagnostics.clear()
                    self._tracker_diagnostics = None
                    self._validated_tracker = None
                    self._tracker_last_update_time = None
                    self._tracker_last_receipt_monotonic_ns = None
                    self._tracker_health_reason = "connection recovering"
                    self._topologies.clear()
                    self._last_sequence.clear()
                    self._ready = False
                    self._connection_state = "recovering"
                    recoverable = payload.get("recoverable", False)
                    if not isinstance(recoverable, bool):
                        raise ManusProtocolError(
                            "MANUS disconnected status recoverable must be boolean"
                        )
                    if not recoverable and self._error is None:
                        self._error = RuntimeError(
                            f"MANUS {self.mode.title()} host disconnected"
                        )
                    self._condition.notify_all()
            return
        if event_type == "hosts":
            raw_hosts = _require_list(payload.get("hosts"), "hosts.hosts")
            hosts = []
            for index, raw_host in enumerate(raw_hosts):
                host = _require_mapping(raw_host, f"hosts.hosts[{index}]")
                if not isinstance(host.get("name"), str) or not isinstance(
                    host.get("ip"), str
                ):
                    raise ManusProtocolError(
                        f"hosts.hosts[{index}] name and ip must be strings"
                    )
                hosts.append(dict(host))
            with self._condition:
                self._available_hosts = tuple(hosts)
                self._condition.notify_all()
            return
        if event_type == "landscape":
            trackers = _require_list(payload.get("trackers"), "landscape.trackers")
            for index, raw_tracker in enumerate(trackers):
                _require_mapping(raw_tracker, f"landscape.trackers[{index}]")
            with self._condition:
                self._landscape = dict(payload)
                self._condition.notify_all()
            return
        if event_type == "trackers":
            if event_generation is not None and self._connection_state not in {
                "starting",
                "qualifying",
                "recovering",
                "streaming",
            }:
                return
            sequence = _require_int(payload.get("sequence"), "trackers.sequence")
            publish_time = _require_int(
                payload.get("manus_publish_time"), "trackers.manus_publish_time"
            )
            capture = _require_int(
                payload.get("capture_monotonic_ns"),
                "trackers.capture_monotonic_ns",
                positive=True,
            )
            raw_trackers = _require_list(payload.get("trackers"), "trackers.trackers")
            trackers: list[Mapping[str, Any]] = []
            for index, raw_tracker in enumerate(raw_trackers):
                tracker = _require_mapping(raw_tracker, f"trackers.trackers[{index}]")
                if not isinstance(tracker.get("id"), str) or not tracker.get("id"):
                    raise ManusProtocolError(
                        f"trackers.trackers[{index}].id must be a non-empty string"
                    )
                for boolean_field in ("is_hmd", "pose_valid"):
                    if not isinstance(tracker.get(boolean_field), bool):
                        raise ManusProtocolError(
                            f"trackers.trackers[{index}].{boolean_field} must be boolean"
                        )
                for string_field in ("type", "quality", "tracking_system"):
                    if not isinstance(tracker.get(string_field), str):
                        raise ManusProtocolError(
                            f"trackers.trackers[{index}].{string_field} must be a string"
                        )
                _require_int(
                    tracker.get("user_id"),
                    f"trackers.trackers[{index}].user_id",
                )
                _require_int(
                    tracker.get("last_update_time"),
                    f"trackers.trackers[{index}].last_update_time",
                )
                if tracker.get("pose_valid"):
                    _vector(
                        tracker.get("position"),
                        3,
                        f"trackers.trackers[{index}].position",
                    )
                    _normalize_quaternion(
                        _vector(
                            tracker.get("orientation_xyzw"),
                            4,
                            f"trackers.trackers[{index}].orientation_xyzw",
                        ),
                        f"trackers.trackers[{index}].orientation_xyzw",
                    )
                trackers.append(dict(tracker))
            session_id = _require_int(
                payload.get("session_id", 0),
                "trackers.session_id",
            )
            if (
                event_generation is not None
                and self.mode == "remote"
                and session_id <= 0
            ):
                raise ManusProtocolError(
                    "Remote tracker packet requires a non-zero MANUS session_id"
                )
            diagnostics = ManusTrackerDiagnostics(
                sequence=sequence,
                manus_publish_time=publish_time,
                capture_monotonic_ns=capture,
                receipt_monotonic_ns=receipt_monotonic_ns,
                connection_generation=event_generation or 0,
                session_id=session_id,
                trackers=tuple(trackers),
            )
            with self._condition:
                current_session = self._bridge_metadata.get("session_id")
                if event_generation is not None and current_session not in (
                    None,
                    session_id,
                ):
                    return
                self._tracker_diagnostics = diagnostics
                self._update_tracker_health_locked(
                    trackers,
                    receipt_monotonic_ns=receipt_monotonic_ns,
                )
                self._condition.notify_all()
            return
        if event_type == "error":
            code = payload.get("code", "unknown")
            message = payload.get("message", "MANUS bridge error")
            sdk_code = payload.get("sdk_return_code")
            suffix = "" if sdk_code is None else f" (SDK return code {sdk_code})"
            self._set_error(ManusProtocolError(f"{code}: {message}{suffix}"))
            return
        if event_type == "topology":
            topology = _parse_topology(payload)
            with self._condition:
                previous = self._topologies.get(topology.glove_id)
                if previous is not None and previous.handedness != topology.handedness:
                    raise ManusProtocolError(
                        f"MANUS glove {topology.glove_id} changed handedness without removal"
                    )
                if any(
                    known.glove_id != topology.glove_id
                    and known.handedness == topology.handedness
                    for known in self._topologies.values()
                ):
                    raise ManusProtocolError(
                        f"MANUS published multiple {topology.handedness.value}-hand glove IDs; "
                        "the side-keyed source requires exactly one glove per hand"
                    )
                self._topologies[topology.glove_id] = topology
                self._condition.notify_all()
            return
        if event_type == "glove_removed":
            glove_id = _require_int(
                payload.get("glove_id"), "glove_removed.glove_id", positive=True
            )
            try:
                handedness = Handedness(payload.get("side"))
            except (TypeError, ValueError) as error:
                raise ManusProtocolError(
                    f"Invalid MANUS glove removal side: {payload.get('side')!r}"
                ) from error
            with self._condition:
                topology = self._topologies.get(glove_id)
                if topology is None:
                    raise ManusProtocolError(
                        f"MANUS removal arrived for unknown glove {glove_id}"
                    )
                if topology.handedness != handedness:
                    raise ManusProtocolError(
                        f"MANUS removal side mismatch for glove {glove_id}: "
                        f"{handedness.value} != {topology.handedness.value}"
                    )
                self._topologies.pop(glove_id, None)
                self._last_sequence.pop(glove_id, None)
                diagnostics = self._diagnostics.get(handedness)
                if handedness == self._required_handedness or (
                    diagnostics is not None and diagnostics.glove_id == glove_id
                ):
                    self._latest_articulations.pop(handedness, None)
                    self._latest_wrists.pop(handedness, None)
                    self._diagnostics.pop(handedness, None)
                    self._pending_articulations[handedness].clear()
                    self._pending_wrists[handedness].clear()
                    self._pending_pairs[handedness].clear()
                if handedness == self._required_handedness:
                    self._ready = False
                    if self._error is None:
                        self._error = RuntimeError(
                            f"Required MANUS {handedness.value}-hand glove {glove_id} disconnected"
                        )
                self._condition.notify_all()
            return
        if event_type == "articulation":
            glove_id = _require_int(
                payload.get("glove_id"), "articulation.glove_id", positive=True
            )
            with self._condition:
                if (
                    event_generation is not None
                    and self._connection_state != "streaming"
                ):
                    return
                topology = self._topologies.get(glove_id)
                bridge_metadata = dict(self._bridge_metadata)
                if event_generation is not None and self.mode == "remote":
                    sample_session = _require_int(
                        payload.get("session_id"),
                        "articulation.session_id",
                        positive=True,
                    )
                    if sample_session != bridge_metadata.get("session_id"):
                        return
            if topology is None:
                raise ManusProtocolError(
                    f"MANUS articulation arrived before topology for glove {glove_id}"
                )
            articulation, wrist, diagnostics = _parse_articulation(
                payload,
                topology,
                receipt_monotonic_ns=receipt_monotonic_ns,
                mode=self.mode,
                calibration=self.core_calibration,
                calibration_sha256=self._core_calibration_sha256,
                bridge_metadata=bridge_metadata,
            )
            with self._condition:
                if event_generation is not None and (
                    event_generation != self._connection_generation
                    or self._connection_state != "streaming"
                ):
                    return
                previous = self._last_sequence.get(glove_id)
                if previous is not None and diagnostics.sequence <= previous:
                    raise ManusProtocolError(
                        f"MANUS sequence did not advance for glove {glove_id}: "
                        f"{diagnostics.sequence} after {previous}"
                    )
                self._last_sequence[glove_id] = diagnostics.sequence
                side = articulation.handedness
                if self.require_tracker_for_wrist and side != self._required_handedness:
                    self._latest_articulations[side] = articulation
                    self._diagnostics[side] = diagnostics
                    self._condition.notify_all()
                    return
                tracker_healthy, tracker_reason, validated_tracker = (
                    self._tracker_health_locked(receipt_monotonic_ns)
                )
                if self.require_tracker_for_wrist:
                    wrist = replace(
                        wrist,
                        confidence=1.0 if tracker_healthy else 0.0,
                        provenance={
                            **dict(wrist.provenance),
                            "tracker_health": (
                                "healthy" if tracker_healthy else "unhealthy"
                            ),
                            "tracker_health_reason": tracker_reason,
                            "validated_tracker_id": (
                                None
                                if validated_tracker is None
                                else validated_tracker.get("id")
                            ),
                            "validated_tracker_type": (
                                None
                                if validated_tracker is None
                                else validated_tracker.get("type")
                            ),
                            "validated_tracker_system": (
                                None
                                if validated_tracker is None
                                else validated_tracker.get("tracking_system")
                            ),
                            "validated_tracker_quality": (
                                None
                                if validated_tracker is None
                                else validated_tracker.get("quality")
                            ),
                            "validated_tracker_last_update_time": (
                                None
                                if validated_tracker is None
                                else validated_tracker.get("last_update_time")
                            ),
                            "tracker_stale_timeout_seconds": self._tracker_stale_timeout,
                        },
                    )
                    if not tracker_healthy:
                        self._latest_articulations[side] = articulation
                        self._diagnostics[side] = diagnostics
                        self._condition.notify_all()
                        return
                if side in self._combined_draining_sides:
                    queue = self._pending_pairs[side]
                    if len(queue) >= self._max_pending_samples:
                        raise SourceUnavailableError(
                            f"MANUS {side.value}-hand acquisition overflowed its "
                            f"{self._max_pending_samples}-sample queue (paired); "
                            "the consumer is not draining fast enough"
                        )
                    queue.append((articulation, wrist))
                else:
                    for draining_sides, queues, value, modality in (
                        (
                            self._articulation_draining_sides,
                            self._pending_articulations,
                            articulation,
                            "articulation",
                        ),
                        (
                            self._wrist_draining_sides,
                            self._pending_wrists,
                            wrist,
                            "wrist",
                        ),
                    ):
                        if side not in draining_sides:
                            continue
                        queue = queues[side]
                        if len(queue) >= self._max_pending_samples:
                            raise SourceUnavailableError(
                                f"MANUS {side.value}-hand acquisition overflowed its "
                                f"{self._max_pending_samples}-sample queue ({modality}); "
                                "the consumer is not draining fast enough"
                            )
                        queue.append(value)
                self._latest_articulations[side] = articulation
                self._latest_wrists[side] = wrist
                self._diagnostics[side] = diagnostics
                if side == self._required_handedness:
                    self._ready = True
                self._condition.notify_all()
            return
        raise ManusProtocolError(f"Unknown MANUS protocol event type: {event_type!r}")

    def read_articulation(
        self, handedness: Handedness
    ) -> HandArticulationSample | None:
        with self._condition:
            return self._latest_articulations.get(Handedness(handedness))

    def read_wrist(self, handedness: Handedness) -> WristPoseSample | None:
        """Return the latest wrist preserved from the same raw-skeleton callback."""

        with self._condition:
            return self._latest_wrists.get(Handedness(handedness))

    def drain_articulations(
        self, handedness: Handedness
    ) -> tuple[HandArticulationSample, ...]:
        """Consume every acquired articulation sample in capture order."""

        self.check_health()
        side = Handedness(handedness)
        with self._condition:
            if side in self._combined_draining_sides:
                raise RuntimeError(
                    "Cannot mix MANUS combined and articulation-only drains for one hand"
                )
            if side not in self._articulation_draining_sides:
                self._articulation_draining_sides.add(side)
                latest = self._latest_articulations.get(side)
                return () if latest is None else (latest,)
            samples = tuple(self._pending_articulations[side])
            self._pending_articulations[side].clear()
        return samples

    def drain_wrists(self, handedness: Handedness) -> tuple[WristPoseSample, ...]:
        """Consume every acquired same-callback wrist sample in capture order."""

        self.check_health()
        side = Handedness(handedness)
        with self._condition:
            if side in self._combined_draining_sides:
                raise RuntimeError(
                    "Cannot mix MANUS combined and wrist-only drains for one hand"
                )
            if side not in self._wrist_draining_sides:
                self._wrist_draining_sides.add(side)
                latest = self._latest_wrists.get(side)
                return () if latest is None else (latest,)
            samples = tuple(self._pending_wrists[side])
            self._pending_wrists[side].clear()
        return samples

    def drain_hand_tracking(self, handedness: Handedness) -> HandTrackingSampleBatch:
        """Atomically drain articulation/wrist pairs from raw-skeleton callbacks."""

        self.check_health()
        side = Handedness(handedness)
        with self._condition:
            if (
                side in self._articulation_draining_sides
                or side in self._wrist_draining_sides
            ):
                raise RuntimeError(
                    "Cannot activate MANUS combined draining after a component-only drain"
                )
            if side not in self._combined_draining_sides:
                self._combined_draining_sides.add(side)
                articulation = self._latest_articulations.get(side)
                wrist = self._latest_wrists.get(side)
                if articulation is None or wrist is None:
                    return HandTrackingSampleBatch()
                return HandTrackingSampleBatch((articulation,), (wrist,))
            pairs = tuple(self._pending_pairs[side])
            self._pending_pairs[side].clear()
        return HandTrackingSampleBatch(
            articulations=tuple(pair[0] for pair in pairs),
            wrists=tuple(pair[1] for pair in pairs),
        )

    def frame_diagnostics(self, handedness: Handedness) -> ManusFrameDiagnostics | None:
        """Return native timing/device metadata for the latest sample."""

        with self._condition:
            return self._diagnostics.get(Handedness(handedness))

    def tracker_diagnostics(self) -> ManusTrackerDiagnostics | None:
        with self._condition:
            return self._tracker_diagnostics

    def recording_metadata(self) -> dict[str, Any]:
        """Return JSON-safe connection, coordinate, and calibration metadata."""

        with self._condition:
            metadata = dict(self._bridge_metadata)
            hosts = [dict(host) for host in self._available_hosts]
            landscape = None if self._landscape is None else dict(self._landscape)
            reconnect_count = self._reconnect_count
            tracker_healthy, tracker_health_reason, validated_tracker = (
                self._tracker_health_locked(time.monotonic_ns())
            )
            connection_generation = self._connection_generation
            connection_state = self._connection_state
            protocol_version = self._protocol_version or self._probed_protocol_version
            capabilities = list(self._negotiated_capabilities)
            protocol_version_history = list(self._protocol_version_history)
        metadata.update(
            {
                "provider": f"MANUS SDK {self.mode.title()}",
                "mode": self.mode,
                "requested_core_host": self.core_host,
                "loopback_only": self.loopback_only,
                "hand_motion": self.hand_motion,
                "bridge_protocol_version": protocol_version,
                "supported_protocol_version": MANUS_PROTOCOL_VERSION,
                "bridge_capabilities": capabilities,
                "protocol_version_history": protocol_version_history,
                "protocol_version_changes_allowed": False,
                "core_world_frame": MANUS_CORE_WORLD_FRAME,
                "discovered_hosts": hosts,
                "landscape": landscape,
                "reconnect_count": reconnect_count,
                "reconnect_count_at_recording_start": reconnect_count,
                "connection_generation": connection_generation,
                "connection_state": connection_state,
                "require_tracker_for_wrist": self.require_tracker_for_wrist,
                "expected_tracker_id": self._expected_tracker_id,
                "expected_tracker_type": self._expected_tracker_type,
                "expected_tracker_system": self._expected_tracker_system,
                "expected_tracker_user_id": self._expected_tracker_user_id,
                "tracker_stale_timeout_seconds": self._tracker_stale_timeout,
                "tracker_healthy": tracker_healthy,
                "tracker_health_reason": tracker_health_reason,
                "validated_tracker": validated_tracker,
                "core_calibration": (
                    None
                    if self.core_calibration is None
                    else self.core_calibration.as_mapping()
                ),
                "core_calibration_file": (
                    None
                    if self._core_calibration_path is None
                    else self._core_calibration_path.name
                ),
                "core_calibration_sha256": self._core_calibration_sha256,
            }
        )
        return metadata

    def check_health(self) -> None:
        with self._condition:
            error = self._error
            process = self._process
            if error is None and process is not None:
                exit_code = process.poll()
                if exit_code is not None and not self._closing:
                    error = RuntimeError(
                        f"MANUS bridge exited unexpectedly with status {exit_code}"
                    )
                    self._error = error
            if (
                error is None
                and self.require_tracker_for_wrist
                and self._connection_state == "streaming"
            ):
                tracker_healthy, tracker_reason, _ = self._tracker_health_locked(
                    time.monotonic_ns()
                )
                if not tracker_healthy:
                    error = RuntimeError(
                        f"required Ultimate tracker is unhealthy: {tracker_reason}"
                    )
        if error is not None:
            raise SourceUnavailableError(
                f"MANUS {self.mode} source failed: {error}"
            ) from error

    def close(self) -> None:
        with self._condition:
            self._closing = True
            process = self._process
            self._condition.notify_all()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        for thread in (self._protocol_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        with self._condition:
            self._process = None
            self._protocol_thread = None
            self._stderr_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# Preferred transport-neutral name. The historical name above remains a true
# alias so existing imports and isinstance checks keep working.
ManusSource = ManusIntegratedSource


__all__ = [
    "MANUS_ARTICULATION_SCHEMA",
    "MANUS_HAND_MOTIONS",
    "MANUS_JOINT_NAMES",
    "MANUS_MODES",
    "MANUS_PROTOCOL",
    "MANUS_PROTOCOL_VERSION",
    "ManusFrameDiagnostics",
    "ManusIntegratedSource",
    "ManusProtocolError",
    "ManusSdkLayout",
    "ManusSource",
    "ManusTrackerDiagnostics",
    "build_manus_bridge",
    "default_manus_bridge_path",
    "discover_manus_sdk",
]

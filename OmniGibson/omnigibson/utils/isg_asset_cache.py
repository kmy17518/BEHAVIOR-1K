"""iSpatialGym vec-eval overlay (campaign vec_next, 2026-10-31): a persistent, content-keyed cache of per-asset geometry results.

The world build recomputes, in every process, values that depend only on an asset's geometry and pose: a collision mesh's volume and
centre of mass (trimesh), its oblong check, its world-frame volume, and a link's link-local collision points. The in-process caches of
prims/rigid_prim.py let a process's clones reuse scene 0's values; this store lets every process and job reuse them, scene 0 included.

Keys are hashes of EXACTLY the inputs of each computation (the mesh's points / face counts / face indices, or a primitive's shape
attributes; plus the world transform with scale, or the link pose, where the computation uses them), never path names: a cached value
is by construction the value the stock path would compute, and a changed asset or pose simply misses. Values are stored with their
original types (numpy vs Python floats, tensor dtypes), so downstream arithmetic is bit-identical.

``ISG_VEC_ASSET_CACHE_DIR=<dir>`` turns it on (unset = off, the code paths are exactly the previous ones); ``ISG_VEC_ASSET_CACHE_MODE``
= ``rw`` (default: read, and add what this process computed) or ``ro``. Storage: one pickle shard per writing process,
``<dir>/shard_<host>_<pid>_<time>.pkl``, written atomically at exit (no two processes ever write one file); every shard is read on
first use. Hits / misses / puts are printed at exit as ``ISG_ASSET_CACHE ...``.
"""
from __future__ import annotations

import atexit
import glob
import hashlib
import os
import pickle
import socket
import time

import numpy as np

_STATE = {"loaded": False, "mem": {}, "new": {}, "hits": 0, "misses": 0, "shards": 0}


def enabled() -> bool:
    return bool(os.environ.get("ISG_VEC_ASSET_CACHE_DIR"))


def _dir():
    return os.environ.get("ISG_VEC_ASSET_CACHE_DIR")


def _load():
    if _STATE["loaded"]:
        return
    _STATE["loaded"] = True
    d = _dir()
    if not d or not os.path.isdir(d):
        return
    for f in sorted(glob.glob(os.path.join(d, "shard_*.pkl"))):
        try:
            with open(f, "rb") as fh:
                _STATE["mem"].update(pickle.load(fh))
            _STATE["shards"] += 1
        except Exception as e:  # noqa: BLE001  (a truncated shard of a killed process: skipped)
            print(f"ISG_ASSET_CACHE skipped unreadable shard {f}: {e!r}", flush=True)


def get(kind: str, key: str):
    """The stored value or None (a computed None is stored and returned as ``STORED_NONE``)."""
    if not enabled():
        return None
    _load()
    v = _STATE["mem"].get((kind, key), _ABSENT)
    if v is _ABSENT:
        _STATE["misses"] += 1
        return None
    _STATE["hits"] += 1
    return v


def put(kind: str, key: str, value) -> None:
    if not enabled() or os.environ.get("ISG_VEC_ASSET_CACHE_MODE", "rw") == "ro":
        return
    _load()
    if (kind, key) not in _STATE["mem"]:
        _STATE["mem"][(kind, key)] = value
        _STATE["new"][(kind, key)] = value


_ABSENT = object()
STORED_NONE = "__isg_asset_cache_none__"  # how a computed None (a link without collision geometry) is stored


def flush(reason: str = "exit"):
    """Write this process's new entries as one shard (callable repeatedly: each call writes only what was added since the last).
    Called by the environment at the end of its build (Kit's shutdown may skip atexit) and at exit."""
    if not enabled():
        return
    new = _STATE["new"]
    _STATE["new"] = {}
    print(f"ISG_ASSET_CACHE {reason} hits={_STATE['hits']} misses={_STATE['misses']} new={len(new)} shards_read={_STATE['shards']} dir={_dir()}",
          flush=True)
    if not new or os.environ.get("ISG_VEC_ASSET_CACHE_MODE", "rw") == "ro":
        return
    d = _dir()
    try:
        os.makedirs(d, exist_ok=True)
        name = f"shard_{socket.gethostname()}_{os.getpid()}_{int(time.time() * 1000)}.pkl"
        tmp = os.path.join(d, "." + name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(new, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(d, name))
    except Exception as e:  # noqa: BLE001
        print(f"ISG_ASSET_CACHE flush failed: {e!r}", flush=True)


atexit.register(flush)


# ---------------------------------------------------------------------------------------------------------------- fingerprints

def _arr_bytes(x) -> bytes:
    if x is None:
        return b"<none>"
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()


def geometry_fingerprint(prim) -> str:
    """Hash of the inputs mesh_prim_to_trimesh_mesh / GeomPrim.points_and_faces read: for a Mesh its points, faceVertexCounts and
    faceVertexIndices; for a primitive shape its type and every non-xform, non-primvar attribute value."""
    h = hashlib.sha1()
    t = prim.GetTypeName()
    h.update(str(t).encode())
    if t == "Mesh":
        for name in ("points", "faceVertexCounts", "faceVertexIndices"):
            a = prim.GetAttribute(name)
            h.update(name.encode())
            h.update(_arr_bytes(a.Get() if a else None))
    else:
        for a in sorted(prim.GetAttributes(), key=lambda x: x.GetName()):
            n = a.GetName()
            if n.startswith("xformOp") or n.startswith("primvars:") or n == "xformOpOrder":
                continue
            v = a.Get()
            h.update(n.encode())
            try:
                h.update(_arr_bytes(v))
            except Exception:  # noqa: BLE001
                h.update(repr(v).encode())
    return h.hexdigest()


def combine(*parts) -> str:
    """Hash of fingerprints / tensors / numbers, in order."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, str):
            h.update(p.encode())
        else:
            h.update(_arr_bytes(p))
        h.update(b"|")
    return h.hexdigest()

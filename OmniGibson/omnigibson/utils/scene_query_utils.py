"""Scene-scoped PhysX scene queries for co-located clones (iSpatialGym vec-eval overlay, 2026-10-27).

With ``ISG_SCENE_COLOCATED=1`` every cloned scene of a vectorized ``og.Environment`` sits at the origin (``scenes/scene_base.py``
of this overlay): physics is isolated per scene by collision groups, but PhysX *scene queries* (``og.sim.psqi.raycast_*``,
``overlap_*``, ``sweep_*``) see the whole stage, so the nearest surface along a ray is usually a twin from another clone. This
module wraps the scene-query interface so that every query is answered within the **querying scene**: hits whose prim path is
under another ``/World/scene_<j>/`` are dropped (prims outside any scene prefix - the ground plane, global lights - are kept);
closest-hit queries become all-hits queries with the nearest own-scene hit; any-hit queries become callback queries that stop at
the first own-scene hit.

The querying scene is the *ambient scope*: ``with query_scene(scene_or_prefix_or_idx): ...`` around the code that issues the
queries (``VecISpatialGymTask`` per-env predicate evaluation on the evaluator side; ``BaseRobot._find_gripper_raycast_collisions``
in this overlay). Without a scope a query is passed through unfiltered and counted (``unscoped`` in the trace) - the report
lists them.

``ISG_QUERY_TRACE=1`` records every scene-query call with its caller (first stack frame outside this module and
``utils/sampling_utils.py``), the method, whether a scope was active and how many hits were dropped; the table is printed as
``ISG_QUERY_TRACE`` lines at exit and written to ``$ISG_QUERY_TRACE_OUT`` (default ``query_trace.json`` in the cwd).

Installation: :func:`install` replaces ``Simulator.psqi`` (a property returning the raw ``omni.physx`` interface) with one
returning :class:`ScopedSceneQuery`; it is called by the overlay's ``Scene.load`` when either env var is set, and is idempotent.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import traceback
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

_PREFIX_RE = re.compile(r"^(/World/scene_\d+)(?:/|$)")
_HERE = os.path.abspath(__file__)
_SAMPLING_UTILS = os.path.join(os.path.dirname(_HERE), "sampling_utils.py")


def colocated_enabled() -> bool:
    return os.environ.get("ISG_SCENE_COLOCATED", "") == "1"


def trace_enabled() -> bool:
    return os.environ.get("ISG_QUERY_TRACE", "") == "1"


def scene_prefix(path: Any) -> Optional[str]:
    """``/World/scene_<k>`` of a prim path under a scene, else None."""
    m = _PREFIX_RE.match(str(path))
    return m.group(1) if m else None


def prefix_of(scene_or_prefix) -> Optional[str]:
    """A scope argument (an OmniGibson scene, a scene index, a prim path / prefix string) as a ``/World/scene_<k>`` prefix."""
    if scene_or_prefix is None:
        return None
    if isinstance(scene_or_prefix, int):
        return f"/World/scene_{scene_or_prefix}"
    if isinstance(scene_or_prefix, str):
        return scene_prefix(scene_or_prefix) or (scene_or_prefix if scene_or_prefix.startswith("/World/scene_") else None)
    idx = getattr(scene_or_prefix, "idx", None)
    if idx is not None:
        return f"/World/scene_{int(idx)}"
    p = getattr(scene_or_prefix, "prim_path", None)
    return scene_prefix(p) if p is not None else None


class _Scope:
    prefix: Optional[str] = None
    depth: int = 0


_SCOPE = _Scope()


@contextmanager
def query_scene(scene_or_prefix):
    """Ambient scope: scene queries issued inside answer within this scene (None = leave the current scope)."""
    prefix = prefix_of(scene_or_prefix)
    if prefix is None:
        yield
        return
    prev = _SCOPE.prefix
    _SCOPE.prefix, _SCOPE.depth = prefix, _SCOPE.depth + 1
    try:
        yield
    finally:
        _SCOPE.prefix, _SCOPE.depth = prev, _SCOPE.depth - 1


def current_prefix() -> Optional[str]:
    return _SCOPE.prefix


def own_path(path: Any, prefix: str) -> bool:
    """True when ``path`` is in the querying scene or outside every scene prefix."""
    p = scene_prefix(path)
    return p is None or p == prefix


# ------------------------------------------------------------------------------------------------------------ trace


class _Trace:
    def __init__(self):
        self.counts: Counter = Counter()  # (method, caller, scoped) -> calls
        self.dropped: Counter = Counter()  # (method, caller) -> hits dropped
        self.unscoped_callers: Counter = Counter()
        self._registered = False

    def caller(self) -> str:
        for fr in reversed(traceback.extract_stack(limit=12)[:-2]):
            fn = os.path.abspath(fr.filename)
            if fn == _HERE or fn == _SAMPLING_UTILS:
                continue
            return f"{fr.filename}:{fr.lineno}:{fr.name}"
        return "?"

    def record(self, method: str, scoped: bool, dropped: int = 0) -> None:
        c = self.caller()
        self.counts[(method, c, scoped)] += 1
        if dropped:
            self.dropped[(method, c)] += dropped
        if not scoped and colocated_enabled():
            self.unscoped_callers[(method, c)] += 1
        if not self._registered:
            atexit.register(self.dump)
            self._registered = True

    def table(self) -> Dict[str, Any]:
        rows = [{"method": m, "caller": c, "scoped": s, "calls": n, "dropped_hits": self.dropped.get((m, c), 0)}
                for (m, c, s), n in sorted(self.counts.items(), key=lambda kv: -kv[1])]
        return {"colocated": colocated_enabled(), "rows": rows,
                "unscoped_in_colocated_mode": [{"method": m, "caller": c, "calls": n} for (m, c), n in self.unscoped_callers.most_common()]}

    def dump(self) -> None:
        t = self.table()
        for r in t["rows"]:
            print(f"ISG_QUERY_TRACE {json.dumps(r)}", flush=True)
        print(f"ISG_QUERY_TRACE_SUMMARY {json.dumps({'sites': len(t['rows']), 'calls': sum(r['calls'] for r in t['rows']), 'unscoped_sites': len(t['unscoped_in_colocated_mode'])})}", flush=True)
        out = os.environ.get("ISG_QUERY_TRACE_OUT") or os.path.join(os.getcwd(), "query_trace.json")
        try:
            with open(out, "w") as f:
                json.dump(t, f, indent=1)
        except Exception as e:  # noqa: BLE001
            print(f"ISG_QUERY_TRACE write failed: {e!r}", flush=True)


TRACE = _Trace()

# -------------------------------------------------------------------------------------------------- the wrapped interface


def _hit_dict(hit) -> Dict[str, Any]:
    """A ``RaycastHit`` / ``SweepHit`` as the dict ``raycast_closest`` / ``sweep_*_closest`` return."""
    d = {"hit": True, "position": hit.position, "normal": hit.normal, "distance": hit.distance, "collision": hit.collision,
         "rigidBody": hit.rigid_body}
    for src, dst in (("face_index", "faceIndex"), ("material", "material"), ("protoIndex", "protoIndex")):
        try:
            d[dst] = getattr(hit, src)
        except Exception:  # noqa: BLE001
            pass
    return d


class ScopedSceneQuery:
    """``og.sim.psqi`` filtered to the ambient scene (module docstring). Methods not listed here are forwarded unchanged."""

    _CALLBACK = {"raycast_all": 3, "overlap_box": 3, "overlap_sphere": 2, "overlap_mesh": 2, "overlap_shape": 2,
                 "sweep_box_all": 5, "sweep_mesh_all": 4, "sweep_shape_all": 4, "sweep_sphere_all": 4}  # index of reportFn
    _CLOSEST = {"raycast_closest": "raycast_all", "sweep_box_closest": "sweep_box_all", "sweep_mesh_closest": "sweep_mesh_all",
                "sweep_shape_closest": "sweep_shape_all", "sweep_sphere_closest": "sweep_sphere_all"}
    _ANY = {"raycast_any": "raycast_all", "overlap_box_any": "overlap_box", "overlap_sphere_any": "overlap_sphere",
            "overlap_mesh_any": "overlap_mesh", "overlap_shape_any": "overlap_shape", "sweep_box_any": "sweep_box_all",
            "sweep_mesh_any": "sweep_mesh_all", "sweep_shape_any": "sweep_shape_all", "sweep_sphere_any": "sweep_sphere_all"}

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        real = object.__getattribute__(self, "_real")
        attr = getattr(real, name)
        if not callable(attr) or not (name in self._CALLBACK or name in self._CLOSEST or name in self._ANY):
            return attr
        return self._wrap(name, attr)

    def _wrap(self, name: str, fn: Callable) -> Callable:
        real = object.__getattribute__(self, "_real")
        trace = trace_enabled()

        def call(*args, **kwargs):
            prefix = current_prefix() if colocated_enabled() else None
            if prefix is None:
                if trace:
                    TRACE.record(name, scoped=False)
                return fn(*args, **kwargs)
            dropped = [0]

            def keep(hit) -> bool:
                ok = own_path(hit.rigid_body, prefix) and own_path(hit.collision, prefix)
                if not ok:
                    dropped[0] += 1
                return ok

            if name in self._CALLBACK:
                pos = self._CALLBACK[name]
                args = list(args)
                if "reportFn" in kwargs:
                    user = kwargs["reportFn"]
                    kwargs["reportFn"] = lambda hit: True if not keep(hit) else user(hit)
                else:
                    user = args[pos]
                    args[pos] = lambda hit: True if not keep(hit) else user(hit)
                out = fn(*args, **kwargs)
            elif name in self._CLOSEST:
                best = [None]

                def collect(hit) -> bool:
                    if keep(hit) and (best[0] is None or hit.distance < best[0].distance):
                        best[0] = hit
                    return True

                self._call_all(real, self._CLOSEST[name], args, kwargs, collect)
                out = _hit_dict(best[0]) if best[0] is not None else {"hit": False}
            else:  # _ANY
                found = [False]

                def first(hit) -> bool:
                    if keep(hit):
                        found[0] = True
                        return False
                    return True

                self._call_all(real, self._ANY[name], args, kwargs, first)
                out = found[0]
            if trace:
                TRACE.record(name, scoped=True, dropped=dropped[0])
            return out

        call.__name__ = name
        return call

    def _call_all(self, real, all_name: str, args, kwargs, report: Callable) -> Any:
        """``raycast_all`` / ``overlap_*`` / ``sweep_*_all`` with ``report`` inserted at the interface's reportFn position (the
        closest / any signatures are the all signature without reportFn)."""
        pos = self._CALLBACK[all_name]
        args = list(args)
        kwargs = dict(kwargs)
        if len(args) >= pos:
            args.insert(pos, report)
        else:
            kwargs["reportFn"] = report
        return getattr(real, all_name)(*args, **kwargs)


_INSTALLED = {"done": False}


def install() -> bool:
    """Make ``og.sim.psqi`` return :class:`ScopedSceneQuery` (when co-located mode or the trace is on); returns True if installed."""
    if _INSTALLED["done"]:
        return True
    if not (colocated_enabled() or trace_enabled()):
        return False
    import omnigibson as og
    import omnigibson.lazy as lazy

    if og.sim is None:
        return False
    cls = type(og.sim)
    cls.psqi = property(lambda self: ScopedSceneQuery(lazy.omni.physx.get_physx_scene_query_interface()))
    _INSTALLED["done"] = True
    print(f"ISG_SCENE_QUERY installed scoped scene queries (colocated={colocated_enabled()}, trace={trace_enabled()})", flush=True)
    return True


__all__ = ["ScopedSceneQuery", "colocated_enabled", "current_prefix", "install", "own_path", "prefix_of", "query_scene", "scene_prefix", "trace_enabled", "TRACE"]

"""Build-phase timers of the iSpatialGym vec-eval overlay (campaign vec_next, 2026-10-29): ``with build_timer("phase"): ...`` prints
``ISG_VEC_BUILD_TIMER <phase> <seconds>`` when ``ISG_VEC_BUILD_TIMERS=1``; otherwise a no-op context (the wrappers in envs/env_base.py,
simulator.py and scenes/scene_base.py cost one perf_counter call each). Where a world build's time goes, per scene and per phase."""

import os
import time


class _Timer:
    __slots__ = ("phase", "t0")

    def __init__(self, phase):
        self.phase = phase

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if os.environ.get("ISG_VEC_BUILD_TIMERS", "0") == "1":
            print(f"ISG_VEC_BUILD_TIMER {self.phase} {time.perf_counter() - self.t0:.2f}", flush=True)
        return False


def build_timer(phase):
    return _Timer(phase)

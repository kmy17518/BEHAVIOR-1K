import torch as th

import omnigibson as og
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import get_reachability_sampling_context
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.sampling_utils import raytest_batch


FIXED_SUPPORT_VERTICAL_TOLERANCE = 2e-3


class OnTop(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({Touching, VerticalAdjacency})
        return deps

    def _set_value(self, other, new_value, reset_before_sampling=False, use_trav_map=True):
        if not new_value:
            raise NotImplementedError("OnTop does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object on top of a cloth object.")

        state = og.sim.dump_state(serialized=False)

        # Possibly reset this object if requested
        if reset_before_sampling:
            self.obj.reset()

        reachability_context = get_reachability_sampling_context(other, "onTop", use_trav_map=use_trav_map)
        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            if sample_kinematics(
                "onTop", self.obj, other, use_trav_map=use_trav_map, reachability_context=reachability_context
            ) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False

    def _get_value(self, other):
        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect if an object is on top of a cloth object.")

        # PhysX does not report contacts between two kinematic bodies. Fixed
        # task fixtures can nevertheless have a well-defined supporting
        # surface, so use a short downward ray at the child's footprint center
        # when the child is intentionally fixed. This also handles articulated
        # supports whose other links extend above the child (for example, an
        # open toolbox), which makes the global vertical-adjacency test
        # ambiguous even though the local support is unambiguous.
        if (self.obj.fixed_base or self.obj.kinematic_only) and self._has_fixed_support_below(other):
            return True

        touching = self.obj.states[Touching].get_value(other)
        if not touching:
            return False

        adjacency = self.obj.states[VerticalAdjacency].get_value()
        return other in adjacency.negative_neighbors and other not in adjacency.positive_neighbors

    def _has_fixed_support_below(self, other):
        lower, upper = self.obj.states[AABB].get_value()
        center = (lower + upper) / 2.0
        tolerance = FIXED_SUPPORT_VERTICAL_TOLERANCE
        ray_start = center.clone()
        ray_end = center.clone()
        ray_start[2] = lower[2] + tolerance
        ray_end[2] = lower[2] - tolerance
        results = raytest_batch(
            ray_start.reshape(1, 3),
            ray_end.reshape(1, 3),
            only_closest=False,
            ignore_bodies=self.obj.link_prim_paths,
            ignore_collisions=self.obj.link_prim_paths,
        )[0]
        other_link_paths = set(other.link_prim_paths)
        for result in results:
            if result["rigidBody"] not in other_link_paths:
                continue
            hit_z = th.as_tensor(result["position"], dtype=lower.dtype, device=lower.device)[2]
            separation = lower[2] - hit_z
            if -tolerance <= separation <= tolerance:
                return True
        return False

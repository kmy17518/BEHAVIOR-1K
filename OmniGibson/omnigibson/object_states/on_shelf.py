import torch as th

from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.usd_utils import RigidContactAPI


# Vertical band around the shelf plane within which the object's lowest point must lie
SHELF_PLANE_TOLERANCE_BELOW = 1e-2
SHELF_PLANE_TOLERANCE_ABOVE = 2e-2
# Horizontal margin added around the shelf link's footprint
SHELF_FOOTPRINT_MARGIN = 1e-2
# Linear speed under which a non-contacting object still counts as resting. Sleeping
# bodies stop appearing in the contact matrices, so a settled object cannot be required
# to show an active contact.
SHELF_RESTING_SPEED = 1e-2


class OnShelf(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    """
    OnShelf(obj, other): obj rests on the designated shelf surface of @other — the top face
    of @other's root link — rather than merely touching @other somewhere. This distinguishes
    e.g. the top of the ARAT box's base shell (the 37 cm shelf) from the same object's open
    lid, which is a different link and serves as the starting support.
    """

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB})
        return deps

    def _set_value(self, other, new_value):
        raise NotImplementedError("OnShelf does not support set_value")

    def _get_value(self, other):
        shelf_link = other.root_link
        shelf_lower, shelf_upper = shelf_link.aabb
        plane_z = shelf_upper[2]

        lower, upper = self.obj.states[AABB].get_value()
        if not (plane_z - SHELF_PLANE_TOLERANCE_BELOW <= lower[2] <= plane_z + SHELF_PLANE_TOLERANCE_ABOVE):
            return False

        # The object's footprint center must be over the shelf surface
        center = (lower + upper) / 2.0
        if not (
            shelf_lower[0] - SHELF_FOOTPRINT_MARGIN <= center[0] <= shelf_upper[0] + SHELF_FOOTPRINT_MARGIN
            and shelf_lower[1] - SHELF_FOOTPRINT_MARGIN <= center[1] <= shelf_upper[1] + SHELF_FOOTPRINT_MARGIN
        ):
            return False

        # Resting on the shelf link: either an active contact with that specific link, or
        # effectively zero velocity (sleeping bodies report no contacts)
        in_contact = RigidContactAPI.is_in_contact(
            scene_idx=self.obj.scene.idx,
            query_set=[self.obj],
            with_set=[shelf_link],
            ignore_set=None,
            current_only=True,
        )
        if in_contact:
            return True
        return bool(th.norm(self.obj.root_link.get_linear_velocity()) < SHELF_RESTING_SPEED)

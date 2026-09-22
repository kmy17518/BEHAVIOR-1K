import torch as th

from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState


# Maximum horizontal offset between the object's center axis and the peg axis
PEG_AXIS_TOLERANCE = 1.2e-2
# The object's lowest point must sit at least this far below the peg's top
PEG_MIN_INSERTION = 5e-3


class Pegged(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    """
    Pegged(obj, other): obj — an object with a central bore, such as a tube or washer — is
    placed over the upright peg @other so the peg passes through the bore. Evaluated
    geometrically: the object's center axis is horizontally aligned with the peg axis and
    the object's vertical extent overlaps the peg below its top.
    """

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({AABB})
        return deps

    def _set_value(self, other, new_value):
        raise NotImplementedError("Pegged does not support set_value")

    def _get_value(self, other):
        peg_lower, peg_upper = other.states[AABB].get_value()
        peg_axis_xy = (peg_lower[:2] + peg_upper[:2]) / 2.0

        lower, upper = self.obj.states[AABB].get_value()
        center_xy = (lower[:2] + upper[:2]) / 2.0

        if th.norm(center_xy - peg_axis_xy) > PEG_AXIS_TOLERANCE:
            return False

        # The bore must actually be over the peg: the object's lowest point sits below the
        # peg's top, and the object is not entirely underneath the peg
        return bool(lower[2] < peg_upper[2] - PEG_MIN_INSERTION and upper[2] > peg_lower[2])

from omnigibson.controllers import IsGraspingState
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin, RelativeObjectState
from omnigibson.sensors import VisionSensor

_IN_REACH_DISTANCE_THRESHOLD = 2.0

_IN_FOV_PIXEL_FRACTION_THRESHOLD = 0.05


class RobotStateMixin:
    @property
    def robot(self):
        from omnigibson.robots.robot import Robot

        assert isinstance(self.obj, Robot), "This state only works with robots."
        return self.obj


class IsGrasping(RelativeObjectState, BooleanStateMixin, RobotStateMixin):
    def _get_value(self, obj):
        # TODO: Make this work with non-assisted grasping
        return any(self.robot._ag_obj_in_hand[arm] == obj for arm in self.robot.arm_names)


class Grasped(RelativeObjectState, BooleanStateMixin, RobotStateMixin):
    """Binary predicate (grasped agent obj): True if the robot is grasping @obj in either gripper.

    Unlike IsGrasping, this delegates to Robot.is_grasping(), so it follows the robot's configured
    grasping_mode and works for both assisted/sticky and physical (non-assisted) grasps.
    """

    def _get_value(self, obj):
        # Only manipulation robots can grasp; non-manipulation robots never satisfy this predicate.
        if not self.robot.is_manipulation:
            return False
        return any(
            self.robot.is_grasping(arm, candidate_obj=obj) == IsGraspingState.TRUE for arm in self.robot.arm_names
        )


# class InReachOfRobot(AbsoluteObjectState, BooleanStateMixin):
#     def _compute_value(self):
#         robot = _get_robot(self.simulator)
#         if not robot:
#             return False

#         robot_pos = robot.get_position_orientation()[0]
#         object_pos = self.obj.get_position_orientation()[0]
#         return th.norm(object_pos - th.tensor(robot_pos)) < _IN_REACH_DISTANCE_THRESHOLD


# class InFOVOfRobot(AbsoluteObjectState, BooleanStateMixin):
#     @staticmethod
#     def get_optional_dependencies():
#         return AbsoluteObjectState.get_optional_dependencies() + [ObjectsInFOVOfRobot]

#     def _get_value(self):
#         robot = _get_robot(self.simulator)
#         if not robot:
#             return False

#         body_ids = set(self.obj.get_body_ids())
#         return not body_ids.isdisjoint(robot.states[ObjectsInFOVOfRobot].get_value())


class ObjectsInFOVOfRobot(AbsoluteObjectState, RobotStateMixin):
    def _get_value(self):
        """
        Gets all objects in the robot's field of view.

        Returns:
            set: Set of objects in the robot's field of view
        """
        if not any(isinstance(sensor, VisionSensor) for sensor in self.robot.sensors.values()):
            raise ValueError("No vision sensors found on robot.")
        objs = set()
        names_to_exclude = set(["background", "unlabelled"])
        for sensor in self.robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                _, info = sensor.get_obs()
                objs.update(
                    set(
                        self.obj.scene.object_registry("name", name)
                        for name in info["seg_instance"].values()
                        if name not in names_to_exclude
                    )
                )
        # Return all objects, minus any that were mapped to None because they were not found in our object registry
        return objs - {None}

"""
Copyright 2025 Enrico Saccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__maintainers__ = ["Enrico Saccon", "Davide De Martini", "Marco Roveri", "Davide Nardi"]

"""The known actions, and what each one decomposes into.

The behaviour tree names an action and an object; everything about how the arm
gets there lives here. Four actions are known -- ``home``, ``move_to``,
``pick`` and ``place`` -- and the last two are the only interesting ones: each
is an approach from a safe height, a straight descent, the fake grasp, and a
straight retreat. The descent and the retreat are Cartesian on purpose, so the
tool comes down on an object rather than swinging into it sideways.

Adding an action means adding a method and an entry in ``DISPATCH``. The
behaviour tree needs no change to call it.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from geometry_msgs.msg import Pose

from plantorv_planner.geometry import pose_above, raised, square_symmetric_yaw, yaw_of
from plantorv_planner.moveit_client import MoveItClient, Plan, PlanningError


@dataclass
class ActionOutcome:
    """What an action did."""

    message: str
    path_length: float = 0.0
    planning_time: float = 0.0

    def __add__(self, other: "ActionOutcome") -> "ActionOutcome":
        """Accumulate the cost of the steps an action is made of."""
        return ActionOutcome(
            message=self.message or other.message,
            path_length=self.path_length + other.path_length,
            planning_time=self.planning_time + other.planning_time,
        )


@dataclass
class Held:
    """The object currently in the fake gripper."""

    name: str
    height: float


class ActionLibrary:
    """Turns a named action into planned, executed motion.

    Parameters
    ----------
    node : rclpy.node.Node
        Used for parameters, logging, and the service calls to the world model
        and the fake grasp.
    moveit : MoveItClient
        The planning back end.
    params : dict
        The contents of ``config/planner.yaml``, already read off the node.
    look_up : Callable[[str], object]
        Resolves an object name to the world model's answer.
    grasp : Callable[[str, bool], None]
        Attaches or releases an object; raises on failure.
    """

    def __init__(self, node, moveit: MoveItClient, params: Dict, look_up: Callable, grasp: Callable):
        self.node = node
        self.moveit = moveit
        self.params = params
        self.look_up = look_up
        self.grasp = grasp
        self.held: Optional[Held] = None

    # -- the actions -----------------------------------------------------

    def execute(self, action_name: str, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        """Run one known action. Raises PlanningError if it cannot."""
        handler = self.DISPATCH.get(action_name)
        if handler is None:
            raise PlanningError(
                f"unknown action '{action_name}'; known actions are "
                f"{', '.join(sorted(self.DISPATCH))}"
            )
        return handler(self, target, pose, report)

    def _home(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        report("home", 0.0)
        plan = self._plan_to_joints(self.params["home_positions"])
        self.moveit.execute(plan)
        report("home", 1.0)
        return ActionOutcome("at home", plan.path_length, plan.planning_time)

    def _move_to(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        goal = pose if not target else self._hover_pose(target)
        if goal is None:
            raise PlanningError("move_to needs either a target object or a pose")
        report("move", 0.0)
        plan = self._plan_to_pose(goal)
        self.moveit.execute(plan)
        report("move", 1.0)
        where = target or "the given pose"
        return ActionOutcome(f"tool at {where}", plan.path_length, plan.planning_time)

    def _pick(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        if not target:
            raise PlanningError("pick needs the name of an object")
        if self.held is not None:
            raise PlanningError(f"already holding {self.held.name}")

        report("locate", 0.0)
        obj = self._require(target)
        grasp_pose = self._grasp_pose(obj)
        approach = raised(grasp_pose, self.params["approach_distance"])

        report("approach", 0.2)
        total = self._go_to(approach)

        report("descend", 0.4)
        total += self._go_straight([grasp_pose])

        report("grasp", 0.6)
        self.grasp(target, True)
        self.held = Held(name=target, height=float(obj.dimensions.z))

        report("retreat", 0.8)
        try:
            total += self._go_straight([approach])
        except PlanningError:
            # The object is in the hand either way; leaving the arm down there
            # would be worse than a joint-space lift.
            self.node.get_logger().warn("straight retreat failed, lifting through joint space")
            total += self._go_to(approach)

        report("done", 1.0)
        return ActionOutcome(f"picked {target}", total.path_length, total.planning_time)

    def _place(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        if self.held is None:
            raise PlanningError("nothing is held, so there is nothing to place")
        if not target and pose is None:
            raise PlanningError("place needs a target container or a pose")

        report("locate", 0.0)
        release = pose if pose is not None and not target else self._release_pose(self._require(target))
        approach = raised(release, self.params["approach_distance"])

        report("approach", 0.2)
        total = self._go_to(approach)

        report("descend", 0.4)
        total += self._go_straight([release])

        report("release", 0.6)
        placed = self.held.name
        self.grasp(placed, False)
        self.held = None

        report("retreat", 0.8)
        total += self._go_straight([approach])

        report("done", 1.0)
        return ActionOutcome(f"placed {placed} in {target}", total.path_length, total.planning_time)

    DISPATCH: Dict[str, Callable] = {
        "home": _home,
        "move_to": _move_to,
        "pick": _pick,
        "place": _place,
    }

    # -- poses -----------------------------------------------------------

    def _require(self, name: str):
        """Ask the world model where something is, and insist on an answer."""
        found = self.look_up(name)
        if found is None or not found.found:
            raise PlanningError(f"the world model does not know an object called '{name}'")
        return found

    def _grasp_pose(self, obj) -> Pose:
        """Where the tool sits to grasp an object from above.

        The tool frame is put just over the object's top face, so that the
        object hangs under it once attached, and turned to line up with it --
        folded to the nearest quarter turn, since the objects are cubes.
        """
        position = obj.pose.pose.position
        yaw = square_symmetric_yaw(yaw_of(obj.pose.pose.orientation))
        top = position.z + obj.dimensions.z / 2.0
        return pose_above(position.x, position.y, top + self.params["tool_gap"], yaw)

    def _hover_pose(self, name: str) -> Pose:
        """A pose over an object, at the height the approach starts from."""
        obj = self._require(name)
        position = obj.pose.pose.position
        top = position.z + obj.dimensions.z / 2.0
        height = top + self.params["tool_gap"] + self.params["approach_distance"]
        return pose_above(position.x, position.y, height)

    def _release_pose(self, container) -> Pose:
        """Where the tool sits to drop the held object into a container.

        The object is let go just above the container's rim rather than lowered
        into it: the fake grasp has no compliance, and the last centimetres of
        a 10 cm cube into a tray of its own size are exactly where a rigid
        placement jams.
        """
        position = container.pose.pose.position
        rim = position.z + container.dimensions.z / 2.0
        height = rim + self.params["drop_gap"] + self.params["tool_gap"] + self.held.height
        return pose_above(position.x, position.y, height)

    # -- motion ----------------------------------------------------------

    def _plan_to_joints(self, joint_values: List[float]) -> Plan:
        return self.moveit.plan_to_joint_goal(
            joint_names=self.params["joint_names"],
            joint_values=joint_values,
            planner_id=self.params["planner_id"],
            attempts=self.params["planning_attempts"],
            allowed_time=self.params["allowed_planning_time"],
            velocity_scaling=self.params["velocity_scaling"],
            acceleration_scaling=self.params["acceleration_scaling"],
        )

    def _plan_to_pose(self, pose: Pose) -> Plan:
        joint_values = self.moveit.inverse_kinematics(pose, self.params["joint_names"])
        return self._plan_to_joints(joint_values)

    def _go_to(self, pose: Pose) -> ActionOutcome:
        """Plan freely to a pose and execute."""
        plan = self._plan_to_pose(pose)
        self.moveit.execute(plan)
        return ActionOutcome("", plan.path_length, plan.planning_time)

    def _go_straight(self, waypoints: List[Pose]) -> ActionOutcome:
        """Move the tool in a straight line and execute."""
        plan = self.moveit.plan_cartesian(
            waypoints,
            velocity_scaling=self.params["velocity_scaling"],
            acceleration_scaling=self.params["acceleration_scaling"],
            step=self.params["cartesian_step"],
            min_fraction=self.params["cartesian_min_fraction"],
        )
        self.moveit.execute(plan)
        return ActionOutcome("", plan.path_length, plan.planning_time)

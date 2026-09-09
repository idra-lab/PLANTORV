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

"""The four things the planner asks MoveIt for.

MoveIt's C++ ``MoveGroupInterface`` has no rclpy equivalent on Humble, so this
is the small part of it the planner needs, talking to move_group's own
interfaces directly: ``/compute_ik`` for a joint goal, ``/move_action`` to plan
to it, ``/compute_cartesian_path`` for the straight approach and retreat, and
``/execute_trajectory`` to run any of it.

Every call here blocks the caller until move_group answers. That is safe, and
only safe, because the node that owns this client puts it in a reentrant
callback group and spins a multi-threaded executor -- the response arrives on
another thread while this one waits.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    PlanningOptions,
    RobotState,
    RobotTrajectory,
    WorkspaceParameters,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState

from plantorv_planner.geometry import joint_path_length

# How long to wait for any one call to move_group before giving up.
CALL_TIMEOUT = 30.0


class PlanningError(RuntimeError):
    """MoveIt could not do what was asked. The message says what was asked."""


@dataclass
class Plan:
    """A trajectory, and what it cost to find it."""

    trajectory: RobotTrajectory
    path_length: float
    planning_time: float

    @property
    def is_empty(self) -> bool:
        return not self.trajectory.joint_trajectory.points


class MoveItClient:
    """A thin, blocking client for move_group.

    Parameters
    ----------
    node : rclpy.node.Node
        The node the interfaces are created on.
    group : str
        Planning group, ``ur_manipulator`` here.
    tool_link : str
        Link the Cartesian moves and the IK are expressed for.
    planning_frame : str
        Frame goals are given in.
    """

    def __init__(self, node: Node, group: str, tool_link: str, planning_frame: str):
        self.node = node
        self.group = group
        self.tool_link = tool_link
        self.planning_frame = planning_frame

        callbacks = ReentrantCallbackGroup()
        self.move_client = ActionClient(node, MoveGroup, "/move_action", callback_group=callbacks)
        self.execute_client = ActionClient(
            node, ExecuteTrajectory, "/execute_trajectory", callback_group=callbacks
        )
        self.ik_client = node.create_client(GetPositionIK, "/compute_ik", callback_group=callbacks)
        self.cartesian_client = node.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=callbacks
        )

        self.joint_state: Optional[JointState] = None
        node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10, callback_group=callbacks
        )

    # -- state -----------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_state = msg

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until move_group and the joint states are both there."""
        deadline = time.monotonic() + timeout
        ready = self.move_client.wait_for_server(timeout_sec=timeout)
        ready = ready and self.execute_client.wait_for_server(timeout_sec=5.0)
        ready = ready and self.ik_client.wait_for_service(timeout_sec=5.0)
        ready = ready and self.cartesian_client.wait_for_service(timeout_sec=5.0)
        while ready and self.joint_state is None and time.monotonic() < deadline:
            time.sleep(0.1)
        return ready and self.joint_state is not None

    def current_state(self) -> RobotState:
        """The arm's current configuration, as a seed and as a plan start."""
        state = RobotState()
        if self.joint_state is not None:
            state.joint_state = self.joint_state
        state.is_diff = False
        return state

    def current_positions(self, joint_names: Sequence[str]) -> List[float]:
        """Current values of the named joints, in the order given."""
        if self.joint_state is None:
            raise PlanningError("no /joint_states yet; is the controller running?")
        lookup = dict(zip(self.joint_state.name, self.joint_state.position))
        missing = [name for name in joint_names if name not in lookup]
        if missing:
            raise PlanningError(f"joints missing from /joint_states: {missing}")
        return [float(lookup[name]) for name in joint_names]

    # -- inverse kinematics ----------------------------------------------

    def inverse_kinematics(
        self, pose: Pose, joint_names: Sequence[str], attempts: int = 5, timeout: float = 0.2
    ) -> List[float]:
        """Return a collision-free configuration putting the tool at ``pose``.

        KDL is a local solver seeded from wherever the arm happens to be, so a
        single failure means little; it is retried, and only a run of failures
        is taken as "unreachable".
        """
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group
        request.ik_request.ik_link_name = self.tool_link
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state = self.current_state()
        request.ik_request.timeout = _duration(timeout)

        stamped = PoseStamped()
        stamped.header.frame_id = self.planning_frame
        stamped.pose = pose
        request.ik_request.pose_stamped = stamped

        last_error = MoveItErrorCodes.FAILURE
        for _ in range(max(1, attempts)):
            response = self._call(self.ik_client, request)
            if response is None:
                raise PlanningError("/compute_ik did not answer")
            if response.error_code.val == MoveItErrorCodes.SUCCESS:
                lookup = dict(
                    zip(response.solution.joint_state.name, response.solution.joint_state.position)
                )
                return [float(lookup[name]) for name in joint_names]
            last_error = response.error_code.val

        raise PlanningError(
            f"no IK solution for the tool at "
            f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) "
            f"[MoveIt error {last_error}]"
        )

    # -- planning --------------------------------------------------------

    def plan_to_joint_goal(
        self,
        joint_names: Sequence[str],
        joint_values: Sequence[float],
        planner_id: str,
        attempts: int,
        allowed_time: float,
        velocity_scaling: float,
        acceleration_scaling: float,
        tolerance: float = 1e-4,
    ) -> Plan:
        """Plan to a configuration, several times, and keep the shortest.

        This is where "optimal" is decided. RRTConnect is quick but its output
        is whatever the tree happened to grow, and two runs from the same state
        differ a lot; running it ``attempts`` times and keeping the shortest
        path costs a fraction of a second and reliably removes the worst of
        those. A single run of an asymptotically optimal planner would be the
        alternative -- set ``planner_id`` to RRTstar for that -- but it spends
        its whole time budget every time, whether or not the first path was
        already good.
        """
        request = MotionPlanRequest()
        request.group_name = self.group
        request.planner_id = planner_id
        request.num_planning_attempts = 1
        request.allowed_planning_time = allowed_time
        request.max_velocity_scaling_factor = velocity_scaling
        request.max_acceleration_scaling_factor = acceleration_scaling
        request.start_state.is_diff = True

        workspace = WorkspaceParameters()
        workspace.header.frame_id = self.planning_frame
        workspace.min_corner.x, workspace.min_corner.y, workspace.min_corner.z = -2.0, -2.0, 0.0
        workspace.max_corner.x, workspace.max_corner.y, workspace.max_corner.z = 2.0, 2.0, 3.0
        request.workspace_parameters = workspace

        constraints = Constraints()
        for name, value in zip(joint_names, joint_values):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(value)
            constraint.tolerance_above = tolerance
            constraint.tolerance_below = tolerance
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        request.goal_constraints = [constraints]

        options = PlanningOptions()
        options.plan_only = True
        options.planning_scene_diff.is_diff = True
        options.planning_scene_diff.robot_state.is_diff = True

        best: Optional[Plan] = None
        total_time = 0.0
        failures: List[int] = []
        for _ in range(max(1, attempts)):
            goal = MoveGroup.Goal()
            goal.request = request
            goal.planning_options = options
            result = self._send_action(self.move_client, goal)
            if result is None:
                failures.append(MoveItErrorCodes.TIMED_OUT)
                continue
            total_time += float(result.planning_time)
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                failures.append(result.error_code.val)
                continue
            candidate = Plan(
                trajectory=result.planned_trajectory,
                path_length=joint_path_length(
                    point.positions for point in result.planned_trajectory.joint_trajectory.points
                ),
                planning_time=float(result.planning_time),
            )
            if best is None or candidate.path_length < best.path_length:
                best = candidate

        if best is None:
            raise PlanningError(f"no plan found in {attempts} attempts [MoveIt errors {failures}]")
        best.planning_time = total_time
        return best

    def plan_cartesian(
        self,
        waypoints: Sequence[Pose],
        velocity_scaling: float,
        acceleration_scaling: float,
        step: float = 0.005,
        min_fraction: float = 0.9,
    ) -> Plan:
        """Plan a straight tool move through ``waypoints``.

        Used for the approach and the retreat, where the arm has to come
        straight down onto an object and straight back up: a joint-space plan
        for the same endpoints is free to swing the tool sideways through the
        object it is reaching for.
        """
        request = GetCartesianPath.Request()
        request.header.frame_id = self.planning_frame
        request.start_state = self.current_state()
        request.group_name = self.group
        request.link_name = self.tool_link
        request.waypoints = list(waypoints)
        request.max_step = step
        # Reject a solution that changes configuration partway along the line.
        #
        # This must not be 0. Zero disables the check, and then consecutive
        # waypoints are free to come back in different IK branches: every one
        # of them is collision-free, so the returned fraction is 1.0 and the
        # path looks good, but the controller interpolates between them in
        # joint space and the arm sweeps through whatever lies between the two
        # configurations. That is what put the forearm through the stand and
        # the table -- a traverse that should have cost about 2 rad of joint
        # motion came back costing 23.5, all of it in one flip between two
        # adjacent waypoints.
        #
        # The value is a multiple of the average joint motion per step, so it
        # scales with max_step rather than being an absolute angle. 5.0 is the
        # MoveIt default and leaves normal motion alone.
        request.jump_threshold = 5.0
        request.avoid_collisions = True
        # These two were added to GetCartesianPath after Humble's first
        # release. Setting them when they are there gets the returned path
        # time-parameterised the same way a free-space plan is; setting them
        # when they are not would raise.
        for field, value in (
            ("max_velocity_scaling_factor", velocity_scaling),
            ("max_acceleration_scaling_factor", acceleration_scaling),
        ):
            if hasattr(request, field):
                setattr(request, field, value)

        response = self._call(self.cartesian_client, request)
        if response is None:
            raise PlanningError("/compute_cartesian_path did not answer")
        if response.fraction < min_fraction:
            raise PlanningError(
                f"only {response.fraction * 100:.0f}% of the straight-line move is free"
            )
        return Plan(
            trajectory=response.solution,
            path_length=joint_path_length(
                point.positions for point in response.solution.joint_trajectory.points
            ),
            planning_time=0.0,
        )

    # -- execution -------------------------------------------------------

    def execute(self, plan: Plan) -> None:
        """Run a trajectory on the controller and wait for it to finish."""
        if plan.is_empty:
            return
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan.trajectory
        result = self._send_action(self.execute_client, goal, timeout=CALL_TIMEOUT * 4)
        if result is None:
            raise PlanningError("execution did not finish in time")
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise PlanningError(f"execution failed [MoveIt error {result.error_code.val}]")

    # -- plumbing --------------------------------------------------------

    def _call(self, client, request, timeout: float = CALL_TIMEOUT):
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=timeout):
            return None
        return _await(client.call_async(request), timeout)

    def _send_action(self, client, goal, timeout: float = CALL_TIMEOUT):
        if not client.wait_for_server(timeout_sec=timeout):
            return None
        handle = _await(client.send_goal_async(goal), timeout)
        if handle is None or not handle.accepted:
            return None
        wrapper = _await(handle.get_result_async(), timeout)
        return wrapper.result if wrapper is not None else None


def _await(future, timeout: float):
    """Wait for a future without blocking the executor that completes it."""
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    return future.result() if future.done() else None


def _duration(seconds: float) -> DurationMsg:
    return DurationMsg(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))

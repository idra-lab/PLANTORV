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

"""The planner: one action server, four known actions.

``plantorv_interfaces/action/ExecuteAction`` is the whole interface. A caller
-- normally a behaviour tree leaf -- names an action and an object; the server
plans it, executes it, and reports what it cost. Goals are served one at a
time, because there is one arm.
"""

import time
from typing import Optional

import rclpy
from plantorv_interfaces.action import ExecuteAction
from plantorv_interfaces.srv import AttachObject, GetObjectPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from plantorv_planner.actions import ActionLibrary
from plantorv_planner.moveit_client import MoveItClient, PlanningError

DEFAULTS = {
    "planning_group": "ur_manipulator",
    "tool_link": "tool0",
    "planning_frame": "world",
    "joint_names": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
    # Mirrors the `home` group state in plantorv_moveit_config's SRDF. The
    # MoveGroup action takes joint values, not named targets -- resolving a
    # name is a client-side job in MoveIt, and this is the client.
    "home_positions": [0.0, -1.5707963, 1.5707963, -1.5707963, -1.5707963, 0.0],
    "planner_id": "RRTConnect",
    "planning_attempts": 6,
    "allowed_planning_time": 2.0,
    "velocity_scaling": 0.25,
    "acceleration_scaling": 0.25,
    "approach_distance": 0.12,
    # Height of the transit plane, in the planning frame. Every action starts
    # and ends with the tool on it, and moves between objects run along it.
    # A straight line between two points of equal height stays at that height,
    # so a traverse cannot dip into the table however far it goes -- which is
    # the whole reason the moves are built this way rather than left to a
    # sampling planner. It has to clear the tallest thing on the table (the
    # tray rims at z = 1.135) by more than the length of a carried cube, and
    # stay inside the arm's reach: at 1.32 the furthest transit point, above a
    # tray, is 0.473 m from base_link against the UR3's 0.5 m.
    "transit_height": 1.32,
    "tool_gap": 0.005,
    "drop_gap": 0.03,
    "cartesian_step": 0.005,
    "cartesian_min_fraction": 0.85,
}


class PlannerNode(Node):
    """Serves ExecuteAction goals against MoveIt."""

    def __init__(self):
        super().__init__("planner")

        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)
        self.params = {name: self.get_parameter(name).value for name in DEFAULTS}

        clients = ReentrantCallbackGroup()
        self.pose_client = self.create_client(
            GetObjectPose, "/get_object_pose", callback_group=clients
        )
        self.attach_client = self.create_client(
            AttachObject, "/attach_object", callback_group=clients
        )

        self.moveit = MoveItClient(
            self,
            group=self.params["planning_group"],
            tool_link=self.params["tool_link"],
            planning_frame=self.params["planning_frame"],
        )
        self.library = ActionLibrary(
            node=self,
            moveit=self.moveit,
            params=self.params,
            look_up=self._look_up,
            grasp=self._grasp,
        )

        self.server = ActionServer(
            self,
            ExecuteAction,
            "execute_action",
            execute_callback=self._execute,
            goal_callback=self._accept,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            "planner up; known actions: " + ", ".join(sorted(ActionLibrary.DISPATCH))
        )

    # -- action server ---------------------------------------------------

    def _accept(self, goal_request) -> GoalResponse:
        if goal_request.action_name not in ActionLibrary.DISPATCH:
            self.get_logger().warn(f"rejecting unknown action '{goal_request.action_name}'")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        request = goal_handle.request
        result = ExecuteAction.Result()

        def report(stage: str, progress: float) -> None:
            feedback = ExecuteAction.Feedback()
            feedback.stage = stage
            feedback.progress = float(progress)
            goal_handle.publish_feedback(feedback)

        overrides = {}
        if request.velocity_scaling > 0.0:
            overrides["velocity_scaling"] = request.velocity_scaling
        if request.acceleration_scaling > 0.0:
            overrides["acceleration_scaling"] = request.acceleration_scaling

        started = time.monotonic()
        with _temporarily(self.params, overrides):
            try:
                self._check_frame(request)
                outcome = self.library.execute(
                    request.action_name,
                    request.target,
                    request.pose.pose if request.pose.header.frame_id else None,
                    report,
                )
            except PlanningError as error:
                goal_handle.abort()
                result.success = False
                result.message = str(error)
                self.get_logger().error(f"{request.action_name}: {error}")
                return result

        goal_handle.succeed()
        result.success = True
        result.message = outcome.message
        result.path_length = outcome.path_length
        result.planning_time = _duration(outcome.planning_time)
        self.get_logger().info(
            f"{request.action_name} {request.target}: {outcome.message} "
            f"(path {outcome.path_length:.2f} rad, planning {outcome.planning_time:.2f} s, "
            f"wall {time.monotonic() - started:.1f} s)"
        )
        return result

    def _check_frame(self, request) -> None:
        """Poses are taken in the planning frame, and nothing else.

        Accepting another frame would mean transforming it, and quietly
        planning to the wrong place if TF happened not to have it yet is worse
        than saying so.
        """
        frame = request.pose.header.frame_id
        if frame and frame != self.params["planning_frame"]:
            raise PlanningError(
                f"pose is in frame '{frame}'; the planner works in "
                f"'{self.params['planning_frame']}'"
            )

    # -- the two things outside MoveIt -----------------------------------

    def _look_up(self, name: str) -> Optional[GetObjectPose.Response]:
        request = GetObjectPose.Request()
        request.name = name
        return self._call(self.pose_client, request)

    def _grasp(self, object_id: str, attach: bool) -> None:
        request = AttachObject.Request()
        request.object_id = object_id
        request.link_name = self.params["tool_link"]
        request.attach = attach
        response = self._call(self.attach_client, request)
        if response is None:
            raise PlanningError("the scene manager did not answer the grasp request")
        if not response.success:
            raise PlanningError(response.message)

    def _call(self, client, request, timeout: float = 15.0):
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        return future.result() if future.done() else None


class _temporarily:
    """Apply per-goal parameter overrides for the length of one goal."""

    def __init__(self, params: dict, overrides: dict):
        self.params = params
        self.overrides = overrides
        self.saved = {}

    def __enter__(self):
        for key, value in self.overrides.items():
            self.saved[key] = self.params[key]
            self.params[key] = value
        return self.params

    def __exit__(self, *exc_info):
        self.params.update(self.saved)
        return False


def _duration(seconds: float):
    from builtin_interfaces.msg import Duration

    return Duration(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Waiting here rather than in the constructor keeps the node spinning, so
    # the joint states this needs can actually arrive.
    import threading

    def wait_for_moveit():
        if node.moveit.wait_until_ready(timeout=120.0):
            node.get_logger().info("move_group and the controllers are up")
        else:
            node.get_logger().error("move_group or /joint_states never appeared")

    threading.Thread(target=wait_for_moveit, daemon=True).start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

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

__maintainers__ = ["Enrico Saccon", "Tommaso Faraci"]

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
from gripper_interfaces.srv import Gripper
from plantorv_interfaces.srv import AttachObject, GetObjectPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from plantorv_planner.actions import ActionLibrary
from plantorv_planner.cartesian_client import CartesianClient
from plantorv_planner.controllers import ControllerSwitcher
from plantorv_planner.errors import PlanningError
from plantorv_planner.joint_client import JointTrajectoryClient
from plantorv_planner.link_guard import LinkGuard
from plantorv_planner.moveit_client import MoveItClient

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
    #
    # Measured on the real cell, in degrees: 90.37, -107.23, -85.97, -80.84,
    # 90.11, 2.16. Puts tool0 at (0.110, 0.360, 1.223) pointing down.
    "home_positions": [1.577254, -1.8715166, -1.5004596, -1.4109242, 1.5727162, 0.0376991],
    "planner_id": "RRTConnect",
    "planning_attempts": 6,
    "allowed_planning_time": 2.0,
    "velocity_scaling": 0.25,
    "acceleration_scaling": 0.25,
    "approach_distance": 0.12,
    # Height of the transit plane, in the planning frame. Every action starts
    # and ends with the tool on it, and moves between objects run along it.
    # A straight line between two points of equal height stays at that height,
    # so a traverse cannot dip into the table however far it goes. It has to
    # clear the tallest thing on the table by more than the length of a
    # carried block, and stay inside the arm's reach. The tray rims are only
    # 2.5 cm, so it is the 6 cm blocks at z = 0.81 that set it, not the trays.
    # Measured, not derived: the height of tool0 when the arm is at the taught
    # home configuration, read off TF on the real cell. See planner.yaml.
    "transit_height": 1.223,
    "tool_gap": 0.005,
    "drop_gap": 0.03,
    "cartesian_step": 0.005,
    "cartesian_min_fraction": 0.85,
    # Which back end runs the straight moves: "cartesian" for
    # cartesian_motion_controller, "moveit" for /compute_cartesian_path. The
    # second is kept so the two can be compared on the same trees, not because
    # it is a supported way to run the cell.
    "motion_backend": "cartesian",
    # Whether to start a MoveIt client at all.
    #
    # False by default. Nothing in the normal cycle needs it any more: `home`
    # is a joint-space move sent straight to the trajectory controller, and the
    # straight moves go to cartesian_motion_controller. What is lost is the
    # free-space fallback in `pick`, which is refused with a message rather
    # than replaced, and the "moveit" motion_backend used to compare the two.
    #
    # Turning it off also removes move_group and OMPL from the picture, which
    # is the point: two fewer things to interrogate when a move does not
    # happen.
    "use_moveit": False,
    # Set true to have every joint-space move build and log its trajectory and
    # send nothing. The way to see what `home` would do before it does it.
    "dry_run": False,
    # Joint-space limits for `home`, before the scaling factors. Slower than
    # the arm can go, because the route is a straight line in joint space and
    # nothing checks what it passes through.
    "joint_speed": 0.40,  # rad/s
    "joint_accel": 0.80,  # rad/s^2
    # Radius the arm's links are inflated by for the link guard, in metres. A
    # UR3's links are about 40 mm in radius; the rest is margin, and it is also
    # how far short of contact a Cartesian move is stopped, since that check
    # runs on the measured configuration rather than a predicted one.
    "link_radius": 0.06,
    # The Robotiq 2f85 driver in grippers_ur_ros2. It is a plain service, not
    # ros2_control: the node tunnels to the robot's RS485 URCap and drives the
    # gripper over Modbus. Off in simulation, where there is nothing to talk to
    # and pick still uses the fake grasp.
    "use_gripper": False,
    "gripper_service": "/r2f85_gripper",
    "gripper_timeout": 15.0,
    # Turn the whole-arm check off. Only for comparing against the behaviour
    # before it existed; nothing should run this way on hardware.
    "use_link_guard": True,
    "trajectory_controller": "joint_trajectory_controller",
    "cartesian_controller": "cartesian_motion_controller",
    # The controller works in its own robot_base_link and refuses a target in
    # any other frame. Setting it to the planning frame -- which the workcell
    # URDF is rooted at -- means no pose has to be transformed. Against a
    # robot_description rooted at the arm instead, set this to base_link and
    # the client transforms through TF.
    "controller_base_frame": "world",
    # The link the arm is mounted on. Distinct from controller_base_frame: that
    # is the frame the controller accepts targets in, this is what the reach
    # guard measures from. Setting them to the same thing when the URDF is
    # rooted at the cell measures reach from the middle of the desk.
    "arm_base_frame": "base_link",
    # Tool speed at velocity_scaling = 1.0. The scaling factors multiply
    # these, so what the arm actually does is a quarter of this by default --
    # get that wrong and a pick takes a minute. See planner.yaml.
    "cartesian_speed": 0.60,  # m/s
    "cartesian_accel": 1.50,  # m/s^2
    "cartesian_rot_speed": 4.00,  # rad/s
    # Rate the target pose is streamed at. Well under the controller's own
    # update rate; it converges towards each target rather than tracking it
    # point by point, so there is nothing to gain from matching that rate.
    "cartesian_rate": 125.0,  # Hz
    "cartesian_pos_tolerance": 0.002,  # m
    # How near is near enough at the end of a move that finishes in open air.
    # The controller closes on a target asymptotically, so the last millimetre
    # costs many times what the first centimetre does; paying for it at the end
    # of a traverse buys nothing, because the descent that follows is aimed at
    # an absolute pose and corrects whatever is left over.
    "cartesian_transit_tolerance": 0.005,  # m
    "cartesian_rot_tolerance": 0.02,  # rad
    "cartesian_timeout": 20.0,  # s
    # How long the tool may go without getting closer before the move is
    # called failed. The solver sits happily short of an unreachable pose, so
    # without this a bad target costs the whole timeout with the arm pulling.
    "cartesian_stall_time": 1.5,  # s
    # The volume the tool may move in, checked on every commanded path since
    # nothing else checks anything. Both reach bounds are measured from
    # base_link, and both come from the layout in plantorv_sim's scene.yaml
    # rather than from the arm's datasheet -- see the note in planner.yaml.
    "workspace_min_z": 0.76,
    "workspace_min_reach": 0.16,
    "workspace_max_reach": 0.48,
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
        self.gripper_client = None
        if self.params["use_gripper"]:
            self.gripper_client = self.create_client(
                Gripper, self.params["gripper_service"], callback_group=clients
            )

        self.switcher = ControllerSwitcher(
            self,
            known=[
                self.params["trajectory_controller"],
                self.params["cartesian_controller"],
            ],
            callback_group=clients,
        )
        self.guard = None
        if self.params["use_link_guard"]:
            self.guard = LinkGuard(
                self, joint_names=self.params["joint_names"], callback_group=clients
            )
        self.joints = JointTrajectoryClient(
            self,
            controller=self.params["trajectory_controller"],
            joint_names=self.params["joint_names"],
            switcher=self.switcher,
            callback_group=clients,
            guard=self.guard,
        )
        self.moveit = None
        if self.params["use_moveit"]:
            self.moveit = MoveItClient(
                self,
                group=self.params["planning_group"],
                tool_link=self.params["tool_link"],
                planning_frame=self.params["planning_frame"],
                switcher=self.switcher,
                controller=self.params["trajectory_controller"],
            )
        self.cartesian = CartesianClient(
            self,
            controller=self.params["cartesian_controller"],
            tool_link=self.params["tool_link"],
            planning_frame=self.params["planning_frame"],
            controller_frame=self.params["controller_base_frame"],
            arm_base_frame=self.params["arm_base_frame"],
            joint_names=self.params["joint_names"],
            switcher=self.switcher,
            callback_group=clients,
            guard=self.guard,
        )
        self.library = ActionLibrary(
            node=self,
            moveit=self.moveit,
            cartesian=self.cartesian,
            joints=self.joints,
            params=self.params,
            look_up=self._look_up,
            grasp=self._grasp,
            gripper=self._gripper if self.gripper_client is not None else None,
        )

        # The action server is deliberately NOT created here.
        #
        # A caller waits for this server and then sends a goal, so the server
        # existing has to mean the planner can serve one. Created in the
        # constructor it appears at once, before /joint_states has arrived and
        # before the link guard has a description, and the first goal is
        # refused with "no /joint_states yet" a fraction of a second before the
        # back ends report ready. MoveIt used to hide that by taking seconds to
        # start; without it the planner comes up fast enough to lose the race
        # every time.
        #
        # start_serving() creates it, and is called once the back ends are up.
        self.server: Optional[ActionServer] = None
        self.get_logger().info(
            "planner starting; known actions: " + ", ".join(sorted(ActionLibrary.DISPATCH))
        )

    def start_serving(self) -> None:
        """Create the action server, now that a goal can actually be served."""
        if self.server is not None:
            return
        self.server = ActionServer(
            self,
            ExecuteAction,
            "execute_action",
            execute_callback=self._execute,
            goal_callback=self._accept,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info("planner ready; accepting goals on /execute_action")

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

    def _gripper(self, command: str) -> str:
        """One command to the gripper driver. Returns what it said.

        The driver matches on the first letter, so "open" and "o" are the same
        request; the words are used here because they read better in a log.
        """
        request = Gripper.Request()
        request.command = command
        response = self._call(
            self.gripper_client, request, timeout=self.params["gripper_timeout"]
        )
        if response is None:
            raise PlanningError(
                f"the gripper driver did not answer '{command}' within "
                f"{self.params['gripper_timeout']:.0f} s; is robotiq_2f85 running and "
                f"is the RS485 URCap started on the pendant?"
            )
        if not response.success:
            raise PlanningError(f"the gripper refused '{command}': {response.status}")
        return response.status

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

    def wait_for_back_ends():
        if node.moveit is not None:
            if not node.moveit.wait_until_ready(timeout=120.0):
                node.get_logger().error("move_group or /joint_states never appeared")
                return
            node.get_logger().info("move_group and the controllers are up")
        else:
            node.get_logger().info("running without MoveIt; `home` goes to the controller")

        if node.guard is not None:
            if node.guard.wait_until_ready(timeout=60.0):
                node.get_logger().info("link guard ready; the whole arm is checked")
            else:
                node.get_logger().error(
                    "/robot_description never arrived, so the link guard has no "
                    "obstacles; every move will be refused"
                )
        else:
            node.get_logger().warn(
                "link guard disabled; nothing checks what the arm passes through"
            )

        if node.joints.wait_until_ready(timeout=120.0):
            node.get_logger().info(
                f"joint back end ready on '{node.params['trajectory_controller']}'"
            )
        else:
            node.get_logger().error(
                f"'{node.params['trajectory_controller']}' has no "
                f"follow_joint_trajectory action, or no /joint_states arrived; "
                f"`home` will fail"
            )
            return

        if node.cartesian.wait_until_ready(timeout=60.0):
            node.get_logger().info(
                f"cartesian back end ready; straight moves run on "
                f"'{node.params['motion_backend']}'"
            )
        else:
            node.get_logger().error(
                "the controller manager or the tool transform never appeared; "
                "straight moves will fail"
            )
            return

        # Only now does /execute_action appear, so a caller that waited for it
        # can send a goal and have it served.
        node.start_serving()

    threading.Thread(target=wait_for_back_ends, daemon=True).start()

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

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

"""The leaves that talk to the rest of the system.

Two kinds. The motion leaves -- ``Home``, ``MoveTo``, ``Pick``, ``Place`` --
each send one goal to the planner and return RUNNING until it finishes, which
is what lets a tree react while the arm is moving. The query leaves --
``DetectObjects``, ``MatchingTray`` -- ask the world model and put the answer
on the blackboard.

Nothing here blocks. Ticks come from a timer on the executor node, and a tick
that waited for the robot would stop the very callbacks that deliver the
robot's answer.
"""

from typing import Optional

from plantorv_bt.core import Status, TreeNode
from plantorv_interfaces.action import ExecuteAction
from plantorv_interfaces.srv import GetObjectPose, ListObjects


class RosBridge:
    """The action and service clients the leaves share.

    One per executor node. Held as the tree's ``context``, so that leaves stay
    free of setup and a tree can be built and inspected without ROS.
    """

    def __init__(self, node):
        from rclpy.action import ActionClient

        self.node = node
        self.logger = node.get_logger()
        self.execute_client = ActionClient(node, ExecuteAction, "/execute_action")
        self.list_client = node.create_client(ListObjects, "/list_objects")
        self.pose_client = node.create_client(GetObjectPose, "/get_object_pose")

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until the planner and the world model are both up."""
        return (
            self.execute_client.wait_for_server(timeout_sec=timeout)
            and self.list_client.wait_for_service(timeout_sec=10.0)
            and self.pose_client.wait_for_service(timeout_sec=10.0)
        )


class PlannerLeaf(TreeNode):
    """Base of the leaves that send one goal to the planner.

    Subclasses say what the goal is; the state machine below -- send, wait for
    acceptance, wait for the result -- is the same for all of them, and is
    spread over ticks so that no tick ever waits.
    """

    #: The action_name field of the goal. Set by the subclass.
    ACTION = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_future = None
        self.result_future = None
        self.goal_handle = None

    # -- to be filled in by subclasses -----------------------------------

    def build_goal(self) -> Optional[ExecuteAction.Goal]:
        """Return the goal to send, or None to fail the leaf."""
        goal = ExecuteAction.Goal()
        goal.action_name = self.ACTION
        goal.target = self.port("target", "") or self.port("object", "") or ""
        return goal

    def on_result(self, result) -> None:
        """Hook for a subclass to record something from a finished goal."""

    # -- the state machine -----------------------------------------------

    def _tick(self) -> Status:
        bridge: RosBridge = self.context
        if self.goal_future is None and self.result_future is None:
            return self._send(bridge)
        if self.goal_future is not None:
            return self._await_acceptance(bridge)
        return self._await_result(bridge)

    def _send(self, bridge: RosBridge) -> Status:
        if not bridge.execute_client.server_is_ready():
            bridge.logger.error(f"{self.describe()}: the planner is not running")
            return Status.FAILURE
        goal = self.build_goal()
        if goal is None:
            return Status.FAILURE
        bridge.logger.info(f"{self.describe()}: sending {goal.action_name} '{goal.target}'")
        self.goal_future = bridge.execute_client.send_goal_async(goal)
        return Status.RUNNING

    def _await_acceptance(self, bridge: RosBridge) -> Status:
        if not self.goal_future.done():
            return Status.RUNNING
        handle = self.goal_future.result()
        self.goal_future = None
        if handle is None or not handle.accepted:
            bridge.logger.error(f"{self.describe()}: the planner rejected the goal")
            return Status.FAILURE
        self.goal_handle = handle
        self.result_future = handle.get_result_async()
        return Status.RUNNING

    def _await_result(self, bridge: RosBridge) -> Status:
        if not self.result_future.done():
            return Status.RUNNING
        wrapper = self.result_future.result()
        self._reset()
        if wrapper is None:
            return Status.FAILURE
        result = wrapper.result
        if not result.success:
            bridge.logger.warn(f"{self.describe()}: {result.message}")
            return Status.FAILURE
        bridge.logger.info(
            f"{self.describe()}: {result.message} ({result.path_length:.2f} rad)"
        )
        self.on_result(result)
        return Status.SUCCESS

    def _reset(self) -> None:
        self.goal_future = None
        self.result_future = None
        self.goal_handle = None

    def halt(self) -> None:
        """Cancel a goal in flight, so the arm stops when the tree moves on."""
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self._reset()
        super().halt()


class Home(PlannerLeaf):
    """``<Home/>`` -- move the arm to its home configuration."""

    ACTION = "home"


class MoveTo(PlannerLeaf):
    """``<MoveTo target="tray_red"/>`` or ``<MoveTo x=".." y=".." z=".."/>``.

    With a target, the tool goes to the approach height over that object; with
    coordinates, straight to the point they name, tool pointing down.

    ``<MoveTo target="cube_1" z="0.90"/>`` gives both: the object says where to
    stand, the z says how high. The height the scene would work out is derived
    from the block's top face and assumes tool0 is where the gripper grips,
    which it is not while there is a real gripper on the flange, so being able
    to say the height outright is what makes a taught height usable.
    """

    ACTION = "move_to"

    def build_goal(self):
        goal = super().build_goal()
        if goal.target:
            if "z" in self.attributes:
                goal.pose.header.frame_id = self.port("frame", "world")
                goal.pose.pose.position.z = float(self.port("z", 0.0, cast=float))
                goal.pose.pose.orientation.x = 1.0
                goal.pose.pose.orientation.w = 0.0
            return goal
        if "x" not in self.attributes:
            self.context.logger.error("MoveTo needs either a target or x, y and z")
            return None
        goal.pose.header.frame_id = self.port("frame", "world")
        goal.pose.pose.position.x = float(self.port("x", 0.0, cast=float))
        goal.pose.pose.position.y = float(self.port("y", 0.0, cast=float))
        goal.pose.pose.position.z = float(self.port("z", 0.0, cast=float))
        # Straight down, which is the only orientation this cell uses.
        goal.pose.pose.orientation.x = 1.0
        goal.pose.pose.orientation.w = 0.0
        return goal


class Pick(PlannerLeaf):
    """``<Pick object="{cube}"/>`` -- grasp an object from above."""

    ACTION = "pick"

    def on_result(self, result) -> None:
        # Remembering what is in the hand lets a tree write
        # <Place target="{tray}"/> without repeating the object.
        self.blackboard.set("held_object", self.port("object", "") or self.port("target", ""))


class Place(PlannerLeaf):
    """``<Place target="{tray}"/>`` -- release the held object over a tray."""

    ACTION = "place"

    def on_result(self, result) -> None:
        self.blackboard.set("held_object", "")


class OpenGripper(PlannerLeaf):
    """``<OpenGripper/>`` -- open the gripper and wait."""

    ACTION = "open_gripper"


class CloseGripper(PlannerLeaf):
    """``<CloseGripper/>`` -- close the gripper and wait.

    Succeeds when the gripper reports that it moved, which is not the same as
    reporting that it is holding something.
    """

    ACTION = "close_gripper"


class ServiceLeaf(TreeNode):
    """Base of the leaves that ask the world model something."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.future = None

    def start(self):
        """Return the future of the call to make, or None to fail."""
        raise NotImplementedError

    def finish(self, response) -> Status:
        """Turn the response into a status, writing any output ports."""
        raise NotImplementedError

    def _tick(self) -> Status:
        if self.future is None:
            self.future = self.start()
            if self.future is None:
                return Status.FAILURE
            return Status.RUNNING
        if not self.future.done():
            return Status.RUNNING
        response = self.future.result()
        self.future = None
        if response is None:
            return Status.FAILURE
        return self.finish(response)

    def halt(self) -> None:
        self.future = None
        super().halt()


class DetectObjects(ServiceLeaf):
    """``<DetectObjects type="cube" output_key="{cubes}"/>``

    Names of everything of a given type, onto the blackboard. This is the leaf
    the perception pipeline will eventually answer, which is why the tree asks
    for a type rather than for the contents of ``scene.yaml``.
    """

    def start(self):
        bridge: RosBridge = self.context
        if not bridge.list_client.service_is_ready():
            bridge.logger.error("DetectObjects: the world model is not running")
            return None
        request = ListObjects.Request()
        request.type_filter = self.port("type", "")
        return bridge.list_client.call_async(request)

    def finish(self, response) -> Status:
        names = list(response.names)
        self.write_port("output_key", names)
        self.context.logger.info(f"DetectObjects: {len(names)} found -- {', '.join(names)}")
        return Status.SUCCESS if names else Status.FAILURE


class MatchingTray(ServiceLeaf):
    """``<MatchingTray object="{cube}" output_key="{tray}"/>``

    The tray whose colour matches an object's. Sorting by colour is the task,
    and putting the rule here rather than in the planner keeps it visible in
    the tree, where it can be changed without touching any robot code.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color: Optional[str] = None

    def start(self):
        bridge: RosBridge = self.context
        if not bridge.pose_client.service_is_ready():
            bridge.logger.error("MatchingTray: the world model is not running")
            return None
        name = self.port("object", "")
        if not name:
            bridge.logger.error("MatchingTray needs an object")
            return None
        if self.color is None:
            request = GetObjectPose.Request()
            request.name = name
            return bridge.pose_client.call_async(request)
        request = ListObjects.Request()
        request.type_filter = self.port("container_type", "tray")
        return bridge.list_client.call_async(request)

    def finish(self, response) -> Status:
        bridge: RosBridge = self.context
        # First call: the object's colour. Second: the containers to match it
        # against. Which one this is, is what self.color records.
        if self.color is None:
            if not getattr(response, "found", False):
                bridge.logger.warn(f"MatchingTray: no object called '{self.port('object', '')}'")
                return Status.FAILURE
            self.color = response.color
            return Status.RUNNING

        color, self.color = self.color, None
        for name, tray_color in zip(response.names, response.colors):
            if tray_color == color:
                self.write_port("output_key", name)
                bridge.logger.info(f"MatchingTray: {color} -> {name}")
                return Status.SUCCESS
        bridge.logger.warn(f"MatchingTray: nothing to hold a {color} object")
        return Status.FAILURE

    def halt(self) -> None:
        self.color = None
        super().halt()


class Log(TreeNode):
    """``<Log message="..."/>`` -- says something and succeeds."""

    def _tick(self) -> Status:
        self.context.logger.info(f"[tree] {self.port('message', '')}")
        return Status.SUCCESS


#: The leaves a tree may use, by the tag it writes them as.
LEAVES = {
    "Home": Home,
    "MoveTo": MoveTo,
    "Pick": Pick,
    "Place": Place,
    "OpenGripper": OpenGripper,
    "CloseGripper": CloseGripper,
    "DetectObjects": DetectObjects,
    "MatchingTray": MatchingTray,
    "Log": Log,
}

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
        from tf2_ros import Buffer, TransformListener

        self.node = node
        self.logger = node.get_logger()
        self.execute_client = ActionClient(node, ExecuteAction, "/execute_action")
        self.list_client = node.create_client(ListObjects, "/list_objects")
        self.pose_client = node.create_client(GetObjectPose, "/get_object_pose")

        # For the leaves that name a point in some other frame. The planner
        # takes poses in its planning frame and nothing else, on purpose, so
        # the conversion belongs on this side of the action.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        # Set from the node's parameter. Kept here rather than read from
        # the node, so a leaf asks its context and not the ROS graph.
        self.preview_seconds = MoveToPoint.DEFAULT_PREVIEW_SECONDS

        # Where a leaf shows what it is about to do. Latched, so RViz
        # picks the marker up even when it connects after the fact, and
        # so the last target stays on screen afterwards.
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from visualization_msgs.msg import MarkerArray

        self.marker_pub = node.create_publisher(
            MarkerArray,
            "/plantorv_bt/target_point",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        # The same target as a frame. A marker needs its own display in
        # RViz; a frame turns up in the TF display that is already
        # there, which is one thing less to have configured wrongly.
        from tf2_ros import StaticTransformBroadcaster

        self.static_tf = StaticTransformBroadcaster(node)

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


class MoveToPoint(PlannerLeaf):
    """``<MoveToPoint x="0.05" y="-0.02" z="0.74"/>`` -- a point seen by the camera.

    The three numbers are metres in ``frame``, which defaults to the colour
    camera's optical frame: x right across the image, y down it, z out along
    the lens. That is where a perception result naturally comes out, and it is
    not where the planner works, so the point is transformed into the planning
    frame here and the goal that leaves this leaf is in ``world`` like every
    other.

    Only the point is taken from ``frame``. The tool still points straight
    down in the planning frame, which is the only orientation this cell uses;
    carrying the camera's orientation across would aim the gripper along the
    lens.

    ``offset_z`` raises the goal above the point, in the planning frame.
    A camera names the surface of a thing; the goal commands tool0, and
    whatever hangs below tool0 needs that surface to be below it rather
    than through it.

    The transform has to be in TF already. If it is not -- the recorded
    transforms not running, or a frame named that nothing publishes -- the
    leaf fails and says so, rather than planning to a point it guessed.

    Before it moves anything the leaf shows where it is about to go, two
    ways: a marker on ``/plantorv_bt/target_point``, and a TF frame
    ``move_to_target`` under the planning frame. The frame is the one
    that needs nothing configured, since the TF display is already
    there; the marker adds the ray from the camera. Then it waits long
    enough to look before anything moves -- two minutes by default,
    ``preview_seconds="0"`` to go straight there.
    """

    ACTION = "move_to"

    #: Where a point is given when the XML does not say.
    DEFAULT_FRAME = "camera_color_optical_frame"

    #: Seconds to show the target before sending it, unless the XML says.
    DEFAULT_PREVIEW_SECONDS = 120.0

    #: The frame the target is broadcast as, unless the XML says.
    DEFAULT_TARGET_FRAME = "move_to_target"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previewed_goal = None
        self.preview_ends = None
        self.counted_down = None

    def _tick(self) -> Status:
        """Show the target, wait, then hand over to the usual machine."""
        import time

        # Once a goal is in flight the base class owns the leaf again.
        if self.goal_future is not None or self.result_future is not None:
            return super()._tick()

        if self.previewed_goal is None:
            goal = self.build_goal()

            if goal is None:
                return Status.FAILURE

            # The tree wins when it says so, otherwise the node's
            # parameter, otherwise the default here.
            seconds = float(
                self.port(
                    "preview_seconds",
                    getattr(
                        self.context,
                        "preview_seconds",
                        self.DEFAULT_PREVIEW_SECONDS,
                    ),
                    cast=float,
                )
            )

            if goal.pose.header.frame_id == "world":
                goal.pose.pose.position.z += 0.17
            else:
                self.context.logger.error("Frame is not world")

            self.previewed_goal = goal
            self.publish_marker(goal)
            self.publish_target_frame(goal)

            if seconds <= 0.0:
                return super()._tick()

            self.preview_ends = time.monotonic() + seconds
            self.counted_down = None

            listeners = self.context.marker_pub.get_subscription_count()

            self.context.logger.warn(
                f"{self.describe()}: about to move to "
                f"({goal.pose.pose.position.x:.3f}, "
                f"{goal.pose.pose.position.y:.3f}, "
                f"{goal.pose.pose.position.z:.3f}) in "
                f"{goal.pose.header.frame_id}. Shown on "
                f"/plantorv_bt/target_point for {seconds:.0f} s, "
                f"{listeners} subscriber(s) listening. Stop the tree "
                "now if that is not where it should go."
            )

            if listeners == 0:
                self.context.logger.warn(
                    f"{self.describe()}: nothing is subscribed to "
                    "/plantorv_bt/target_point. Add a MarkerArray "
                    "display on that topic in RViz, or check with "
                    "`ros2 topic echo /plantorv_bt/target_point`."
                )

            return Status.RUNNING

        remaining = self.preview_ends - time.monotonic()

        if remaining > 0.0:
            # One line a second, not one a tick.
            whole = int(remaining) + 1

            if whole != self.counted_down:
                self.counted_down = whole

                # Published again every second, not once at the start.
                # A latched marker still misses an RViz that was already
                # running when the display was added, or one whose QoS
                # does not match, and a target nobody can see is worse
                # than a few extra messages.
                self.publish_marker(self.previewed_goal)

                self.context.logger.info(
                    f"{self.describe()}: moving in {whole} s"
                )

            return Status.RUNNING

        return super()._tick()

    def publish_target_frame(self, goal) -> None:
        """Broadcast the target as a frame under the planning frame.

        Static rather than repeated: it is one pose that does not move,
        and it stays on the tree after the leaf is done, which is what
        makes it comparable with where the arm actually ended up.
        """
        from geometry_msgs.msg import TransformStamped

        name = self.port("target_frame", self.DEFAULT_TARGET_FRAME)

        transform = TransformStamped()
        transform.header.stamp = (
            self.context.node.get_clock().now().to_msg()
        )
        transform.header.frame_id = goal.pose.header.frame_id
        transform.child_frame_id = name

        position = goal.pose.pose.position
        transform.transform.translation.x = position.x
        transform.transform.translation.y = position.y
        transform.transform.translation.z = position.z

        # The orientation the tool will hold there, so the frame shows
        # the pose and not just the point.
        transform.transform.rotation = goal.pose.pose.orientation

        self.context.static_tf.sendTransform(transform)

        self.context.logger.info(
            f"{self.describe()}: published frame '{name}' under "
            f"'{goal.pose.header.frame_id}'. It is in the RViz TF "
            "display, and `ros2 run tf2_ros tf2_echo "
            f"{goal.pose.header.frame_id} {name}` prints it."
        )

    def publish_marker(self, goal) -> None:
        """Draw the target, and the ray from the camera that named it."""
        from builtin_interfaces.msg import Duration
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray

        position = goal.pose.pose.position
        frame = goal.pose.header.frame_id
        markers = MarkerArray()

        def base(marker_id, marker_type):
            marker = Marker()
            marker.header.frame_id = frame
            marker.header.stamp = self.context.node.get_clock().now().to_msg()
            marker.ns = "move_to_point"
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            # Outlives the leaf on purpose: the last target stays visible
            # for comparison with where the arm actually went.
            marker.lifetime = Duration()
            return marker

        target = base(0, Marker.SPHERE)
        target.pose.position = position
        target.scale.x = target.scale.y = target.scale.z = 0.03
        target.color = ColorRGBA(r=1.0, g=0.3, b=0.0, a=0.9)

        label = base(1, Marker.TEXT_VIEW_FACING)
        label.pose.position.x = position.x
        label.pose.position.y = position.y
        label.pose.position.z = position.z + 0.06
        label.scale.z = 0.03
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = (
            f"{position.x:.3f}, {position.y:.3f}, {position.z:.3f}"
        )

        markers.markers.append(target)
        markers.markers.append(label)

        # The ray from wherever the point was given to the point itself.
        # A target in the right direction but at the wrong distance looks
        # quite different from one in the wrong direction, and the line
        # is what tells them apart.
        origin = self.source_origin(frame)

        if origin is not None:
            ray = base(2, Marker.LINE_STRIP)
            ray.scale.x = 0.004
            ray.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.6)
            ray.points.append(origin)
            ray.points.append(
                Point(x=position.x, y=position.y, z=position.z)
            )
            markers.markers.append(ray)

        self.context.marker_pub.publish(markers)

    def source_origin(self, planning_frame):
        """The origin of the frame the point was given in, or None."""
        import rclpy
        from geometry_msgs.msg import Point

        source_frame = self.port("frame", self.DEFAULT_FRAME)

        if source_frame == planning_frame:
            return None

        try:
            transform = self.context.tf_buffer.lookup_transform(
                planning_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except Exception:
            return None

        translation = transform.transform.translation

        return Point(
            x=translation.x, y=translation.y, z=translation.z
        )

    def build_goal(self):
        """The goal to send, which is the one the preview drew.

        The base class asks for this when it sends; the preview asked
        for it a moment earlier. Handing the same one back is what makes
        the marker a promise rather than an illustration -- TF cannot
        have moved underneath in between.
        """
        if self.previewed_goal is not None:
            goal, self.previewed_goal = self.previewed_goal, None
            return goal

        goal = super().build_goal()

        # A target would send the planner to a scene object instead, which is
        # what MoveTo is for.
        goal.target = ""

        for axis in ("x", "y", "z"):
            if axis not in self.attributes:
                self.context.logger.error(
                    f"{self.describe()}: MoveToPoint needs x, y and z"
                )
                return None

        source_frame = self.port("frame", self.DEFAULT_FRAME)
        planning_frame = self.port("planning_frame", "world")

        point = self._in_planning_frame(
            source_frame,
            planning_frame,
            float(self.port("x", 0.0, cast=float)),
            float(self.port("y", 0.0, cast=float)),
            float(self.port("z", 0.0, cast=float)),
        )

        if point is None:
            return None

        # A camera names the surface of a thing, and the tool has to stop
        # above it: tool0 is the commanded frame, the gripper hangs below
        # it, and a goal placed on the surface drives the gripper through
        # the surface. offset_z is that clearance, in the planning frame,
        # so it is a height above the thing and not a distance along the
        # lens.
        offset_z = float(self.port("offset_z", 0.0, cast=float))

        goal.pose.header.frame_id = planning_frame
        goal.pose.pose.position.x = point.x
        goal.pose.pose.position.y = point.y
        goal.pose.pose.position.z = point.z + offset_z
        # Straight down, which is the only orientation this cell uses.
        goal.pose.pose.orientation.x = 1.0
        goal.pose.pose.orientation.w = 0.0

        if offset_z:
            self.context.logger.warn(
                f"{self.describe()}: the point is at z = {point.z:.3f}, "
                f"the goal {offset_z:+.3f} above it at "
                f"{goal.pose.pose.position.z:.3f}"
            )

        return goal

    def _reset(self) -> None:
        super()._reset()
        self.previewed_goal = None
        self.preview_ends = None
        self.counted_down = None

    def _in_planning_frame(self, source_frame, planning_frame, x, y, z):
        """Carry one point across into the planning frame, or None."""
        import rclpy
        from geometry_msgs.msg import PointStamped
        from tf2_geometry_msgs import do_transform_point

        stamped = PointStamped()
        stamped.header.frame_id = source_frame
        stamped.point.x = x
        stamped.point.y = y
        stamped.point.z = z

        if source_frame == planning_frame:
            return stamped.point

        try:
            # The latest transform rather than one at a stamp: these frames
            # are static, and asking for "now" only races the clock.
            transform = self.context.tf_buffer.lookup_transform(
                planning_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
        except Exception as error:
            self.context.logger.error(
                f"{self.describe()}: no transform from '{source_frame}' to "
                f"'{planning_frame}': {error}"
            )
            return None

        carried = do_transform_point(stamped, transform)

        # Loud rather than informational: this is the number the whole
        # thing turns on, and it is the one worth reading back against
        # the scene before anything moves.
        self.context.logger.warn(
            f"{self.describe()}: ({x:.3f}, {y:.3f}, {z:.3f}) in "
            f"'{source_frame}'  ->  ({carried.point.x:.3f}, "
            f"{carried.point.y:.3f}, {carried.point.z:.3f}) in "
            f"'{planning_frame}'"
        )

        return carried.point


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
    "MoveToPoint": MoveToPoint,
    "Pick": Pick,
    "Place": Place,
    "OpenGripper": OpenGripper,
    "CloseGripper": CloseGripper,
    "DetectObjects": DetectObjects,
    "MatchingTray": MatchingTray,
    "Log": Log,
}

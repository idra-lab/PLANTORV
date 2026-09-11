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

"""Puts the scene into Gazebo and into MoveIt, and keeps the two agreeing.

Three jobs that are really one, which is why they are one node:

* spawning the cubes described by ``scene.yaml`` into a running Gazebo,
* mirroring the trays and the cubes into the MoveIt planning scene, so the
  planner avoids the things the simulator can actually hit,
* the fake grasp: ``plantorv_interfaces/srv/AttachObject``.

The fake grasp is the reason there is no gripper. Picking deletes the cube's
Gazebo model and adds it to the planning scene as an attached collision object
of the tool link; the arm then carries it, and MoveIt keeps planning around it.
Releasing does the reverse, respawning the model at wherever the tool has
carried it to, from where Gazebo drops it into the tray. Nothing is simulated
about the contact, which is the point: the pick-and-place logic can be got
right before any gripper dynamics are.
"""

import threading
import time
from typing import Dict, Optional, Tuple

import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from plantorv_interfaces.srv import AttachObject
from plantorv_sim.scene import TYPE_CUBE, TYPE_TRAY, Scene, quaternion_from_yaw
from plantorv_sim.transforms import compose, make_pose, relative_to, transform_to_pose

# How long to wait for Gazebo and MoveIt to come up before giving up on a call.
SERVICE_TIMEOUT = 10.0


class SceneManager(Node):
    """Owner of everything in the workcell that is not the robot."""

    def __init__(self):
        super().__init__("scene_manager")

        self.declare_parameter("scene_package", "plantorv_sim")
        self.declare_parameter("planning_frame", "world")
        self.declare_parameter("tool_link", "tool0")
        # Links allowed to touch a carried object without it counting as a
        # collision. Without these every grasp is in collision from the start.
        self.declare_parameter(
            "touch_links", ["tool0", "flange", "wrist_3_link", "wrist_2_link"]
        )
        # Skip the Gazebo half and only drive the planning scene. This is what
        # makes the node usable against mock hardware, with no simulator.
        self.declare_parameter("use_gazebo", True)

        self.planning_frame = self.get_parameter("planning_frame").value
        self.tool_link = self.get_parameter("tool_link").value
        self.touch_links = list(self.get_parameter("touch_links").value)
        self.use_gazebo = bool(self.get_parameter("use_gazebo").value)

        self.scene = Scene.from_package(self.get_parameter("scene_package").value)
        # object id -> (link it hangs from, its pose in that link's frame),
        # empty while nothing is held. Keeping the grasp pose is what lets the
        # object be released in the orientation it was picked up in.
        self.attached: Dict[str, Tuple[str, Pose]] = {}
        self.lock = threading.Lock()

        clients = ReentrantCallbackGroup()
        services = MutuallyExclusiveCallbackGroup()

        self.spawn_client = self.create_client(
            SpawnEntity, "/spawn_entity", callback_group=clients
        )
        self.delete_client = self.create_client(
            DeleteEntity, "/delete_entity", callback_group=clients
        )
        self.planning_scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene", callback_group=clients
        )

        self.model_poses: Dict[str, Pose] = {}
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._on_model_states, 1,
            callback_group=clients,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.attach_service = self.create_service(
            AttachObject, "attach_object", self._on_attach, callback_group=services
        )

        # Setting the scene up talks to services, so it cannot run in the
        # constructor -- the executor is not spinning yet. A one-shot timer
        # runs it on the first tick instead.
        self.setup_timer = self.create_timer(1.0, self._setup, callback_group=clients)

    # -- start-up --------------------------------------------------------

    def _setup(self) -> None:
        """Spawn the cubes and publish the static part of the planning scene."""
        self.setup_timer.cancel()

        if self.use_gazebo:
            if not self.spawn_client.wait_for_service(timeout_sec=60.0):
                self.get_logger().error("/spawn_entity never appeared; is Gazebo running?")
                return

        if not self.planning_scene_client.wait_for_service(timeout_sec=60.0):
            self.get_logger().error("/apply_planning_scene never appeared; is move_group running?")
            return

        try:
            if self.use_gazebo:
                for cube in self.scene.of_type(TYPE_CUBE):
                    self._spawn(
                        cube.name, self.scene.model_sdf(cube), self._nominal_pose(cube.name)
                    )
            objects = [
                self._collision_object(obj)
                for obj in list(self.scene.of_type(TYPE_TRAY))
                + list(self.scene.of_type(TYPE_CUBE))
            ]
            self._apply(world_objects=objects)
        except RuntimeError as error:
            # A timer callback that raises takes the executor down with it, and
            # the scene not being set up is a thing to report, not to die of.
            self.get_logger().error(f"could not set the scene up: {error}")
            return
        self.get_logger().info(
            f"scene ready: {len(self.scene.of_type(TYPE_CUBE))} cubes, "
            f"{len(self.scene.of_type(TYPE_TRAY))} trays"
        )

        # From here on the cubes move, and the planning scene has to follow.
        self.create_timer(1.0, self._sync_cubes)

    # -- keeping MoveIt in step with Gazebo ------------------------------

    def _on_model_states(self, msg: ModelStates) -> None:
        self.model_poses = dict(zip(msg.name, msg.pose))

    def _sync_cubes(self) -> None:
        """Move the planning scene's cubes to where Gazebo has them.

        Only the ones that are lying about: a carried cube is attached to the
        tool, so MoveIt already moves it, and republishing it as a world object
        would give the arm something to collide with in its own hand.
        """
        if not self.use_gazebo:
            return
        updates = []
        with self.lock:
            held = set(self.attached)
        for cube in self.scene.of_type(TYPE_CUBE):
            if cube.name in held:
                continue
            pose = self.model_poses.get(cube.name)
            if pose is None:
                continue
            updates.append(self._collision_object(cube, pose=pose))
        if not updates:
            return
        try:
            self._apply(world_objects=updates)
        except RuntimeError as error:
            self.get_logger().warn(f"could not update the planning scene: {error}", once=True)

    # -- the fake grasp --------------------------------------------------

    def _on_attach(self, request, response):
        object_id = request.object_id
        link = request.link_name or self.tool_link
        obj = self.scene.get(object_id)

        if obj is None:
            response.success = False
            response.message = f"unknown scene object '{object_id}'"
            return response

        try:
            with self.lock:
                if request.attach:
                    self._attach(object_id, link)
                else:
                    self._detach(object_id, link)
        except RuntimeError as error:
            response.success = False
            response.message = str(error)
            self.get_logger().error(f"{'attach' if request.attach else 'detach'}: {error}")
            return response

        response.success = True
        response.message = f"{'attached' if request.attach else 'released'} {object_id}"
        self.get_logger().info(response.message)
        return response

    def _attach(self, object_id: str, link: str) -> None:
        if object_id in self.attached:
            raise RuntimeError(f"{object_id} is already held")

        obj = self.scene.get(object_id)
        world_pose = self._current_pose(object_id)
        tool_pose = self._link_pose(link)
        pose_in_tool = relative_to(tool_pose, world_pose)

        if self.use_gazebo:
            self._delete(object_id)

        # Take it out of the world before putting it in the hand, so the two
        # copies never exist at once and the arm is not planning around a cube
        # it is holding.
        self._apply(world_objects=[self._removal(object_id)])
        attached = AttachedCollisionObject()
        attached.link_name = link
        attached.touch_links = self.touch_links
        attached.object = self._collision_object(obj, pose=pose_in_tool, frame_id=link)
        self._apply(attached_objects=[attached])
        self.attached[object_id] = (link, pose_in_tool)

    def _detach(self, object_id: str, link: str) -> None:
        if object_id not in self.attached:
            raise RuntimeError(f"{object_id} is not held")

        obj = self.scene.get(object_id)
        held_link, pose_in_tool = self.attached[object_id]
        # Where the tool has carried it to.
        world_pose = compose(self._link_pose(held_link), pose_in_tool)

        detached = AttachedCollisionObject()
        detached.link_name = held_link
        detached.object.id = object_id
        detached.object.operation = CollisionObject.REMOVE
        self._apply(attached_objects=[detached])
        del self.attached[object_id]

        if self.use_gazebo:
            self._spawn(object_id, self.scene.model_sdf(obj), world_pose)
        self._apply(world_objects=[self._collision_object(obj, pose=world_pose)])

    # -- poses -----------------------------------------------------------

    def _nominal_pose(self, name: str) -> Pose:
        """The pose ``scene.yaml`` puts an object at."""
        obj = self.scene.get(name)
        return make_pose(obj.position, quaternion_from_yaw(obj.yaw))

    def _current_pose(self, name: str) -> Pose:
        """Where the object is now, from Gazebo, falling back to the yaml."""
        pose = self.model_poses.get(name)
        return pose if pose is not None else self._nominal_pose(name)

    def _link_pose(self, link: str) -> Pose:
        """The pose of a robot link in the planning frame, from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame, link, rclpy.time.Time()
            )
        except Exception as error:  # tf2 raises several unrelated types
            raise RuntimeError(f"no transform {self.planning_frame} -> {link}: {error}")
        return transform_to_pose(transform.transform)

    # -- message building ------------------------------------------------

    def _collision_object(self, obj, pose: Optional[Pose] = None, frame_id: str = "") -> CollisionObject:
        """One scene object as a MoveIt collision object.

        Composite objects -- the trays -- become several box primitives under a
        single id, so that attaching, moving or removing a tray is one
        operation.
        """
        collision = CollisionObject()
        collision.id = obj.name
        collision.header.frame_id = frame_id or self.planning_frame
        collision.pose = pose if pose is not None else self._nominal_pose(obj.name)
        collision.operation = CollisionObject.ADD

        for centre, size in obj.boxes():
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [float(size[0]), float(size[1]), float(size[2])]
            collision.primitives.append(box)
            # Relative to the object's own pose, which is what `pose` above is.
            collision.primitive_poses.append(
                make_pose(
                    (
                        centre[0] - obj.position[0],
                        centre[1] - obj.position[1],
                        centre[2] - obj.position[2],
                    ),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        return collision

    def _removal(self, object_id: str) -> CollisionObject:
        collision = CollisionObject()
        collision.id = object_id
        collision.header.frame_id = self.planning_frame
        collision.operation = CollisionObject.REMOVE
        return collision

    # -- service plumbing ------------------------------------------------

    def _apply(self, world_objects=None, attached_objects=None) -> None:
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        if world_objects:
            scene.world.collision_objects = list(world_objects)
        if attached_objects:
            scene.robot_state.attached_collision_objects = list(attached_objects)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        result = self._call(self.planning_scene_client, request)
        if result is None or not result.success:
            raise RuntimeError("move_group refused the planning scene update")

    def _spawn(self, name: str, sdf: str, pose: Pose) -> None:
        request = SpawnEntity.Request()
        request.name = name
        request.xml = sdf
        request.initial_pose = pose
        request.reference_frame = "world"
        result = self._call(self.spawn_client, request)
        if result is None or not result.success:
            reason = "no response" if result is None else result.status_message
            raise RuntimeError(f"could not spawn {name}: {reason}")

    def _delete(self, name: str) -> None:
        request = DeleteEntity.Request()
        request.name = name
        result = self._call(self.delete_client, request)
        if result is None or not result.success:
            reason = "no response" if result is None else result.status_message
            raise RuntimeError(f"could not delete {name}: {reason}")

    def _call(self, client, request, timeout: float = SERVICE_TIMEOUT):
        """Call a service and wait, without blocking the executor.

        The clients are in a reentrant callback group and the executor is
        multi-threaded, so waiting here still lets the response be delivered.
        """
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None


def main(args=None):
    rclpy.init(args=args)
    node = SceneManager()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
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

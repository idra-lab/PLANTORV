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

"""Answers "where is ``cube_2``?" -- for now, by asking Gazebo.

This is the seam the perception pipeline replaces. The planner and the
behaviour tree never look up an object themselves; they name it and ask, so
that swapping this node for one backed by segmentation and depth estimation
changes nothing above it.

Until then the answer is ground truth: the poses Gazebo publishes on
``/gazebo/model_states``, with the dimensions and colours out of
``scene.yaml``. An object that vanishes from Gazebo -- which is what the fake
grasp does to a cube it picks up -- keeps its last known pose, flagged by
``found`` staying true, since it is still a thing in the world, just one being
carried.
"""

from typing import Dict

import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from rclpy.node import Node

from plantorv_interfaces.srv import GetObjectPose, ListObjects
from plantorv_sim.scene import Scene, quaternion_from_yaw
from plantorv_sim.transforms import make_pose


class WorldModel(Node):
    """Ground-truth object poses, behind the interface perception will use."""

    def __init__(self):
        super().__init__("world_model")

        self.declare_parameter("scene_package", "plantorv_sim")
        self.declare_parameter("planning_frame", "world")

        self.planning_frame = self.get_parameter("planning_frame").value
        self.scene = Scene.from_package(self.get_parameter("scene_package").value)
        self.poses: Dict[str, Pose] = {}

        self.create_subscription(ModelStates, "/gazebo/model_states", self._on_model_states, 1)
        self.create_service(GetObjectPose, "get_object_pose", self._on_get_object_pose)
        self.create_service(ListObjects, "list_objects", self._on_list_objects)

        self.get_logger().info(f"world model up, {len(self.scene)} objects known")

    def _on_model_states(self, msg: ModelStates) -> None:
        # Names not in the scene -- the ground plane, the robot -- are ignored;
        # the poses of the ones that are get remembered even after the model
        # goes away.
        for name, pose in zip(msg.name, msg.pose):
            if self.scene.get(name) is not None:
                self.poses[name] = pose

    def _pose_of(self, name: str) -> Pose:
        """Last seen pose, or the one the scene file declares."""
        known = self.poses.get(name)
        if known is not None:
            return known
        obj = self.scene.get(name)
        return make_pose(obj.position, quaternion_from_yaw(obj.yaw))

    def _on_get_object_pose(self, request, response):
        obj = self.scene.get(request.name)
        if obj is None:
            response.found = False
            return response

        response.found = True
        response.pose.header.frame_id = self.planning_frame
        response.pose.header.stamp = self.get_clock().now().to_msg()
        response.pose.pose = self._pose_of(obj.name)
        response.dimensions.x, response.dimensions.y, response.dimensions.z = obj.dimensions
        response.type = obj.type
        response.color = _color_name(obj.color)
        return response

    def _on_list_objects(self, request, response):
        for obj in self.scene:
            if request.type_filter and obj.type != request.type_filter:
                continue
            response.names.append(obj.name)
            response.types.append(obj.type)
            response.colors.append(_color_name(obj.color))
            response.poses.append(self._pose_of(obj.name))
        return response


def _color_name(rgba) -> str:
    """A colour word for an RGBA tuple.

    The behaviour tree says "put it in the red tray", not "put it in the tray
    whose diffuse is 0.8 0.1 0.1", so the colour has to survive as a word.
    """
    r, g, b = rgba[0], rgba[1], rgba[2]
    names = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "grey": (0.5, 0.5, 0.5),
        "brown": (0.65, 0.5, 0.39),
    }
    return min(
        names,
        key=lambda name: sum((c - v) ** 2 for c, v in zip((r, g, b), names[name])),
    )


def main(args=None):
    rclpy.init(args=args)
    node = WorldModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

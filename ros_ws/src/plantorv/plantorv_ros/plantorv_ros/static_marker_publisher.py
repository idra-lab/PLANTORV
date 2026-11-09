#!/usr/bin/env python3

"""Replay the recorded marker frames as static transforms.

Reads the file ``save_marker_transforms`` wrote and publishes every
transform in it once, on the latched static TF topic. The markers do
not have to be visible, or even present, for the frames to exist, so
this is what a bringup runs instead of the detector nodes.

The frames are only as true as the setup they were recorded in. Moving
the camera or the robot invalidates the file, and nothing here can
detect that: record it again.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from plantorv_ros import marker_transform_file


class StaticMarkerPublisher(Node):
    def __init__(self):
        super().__init__("static_marker_publisher")

        self.declare_parameter(
            "input_file", marker_transform_file.default_file()
        )

        # Publish only these entries. Empty means all of them. An
        # entry matches by its key in the file or by the frame it
        # publishes, which are not always the same name.
        self.declare_parameter("frames", [""])

        self.input_file = marker_transform_file.resolve(
            self.get_parameter("input_file").value
        )

        wanted = [
            str(value)
            for value in self.get_parameter("frames").value
            if str(value)
        ]

        document = marker_transform_file.read(self.input_file)
        entries = document["transforms"]

        transforms = []

        for key, entry in entries.items():
            transform = self.build(key, entry)

            if wanted and not {
                key, transform.child_frame_id
            } & set(wanted):
                continue

            transforms.append(transform)

        if not transforms:
            available = ", ".join(
                sorted(
                    {key for key in entries}
                    | {
                        str((entry or {}).get("child_frame", key))
                        for key, entry in entries.items()
                    }
                )
            )

            raise ValueError(
                f"{self.input_file} has none of the requested frames: "
                f"{', '.join(wanted)}. It offers: {available}"
            )

        # A static broadcaster latches, so one publish reaches every
        # later subscriber and the node just has to stay alive.
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(transforms)

        recorded_at = document.get("recorded_at", "an unknown time")

        self.get_logger().info(
            f"Published {len(transforms)} static transform(s) from "
            f"{self.input_file}, recorded at {recorded_at}: "
            + ", ".join(
                f"{transform.header.frame_id} -> "
                f"{transform.child_frame_id}"
                for transform in transforms
            )
        )

    def build(self, child_frame: str, entry: dict) -> TransformStamped:
        """Turn one file entry into a transform message."""
        try:
            parent_frame = entry["parent_frame"]
            translation = entry["translation"]
            rotation = entry["rotation"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Entry '{child_frame}' in {self.input_file} is "
                f"missing {error}"
            ) from error

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = str(parent_frame)
        transform.child_frame_id = str(
            entry.get("child_frame", child_frame)
        )

        transform.transform.translation.x = float(translation["x"])
        transform.transform.translation.y = float(translation["y"])
        transform.transform.translation.z = float(translation["z"])

        transform.transform.rotation.x = float(rotation["x"])
        transform.transform.rotation.y = float(rotation["y"])
        transform.transform.rotation.z = float(rotation["z"])
        transform.transform.rotation.w = float(rotation["w"])

        return transform


def main(args=None):
    rclpy.init(args=args)
    node = StaticMarkerPublisher()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    node.destroy_node()

    # A SIGINT from a launch file shuts the context down through
    # rclpy's own signal handler, so shutting it down again raises.
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

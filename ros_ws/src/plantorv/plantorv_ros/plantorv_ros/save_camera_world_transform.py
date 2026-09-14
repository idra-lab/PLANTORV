#!/usr/bin/env python3

"""Save the current static-colour-camera transform into ``world`` once.

This one-shot node waits for TF to know how to carry a point in
``static_camera_color_optical_frame`` into ``world``. It then writes exactly
that ``world <- camera`` transform to YAML for the offline mapping pipeline.
Run it again whenever either the camera or robot moves.
"""

from datetime import datetime, timezone
from typing import Optional, Sequence

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from plantorv_ros import camera_world_transform_file


class SaveCameraWorldTransform(Node):
    """Wait for one TF transform, write it, and exit."""

    def __init__(self) -> None:
        super().__init__("save_camera_world_transform")

        self.declare_parameter("target_frame", "world")
        self.declare_parameter(
            "source_frame", "static_camera_color_optical_frame"
        )
        self.declare_parameter(
            "output_file", camera_world_transform_file.default_file()
        )
        self.declare_parameter("timeout", 30.0)
        self.declare_parameter("lookup_period", 0.1)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.source_frame = str(self.get_parameter("source_frame").value)
        self.output_file = camera_world_transform_file.resolve(
            self.get_parameter("output_file").value
        )
        self.timeout = float(self.get_parameter("timeout").value)
        period = float(self.get_parameter("lookup_period").value)

        if not self.target_frame or not self.source_frame:
            raise ValueError("target_frame and source_frame must not be empty")
        if self.target_frame != "world":
            raise ValueError(
                "target_frame must be 'world': mapping writes object_point_world_m"
            )
        if self.timeout <= 0.0 or period <= 0.0:
            raise ValueError("timeout and lookup_period must be positive")

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.started = self.get_clock().now()
        self.finished = False
        self.succeeded = False
        self.last_error = "TF data has not arrived yet"
        self.timer = self.create_timer(period, self.capture)

        self.get_logger().info(
            f"Waiting up to {self.timeout:.1f} s for TF transform "
            f"'{self.source_frame}' -> '{self.target_frame}', then writing "
            f"{self.output_file}"
        )

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.started).nanoseconds / 1e9

    def capture(self) -> None:
        """Try the latest transform; finish after one successful lookup."""
        if self.finished:
            return

        try:
            transform = self.buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception as error:
            self.last_error = str(error)
            if self.elapsed() >= self.timeout:
                self.get_logger().error(
                    f"No transform from '{self.source_frame}' to "
                    f"'{self.target_frame}' after {self.timeout:.1f} s: "
                    f"{self.last_error}\nKnown TF frames:\n"
                    + self.buffer.all_frames_as_string()
                )
                self.finish(False)
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        document = {
            "target_frame": self.target_frame,
            "source_frame": self.source_frame,
            "recorded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "translation": {
                "x": float(translation.x),
                "y": float(translation.y),
                "z": float(translation.z),
            },
            "rotation": {
                "x": float(rotation.x),
                "y": float(rotation.y),
                "z": float(rotation.z),
                "w": float(rotation.w),
            },
        }
        written = camera_world_transform_file.write(self.output_file, document)
        self.get_logger().info(
            f"Saved '{self.source_frame}' -> '{self.target_frame}' to "
            f"{written}: xyz ({translation.x:+.4f}, {translation.y:+.4f}, "
            f"{translation.z:+.4f}) m"
        )
        self.finish(True)

    def finish(self, succeeded: bool) -> None:
        self.finished = True
        self.succeeded = succeeded
        self.timer.cancel()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = SaveCameraWorldTransform()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    succeeded = node.succeeded
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

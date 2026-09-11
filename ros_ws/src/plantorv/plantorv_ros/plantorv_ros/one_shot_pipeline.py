#!/usr/bin/env python3

import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


def stamp_ns(msg) -> int:
    """Return a ROS message header timestamp in nanoseconds."""
    return (
        msg.header.stamp.sec * 1_000_000_000
        + msg.header.stamp.nanosec
    )


class OneShotPipelineNode(Node):
    def __init__(self):
        super().__init__("one_shot_pipeline")

        # Topics
        self.declare_parameter(
            "rgb_topic",
            "/camera/color/image_raw",
        )
        self.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "pointcloud_topic",
            "/camera/depth/color/points",
        )

        # Synchronization
        self.declare_parameter("sync_slop", 0.02)

        # If the cloud stream never crosses the RGB-D timestamp,
        # use the closest cloud seen after this many seconds.
        self.declare_parameter("pointcloud_timeout", 2.0)

        # Pipeline
        self.declare_parameter(
            "pipeline_script",
            "/path/to/project/samgpt.py",
        )
        self.declare_parameter(
            "output_dir",
            "/tmp/samgpt_ros_run",
        )

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.pointcloud_topic = self.get_parameter(
            "pointcloud_topic"
        ).value

        self.pipeline_script = Path(
            self.get_parameter("pipeline_script").value
        )

        self.output_dir = Path(
            self.get_parameter("output_dir").value
        )

        sync_slop = float(
            self.get_parameter("sync_slop").value
        )

        self.pointcloud_timeout = float(
            self.get_parameter("pointcloud_timeout").value
        )

        # Input directories given to the existing pipeline.
        self.rgb_dir = self.output_dir / "input" / "rgb"
        self.depth_dir = self.output_dir / "input" / "depth"
        self.pointcloud_dir = (
            self.output_dir / "input" / "pointcloud"
        )

        # Human-viewable copy of the depth frame, kept out of
        # depth_dir so it doesn't interfere with the pipeline's
        # one-file-per-frame indexing.
        self.depth_preview_dir = (
            self.output_dir / "input" / "depth_preview"
        )

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.pointcloud_dir.mkdir(parents=True, exist_ok=True)
        self.depth_preview_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()

        # State
        self.capture_started = False
        self.finished = False

        # Timestamp representing the RGB-D pair.
        self.target_stamp_ns = None

        # Wall-clock time at which the RGB-D pair was captured.
        self.capture_start_time = None

        # Keep several recent point clouds around so clouds arriving
        # slightly before the RGB-D callback are available.
        self.cloud_buffer = deque(maxlen=10)

        # ----------------------------------------------------------
        # RGB + depth synchronization
        # ----------------------------------------------------------

        self.rgb_sub = Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=sync_slop,
        )

        self.sync.registerCallback(self.rgbd_callback)

        # ----------------------------------------------------------
        # Independent point-cloud subscription
        # ----------------------------------------------------------

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        # Used only as a fallback if no cloud crosses target time.
        self.timeout_timer = self.create_timer(
            0.05,
            self.check_pointcloud_timeout,
        )

        self.get_logger().info(
            "Waiting for one capture:\n"
            f"  RGB:        {self.rgb_topic}\n"
            f"  Depth:      {self.depth_topic}\n"
            f"  PointCloud: {self.pointcloud_topic}"
        )

    # ------------------------------------------------------------------
    # RGB-D
    # ------------------------------------------------------------------

    def rgbd_callback(
        self,
        rgb_msg: Image,
        depth_msg: Image,
    ) -> None:
        if self.capture_started:
            return

        self.capture_started = True
        self.capture_start_time = time.monotonic()

        rgb_ns = stamp_ns(rgb_msg)
        depth_ns = stamp_ns(depth_msg)

        # Since RGB and depth are approximately synchronized, use
        # their midpoint as the timestamp representing the pair.
        self.target_stamp_ns = (rgb_ns + depth_ns) // 2

        self.get_logger().info(
            "Captured RGB-D pair:\n"
            f"  RGB   = {rgb_ns / 1e9:.9f}\n"
            f"  Depth = {depth_ns / 1e9:.9f}\n"
            f"  delta = {abs(rgb_ns - depth_ns) / 1e6:.3f} ms"
        )

        try:
            self.save_rgbd(rgb_msg, depth_msg)
        except Exception as exc:
            self.get_logger().error(
                f"Failed to save RGB-D pair: {exc}"
            )
            self.finish_without_pipeline()
            return

        # A suitable pointcloud may already be in the buffer.
        self.try_select_pointcloud()

    def save_rgbd(
        self,
        rgb_msg: Image,
        depth_msg: Image,
    ) -> None:
        rgb = self.bridge.imgmsg_to_cv2(
            rgb_msg,
            desired_encoding="bgr8",
        )

        depth = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )

        rgb_path = self.rgb_dir / "img_0.png"
        depth_path = self.depth_dir / "img_0.png"

        if not cv2.imwrite(str(rgb_path), rgb):
            raise RuntimeError(
                f"Could not write {rgb_path}"
            )

        # Typical ROS depth representations:
        #
        #   16UC1 -> uint16, usually millimetres
        #   32FC1 -> float32, usually metres
        #
        # Store sensor depth as uint16 millimetres.
        if np.issubdtype(depth.dtype, np.floating):
            depth = np.nan_to_num(
                depth,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            depth = np.clip(
                depth * 1000.0,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)

        elif depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)

        if not cv2.imwrite(str(depth_path), depth):
            raise RuntimeError(
                f"Could not write {depth_path}"
            )

        # Raw depth values only span a small fraction of the
        # uint16 range, so the file above looks solid black in a
        # normal viewer. Save a rescaled, colorized copy too.
        preview_path = self.depth_preview_dir / "img_0.png"

        if not cv2.imwrite(
            str(preview_path),
            self.colorize_depth(depth),
        ):
            raise RuntimeError(
                f"Could not write {preview_path}"
            )

        self.get_logger().info(
            f"Saved RGB:           {rgb_path}\n"
            f"Saved depth:         {depth_path}\n"
            f"Saved depth preview: {preview_path}"
        )

    @staticmethod
    def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
        """Rescale a raw millimetre depth map into a viewable BGR image."""
        depth_f = depth_mm.astype(np.float32)
        valid = depth_f > 0

        if not np.any(valid):
            return np.zeros(
                (*depth_f.shape, 3),
                dtype=np.uint8,
            )

        near, far = np.percentile(depth_f[valid], (2.0, 98.0))

        if far <= near:
            far = near + 1.0

        scaled = np.clip(
            (depth_f - near) * (255.0 / (far - near)),
            0,
            255,
        ).astype(np.uint8)

        preview = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        preview[~valid] = 0

        return preview

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------

    def pointcloud_callback(
        self,
        cloud_msg: PointCloud2,
    ) -> None:
        if self.finished:
            return

        self.cloud_buffer.append(cloud_msg)

        if not self.capture_started:
            return

        self.try_select_pointcloud()

    def try_select_pointcloud(self) -> None:
        """Select a cloud once its stream has crossed target time."""
        if (
            self.finished
            or self.target_stamp_ns is None
            or not self.cloud_buffer
        ):
            return

        newest_stamp = stamp_ns(self.cloud_buffer[-1])

        # Once we have received a cloud at or after the target,
        # any future cloud should be even later. Assuming timestamps
        # are monotonically increasing, the closest cloud must now
        # be somewhere in this buffer.
        if newest_stamp < self.target_stamp_ns:
            return

        cloud = min(
            self.cloud_buffer,
            key=lambda msg: abs(
                stamp_ns(msg) - self.target_stamp_ns
            ),
        )

        self.process_capture(cloud)

    def check_pointcloud_timeout(self) -> None:
        if (
            self.finished
            or not self.capture_started
            or self.capture_start_time is None
        ):
            return

        elapsed = (
            time.monotonic()
            - self.capture_start_time
        )

        if elapsed < self.pointcloud_timeout:
            return

        if self.cloud_buffer:
            cloud = min(
                self.cloud_buffer,
                key=lambda msg: abs(
                    stamp_ns(msg) - self.target_stamp_ns
                ),
            )

            self.get_logger().warning(
                "Point-cloud timeout reached. "
                "Using closest buffered cloud."
            )

            self.process_capture(cloud)

        else:
            self.get_logger().error(
                "Point-cloud timeout reached, but no "
                "PointCloud2 messages were received."
            )

            # The RGB-D pipeline can still run.
            self.process_capture(None)

    # ------------------------------------------------------------------
    # Final capture
    # ------------------------------------------------------------------

    def process_capture(
        self,
        cloud_msg: PointCloud2 | None,
    ) -> None:
        if self.finished:
            return

        # Important: set immediately, because running the pipeline
        # below is blocking.
        self.finished = True

        if cloud_msg is not None:
            cloud_ns = stamp_ns(cloud_msg)

            delta_ms = abs(
                cloud_ns - self.target_stamp_ns
            ) / 1e6

            self.get_logger().info(
                "Selected point cloud:\n"
                f"  stamp = {cloud_ns / 1e9:.9f}\n"
                f"  delta = {delta_ms:.3f} ms"
            )

            ply_path = (
                self.pointcloud_dir
                / "pointcloud0.ply"
            )

            try:
                self.save_pointcloud_ply(
                    cloud_msg,
                    ply_path,
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Could not save point cloud: {exc}"
                )

        try:
            self.run_pipeline()

            self.get_logger().info(
                f"Pipeline complete: {self.output_dir}"
            )

        except Exception as exc:
            self.get_logger().error(
                f"Pipeline failed: {exc}"
            )

        finally:
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # PointCloud2 -> PLY
    # ------------------------------------------------------------------

    def save_pointcloud_ply(
        self,
        cloud: PointCloud2,
        path: Path,
    ) -> None:
        """Write PointCloud2 as a binary little-endian PLY.

        x/y/z are always saved.

        If the PointCloud2 contains a conventional packed `rgb`
        or `rgba` field, RGB colors are saved as well.
        """
        available_fields = {
            field.name for field in cloud.fields
        }

        required = {"x", "y", "z"}

        if not required.issubset(available_fields):
            raise ValueError(
                "PointCloud2 must contain x, y and z fields. "
                f"Available fields: {sorted(available_fields)}"
            )

        color_field = None

        if "rgb" in available_fields:
            color_field = "rgb"
        elif "rgba" in available_fields:
            color_field = "rgba"

        fields = ["x", "y", "z"]

        if color_field is not None:
            fields.append(color_field)

        # ROS 2's sensor_msgs_py returns a structured NumPy array.
        points = point_cloud2.read_points(
            cloud,
            field_names=fields,
            skip_nans=False,
        )

        # Drop invalid XYZ points.
        valid = (
            np.isfinite(points["x"])
            & np.isfinite(points["y"])
            & np.isfinite(points["z"])
        )

        points = points[valid]

        if len(points) == 0:
            raise ValueError(
                "Point cloud contains no finite XYZ points."
            )

        has_color = color_field is not None

        if has_color:
            packed = points[color_field]

            # Most ROS/PCL clouds store packed RGB as FLOAT32.
            # Others expose it directly as an integer.
            if packed.dtype.kind == "f":
                if packed.dtype.itemsize != 4:
                    raise ValueError(
                        f"Unsupported {color_field} type: "
                        f"{packed.dtype}"
                    )

                packed = (
                    packed.astype(np.float32, copy=False)
                    .view(np.uint32)
                )

            else:
                packed = packed.astype(
                    np.uint32,
                    copy=False,
                )

            red = (
                (packed >> 16) & 0xFF
            ).astype(np.uint8)

            green = (
                (packed >> 8) & 0xFF
            ).astype(np.uint8)

            blue = (
                packed & 0xFF
            ).astype(np.uint8)

            ply_dtype = np.dtype([
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ])

        else:
            ply_dtype = np.dtype([
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
            ])

        ply_points = np.empty(
            len(points),
            dtype=ply_dtype,
        )

        ply_points["x"] = points["x"]
        ply_points["y"] = points["y"]
        ply_points["z"] = points["z"]

        if has_color:
            ply_points["red"] = red
            ply_points["green"] = green
            ply_points["blue"] = blue

        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(ply_points)}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if has_color:
            header += [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]

        header.append("end_header")

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(path, "wb") as file:
            file.write(
                ("\n".join(header) + "\n").encode("ascii")
            )

            file.write(
                ply_points.tobytes()
            )

        self.get_logger().info(
            f"Saved {len(ply_points)} points "
            f"to {path}"
        )

    # ------------------------------------------------------------------
    # Existing pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self) -> None:
        command = [
            sys.executable,
            str(self.pipeline_script),

            "--input-dir",
            str(self.rgb_dir),

            "--depth-dir",
            str(self.depth_dir),

            "--depth-source",
            "sensor",

            "--output-dir",
            str(self.output_dir),
        ]

        self.get_logger().info(
            "Running segmentation -> annotation "
            "-> depth pipeline..."
        )

        subprocess.run(
            command,
            check=True,
            cwd=self.pipeline_script.parent,
        )

    def finish_without_pipeline(self) -> None:
        self.finished = True

        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = OneShotPipelineNode()

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
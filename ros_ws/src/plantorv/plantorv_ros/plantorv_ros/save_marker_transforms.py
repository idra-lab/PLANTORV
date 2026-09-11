#!/usr/bin/env python3

"""Record the live marker frames once and write them to a YAML file.

``charuco_tf_publisher`` and ``aruco_tf_publisher`` re-solve their
frames on every image, which is what you want while placing the board
and the robot. Once both are where they should stay, this node samples
those frames for a moment, averages each one and writes the result to
disk. ``static_marker_publisher`` then replays the file at every
bringup, so the camera does not have to see the markers again.

The node is one-shot: it exits as soon as it has written the file, or
as soon as it gives up waiting for a frame.
"""

from datetime import datetime, timezone

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from plantorv_ros import marker_transform_file
from plantorv_ros.marker_transform_file import DEFAULT_FILE
from plantorv_ros.pose_overlay import (
    quaternion_angle_degrees,
    quaternion_average,
)


class SaveMarkerTransforms(Node):
    def __init__(self):
        super().__init__("save_marker_transforms")

        # Which frames to record, and the frame they are recorded in.
        self.declare_parameter("parent_frame", "camera_color_optical_frame")
        self.declare_parameter(
            "frames", ["charuco_board", "robot_base_marker"]
        )

        self.declare_parameter("output_file", DEFAULT_FILE)

        # How many samples to average, and how long to wait for them.
        self.declare_parameter("samples", 30)
        self.declare_parameter("timeout", 30.0)
        self.declare_parameter("sample_period", 0.1)

        # Ignore the first samples after start-up. The camera arrives
        # with auto exposure and auto white balance still settling, and
        # those frames detect worse than the ones after them.
        self.declare_parameter("settle_time", 2.0)

        # A frame that moves more than this while sampling was not
        # standing still, so averaging it would hide the problem. The
        # limits are loose enough for a single ArUco marker a metre
        # out, which jitters a few millimetres and a degree or two; a
        # board is steadier than that. They are tight enough to catch
        # the real failures, such as two nodes publishing one frame.
        self.declare_parameter("max_translation_spread", 0.012)
        self.declare_parameter("max_rotation_spread", 3.0)

        # Drop samples this many standard deviations out before
        # averaging. Zero keeps every sample.
        self.declare_parameter("outlier_sigma", 2.5)

        # Write the file even when a frame is that unsteady.
        self.declare_parameter("force", False)

        self.parent_frame = self.get_parameter("parent_frame").value
        self.frames = [
            str(value) for value in self.get_parameter("frames").value
        ]
        self.output_file = marker_transform_file.resolve(
            self.get_parameter("output_file").value
        )

        self.samples = int(self.get_parameter("samples").value)
        self.timeout = float(self.get_parameter("timeout").value)
        sample_period = float(
            self.get_parameter("sample_period").value
        )

        self.settle_time = float(
            self.get_parameter("settle_time").value
        )

        self.max_translation_spread = float(
            self.get_parameter("max_translation_spread").value
        )
        self.max_rotation_spread = float(
            self.get_parameter("max_rotation_spread").value
        )
        self.outlier_sigma = float(
            self.get_parameter("outlier_sigma").value
        )
        self.force = bool(self.get_parameter("force").value)

        if not self.frames:
            raise ValueError("frames must name at least one frame")

        if self.samples < 1:
            raise ValueError(
                f"samples must be at least 1, got {self.samples}"
            )

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Samples per frame, each an (x, y, z, qx, qy, qz, qw) row.
        self.collected = {frame: [] for frame in self.frames}

        self.started = self.get_clock().now()
        self.finished = False
        self.succeeded = False

        self.timer = self.create_timer(sample_period, self.sample)

        self.get_logger().info(
            f"Recording {', '.join(self.frames)} in "
            f"{self.parent_frame}, {self.samples} samples each, into "
            f"{self.output_file}"
        )

    def elapsed(self) -> float:
        """Seconds since the node started."""
        return (
            self.get_clock().now() - self.started
        ).nanoseconds / 1e9

    def sample(self):
        """Take one sample of every frame that still needs them."""
        if self.finished:
            return

        if self.elapsed() < self.settle_time:
            return

        for frame in self.frames:
            if len(self.collected[frame]) >= self.samples:
                continue

            try:
                # The latest available transform, which is what the
                # detector published on its last image.
                transform = self.buffer.lookup_transform(
                    self.parent_frame,
                    frame,
                    rclpy.time.Time(),
                )
            except Exception:
                continue

            translation = transform.transform.translation
            rotation = transform.transform.rotation

            self.collected[frame].append([
                translation.x,
                translation.y,
                translation.z,
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ])

        missing = [
            frame
            for frame in self.frames
            if len(self.collected[frame]) < self.samples
        ]

        if not missing:
            self.finish(self.write())
            return

        if self.elapsed() > self.timeout:
            for frame in missing:
                self.get_logger().error(
                    f"Only got {len(self.collected[frame])} of "
                    f"{self.samples} samples of '{frame}' in "
                    f"{self.timeout:.0f} s. Is it being published, and "
                    f"is its parent '{self.parent_frame}'?"
                )

            self.finish(False)
            return

        self.get_logger().info(
            "Waiting on "
            + ", ".join(
                f"{frame} ({len(self.collected[frame])}/{self.samples})"
                for frame in missing
            ),
            throttle_duration_sec=5.0,
        )

    def average(self, translations, quaternions):
        """Average a set of samples into one translation and rotation.

        The translation is a median, which ignores the occasional
        sample thrown off by a missed corner, and the rotation is the
        eigenvector average.
        """
        return (
            np.median(translations, axis=0),
            quaternion_average(quaternions),
        )

    def deviations(self, translations, quaternions, translation, rotation):
        """Per-sample distance and angle from an average pose."""
        return (
            np.linalg.norm(translations - translation, axis=1),
            np.array([
                quaternion_angle_degrees(rotation, sample)
                for sample in quaternions
            ]),
        )

    def inliers(self, distances, angles) -> np.ndarray:
        """Mask of the samples within outlier_sigma of the average.

        A pose that jumps once, because a corner was missed on that
        frame, should not drag the average with it. A pose that is
        spread out all the time still fails the gate afterwards, since
        the spread is measured on what is kept.
        """
        keep = np.ones(len(distances), dtype=bool)

        if self.outlier_sigma <= 0.0:
            return keep

        for values in (distances, angles):
            deviation = values.std()

            if deviation <= 0.0:
                continue

            keep &= values <= values.mean() + self.outlier_sigma * deviation

        # Never throw away so much that the average means nothing.
        if keep.sum() < max(3, len(distances) // 2):
            return np.ones(len(distances), dtype=bool)

        return keep

    def summarise(self, frame: str) -> dict:
        """Average one frame's samples and measure how much they moved."""
        rows = np.asarray(self.collected[frame], dtype=np.float64)

        translations = rows[:, 0:3]
        quaternions = rows[:, 3:7]

        translation, rotation = self.average(translations, quaternions)

        distances, angles = self.deviations(
            translations, quaternions, translation, rotation
        )

        keep = self.inliers(distances, angles)
        dropped = int(len(rows) - keep.sum())

        if dropped:
            translations = translations[keep]
            quaternions = quaternions[keep]

            translation, rotation = self.average(
                translations, quaternions
            )
            distances, angles = self.deviations(
                translations, quaternions, translation, rotation
            )

        # Spread as the worst kept sample, not a standard deviation:
        # one bad frame is what makes a saved transform wrong.
        return {
            "parent_frame": self.parent_frame,
            "child_frame": frame,
            "translation": {
                "x": float(translation[0]),
                "y": float(translation[1]),
                "z": float(translation[2]),
            },
            "rotation": {
                "x": float(rotation[0]),
                "y": float(rotation[1]),
                "z": float(rotation[2]),
                "w": float(rotation[3]),
            },
            "samples": int(len(translations)),
            "dropped_samples": dropped,
            "distance_m": float(np.linalg.norm(translation)),
            "translation_spread_m": float(distances.max()),
            "rotation_spread_deg": float(angles.max()),
        }

    def write(self) -> bool:
        """Average every frame and write the file. False if refused."""
        entries = {}
        unsteady = []

        for frame in self.frames:
            entry = self.summarise(frame)
            entries[frame] = entry

            self.get_logger().info(
                f"{frame}: xyz "
                f"{entry['translation']['x']:+.4f} "
                f"{entry['translation']['y']:+.4f} "
                f"{entry['translation']['z']:+.4f} m at "
                f"{entry['distance_m']:.3f} m, spread "
                f"{entry['translation_spread_m'] * 1000:.1f} mm and "
                f"{entry['rotation_spread_deg']:.2f} deg over "
                f"{entry['samples']} samples"
                + (
                    f", {entry['dropped_samples']} dropped as outliers"
                    if entry["dropped_samples"]
                    else ""
                )
            )

            too_loose = (
                entry["translation_spread_m"]
                > self.max_translation_spread
                or entry["rotation_spread_deg"]
                > self.max_rotation_spread
            )

            if too_loose:
                unsteady.append(frame)

        if unsteady and not self.force:
            for frame in unsteady:
                entry = entries[frame]
                self.get_logger().error(
                    f"'{frame}' moved "
                    f"{entry['translation_spread_m'] * 1000:.1f} mm and "
                    f"{entry['rotation_spread_deg']:.2f} deg while "
                    "sampling, over the "
                    f"{self.max_translation_spread * 1000:.1f} mm / "
                    f"{self.max_rotation_spread:.1f} deg limit. Nothing "
                    "was written. Steady the marker and the camera, or "
                    "pass force:=true to save it anyway."
                )

            return False

        document = {
            "parent_frame": self.parent_frame,
            "recorded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "transforms": entries,
        }

        marker_transform_file.write(self.output_file, document)

        self.get_logger().info(f"Wrote {self.output_file}")

        return True

    def finish(self, succeeded: bool):
        """Stop sampling and let main() return the right exit code."""
        self.finished = True
        self.succeeded = succeeded
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = SaveMarkerTransforms()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    succeeded = node.succeeded

    node.destroy_node()

    # A SIGINT from a launch file shuts the context down through
    # rclpy's own signal handler, so shutting it down again raises.
    if rclpy.ok():
        rclpy.shutdown()

    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

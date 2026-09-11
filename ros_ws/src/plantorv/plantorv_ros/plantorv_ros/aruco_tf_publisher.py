#!/usr/bin/env python3

"""Detect single ArUco markers and publish a frame for each one.

Companion to ``charuco_tf_publisher``: the ChArUco board frames the
scene, this node frames the things standing in it, such as the robot
base. Each configured marker id gets its own TF frame, published from
the optical frame of the colour image the same way.

OpenCV puts a single marker's origin at its centre, with x to the
right along the top edge, y down and z out of the marker towards the
camera. ``marker_length`` is the side of the black square, border
included, in metres.

A single square marker is a four-point planar target, so its pose is
weaker than a board's: it flips between two solutions at oblique
angles and jitters more the further away it is. IPPE_SQUARE picks the
better solution and a VVS refinement pass tightens it, but a marker
seen edge-on is still not a good frame. Watch the reprojection error
in the annotated image.

The ArUco API was rewritten in OpenCV 4.7, and which one is available
depends on the interpreter the node runs under: ROS 2 Humble ships
OpenCV 4.5 system-wide, while the project virtualenv has 4.10. Both
are supported here, behind DETECTOR_API.
"""

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from plantorv_ros.pose_overlay import (
    draw_axes,
    draw_readout,
    pose_lines,
    transform_from_pose,
)

# OpenCV 4.7 replaced cv2.aruco.detectMarkers with ArucoDetector.
DETECTOR_API = "new" if hasattr(cv2.aruco, "ArucoDetector") else "legacy"


class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__("aruco_tf_publisher")

        # Topics
        self.declare_parameter(
            "rgb_topic",
            "/camera/color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/camera/color/camera_info",
        )

        # Marker geometry. marker_length is the side of the black
        # square in metres, the printed border included but not the
        # white margin around it.
        self.declare_parameter("marker_length", 0.10)
        self.declare_parameter("dictionary", "DICT_7X7_50")

        # Which markers to publish, and under which frame name. The two
        # lists line up: marker_ids[i] is published as
        # marker_frames[i]. A shorter marker_frames is filled in with
        # aruco_<id>.
        self.declare_parameter("marker_ids", [0])
        self.declare_parameter("marker_frames", ["robot_base_marker"])

        # A pose this far off its own corners is not usable, in pixels.
        # Zero disables the check.
        self.declare_parameter("max_reprojection_error", 4.0)

        # Output
        self.declare_parameter("axis_length", 0.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_markers", True)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter(
            "camera_info_topic"
        ).value

        self.marker_length = float(
            self.get_parameter("marker_length").value
        )
        dictionary_name = self.get_parameter("dictionary").value

        marker_ids = [
            int(value)
            for value in self.get_parameter("marker_ids").value
        ]
        marker_frames = [
            str(value)
            for value in self.get_parameter("marker_frames").value
        ]

        self.max_reprojection_error = float(
            self.get_parameter("max_reprojection_error").value
        )

        axis_length = float(self.get_parameter("axis_length").value)
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        self.publish_markers = bool(
            self.get_parameter("publish_markers").value
        )

        if self.marker_length <= 0.0:
            raise ValueError(
                f"marker_length must be positive, got {self.marker_length}"
            )

        if not marker_ids:
            raise ValueError("marker_ids must name at least one marker")

        if len(marker_frames) > len(marker_ids):
            raise ValueError(
                f"marker_frames has {len(marker_frames)} entries but "
                f"marker_ids only {len(marker_ids)}"
            )

        # Frame name per marker id, defaulted for any id left unnamed.
        self.frames = {}
        for index, marker_id in enumerate(marker_ids):
            if index < len(marker_frames):
                self.frames[marker_id] = marker_frames[index]
            else:
                self.frames[marker_id] = f"aruco_{marker_id}"

        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(
                f"Unknown ArUco dictionary '{dictionary_name}'"
            )

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )

        if DETECTOR_API == "new":
            self.detector = cv2.aruco.ArucoDetector(
                self.dictionary, cv2.aruco.DetectorParameters()
            )
            self.detector_params = None
        else:
            self.detector = None
            self.detector_params = (
                cv2.aruco.DetectorParameters_create()
            )

        # Marker corners in marker coordinates, in the order
        # detectMarkers returns them: top-left, top-right,
        # bottom-right, bottom-left, seen from the front.
        half = self.marker_length / 2.0
        self.object_points = np.array([
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

        self.axis_length = (
            axis_length if axis_length > 0.0 else self.marker_length
        )

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        self.logged_parent_frame = False

        self.tf_broadcaster = TransformBroadcaster(self)

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(
                Image, "~/debug_image", 1
            )

        self.marker_pub = None
        if self.publish_markers:
            self.marker_pub = self.create_publisher(
                MarkerArray, "~/marker_outlines", 1
            )

        wanted = ", ".join(
            f"{marker_id} as {frame}"
            for marker_id, frame in sorted(self.frames.items())
        )

        self.get_logger().info(
            f"Looking for {dictionary_name} markers of "
            f"{self.marker_length} m on {self.rgb_topic}, publishing "
            f"{wanted}, using the {DETECTOR_API} OpenCV "
            f"{cv2.__version__} API"
        )

    def camera_info_callback(self, msg: CameraInfo):
        """Latch the intrinsics; they do not change while streaming."""
        self.camera_matrix = np.array(
            msg.k, dtype=np.float64
        ).reshape(3, 3)

        self.dist_coeffs = np.array(
            msg.d, dtype=np.float64
        ).reshape(-1, 1)

        if self.dist_coeffs.size == 0:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            self.get_logger().warn(
                f"Waiting for {self.camera_info_topic}",
                throttle_duration_sec=5.0,
            )
            return

        if not self.logged_parent_frame:
            self.logged_parent_frame = True
            self.get_logger().info(
                "Publishing the marker frames as children of "
                f"'{msg.header.frame_id}', the frame the driver stamps "
                f"on {self.rgb_topic}"
            )

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids = self.detect_markers(gray)

        # Pose per wanted marker, keyed by id.
        poses = {}

        for corners, marker_id in self.wanted_markers(
            marker_corners, marker_ids
        ):
            pose = self.estimate_pose(corners)

            if pose is None:
                continue

            rvec, tvec, error = pose

            if (
                self.max_reprojection_error > 0.0
                and error > self.max_reprojection_error
            ):
                self.get_logger().warn(
                    f"Marker {marker_id} reprojects {error:.1f} px off "
                    f"its own corners, over the "
                    f"{self.max_reprojection_error:.1f} px limit; "
                    "not publishing it",
                    throttle_duration_sec=2.0,
                )
                continue

            poses[marker_id] = pose

            self.tf_broadcaster.sendTransform(
                transform_from_pose(
                    msg.header, self.frames[marker_id], rvec, tvec
                )
            )

        if not poses:
            self.get_logger().warn(
                "None of the wanted markers were found",
                throttle_duration_sec=2.0,
            )

        if self.marker_pub is not None:
            self.publish_marker_outlines(msg.header, poses)

        if self.debug_pub is not None:
            self.publish_debug(
                msg.header, frame, marker_corners, marker_ids, poses
            )

    def detect_markers(self, gray):
        """Detect every marker of the dictionary in a frame."""
        if DETECTOR_API == "new":
            marker_corners, marker_ids, _ = self.detector.detectMarkers(
                gray
            )
            return marker_corners, marker_ids

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.detector_params,
        )

        return marker_corners, marker_ids

    def wanted_markers(self, marker_corners, marker_ids):
        """Yield (corners, id) for the detected markers we publish."""
        if marker_ids is None:
            return

        for corners, marker_id in zip(
            marker_corners, np.asarray(marker_ids).reshape(-1)
        ):
            marker_id = int(marker_id)

            if marker_id in self.frames:
                yield corners, marker_id

    def estimate_pose(self, corners):
        """Return (rvec, tvec, error) of one marker, or None.

        IPPE_SQUARE is the solver meant for a four-point square, and
        the VVS pass afterwards refines it against the same corners.
        """
        image_points = np.asarray(
            corners, dtype=np.float64
        ).reshape(-1, 2)

        if len(image_points) != 4:
            return None

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not ok:
            return None

        rvec, tvec = cv2.solvePnPRefineVVS(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
        )

        return rvec, tvec, self.reprojection_error(
            image_points, rvec, tvec
        )

    def reprojection_error(self, image_points, rvec, tvec) -> float:
        """Return the RMS reprojection error of a marker's corners."""
        projected, _ = cv2.projectPoints(
            self.object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )

        residuals = projected.reshape(-1, 2) - image_points

        return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    def publish_marker_outlines(self, header, poses):
        """Outline each published marker, in its own frame."""
        markers = MarkerArray()

        half = self.marker_length / 2.0

        for index, marker_id in enumerate(sorted(poses)):
            frame = self.frames[marker_id]

            outline = Marker()
            outline.header.stamp = header.stamp
            outline.header.frame_id = frame
            outline.ns = "aruco_marker"
            outline.id = index * 2
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.scale.x = 0.004
            outline.color = ColorRGBA(r=1.0, g=0.6, b=0.1, a=1.0)
            outline.pose.orientation.w = 1.0

            for corner_x, corner_y in (
                (-half, half),
                (half, half),
                (half, -half),
                (-half, -half),
                (-half, half),
            ):
                outline.points.append(
                    Point(x=corner_x, y=corner_y, z=0.0)
                )

            label = Marker()
            label.header.stamp = header.stamp
            label.header.frame_id = frame
            label.ns = "aruco_marker"
            label.id = index * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.text = f"{frame} (id {marker_id})"
            label.scale.z = 0.03
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.pose.orientation.w = 1.0
            label.pose.position.y = half + 0.03

            markers.markers.append(outline)
            markers.markers.append(label)

        self.marker_pub.publish(markers)

    def publish_debug(
        self, header, frame, marker_corners, marker_ids, poses
    ):
        """Publish the frame with the detection drawn on top."""
        annotated = frame.copy()

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                annotated, marker_corners, marker_ids
            )

        lines = []

        for marker_id in sorted(poses):
            rvec, tvec, error = poses[marker_id]

            draw_axes(
                annotated,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                self.axis_length,
                label=self.frames[marker_id],
            )

            distance = float(
                np.linalg.norm(np.asarray(tvec, dtype=np.float64))
            )

            lines.append(
                f"id {marker_id} {self.frames[marker_id]}  "
                f"{distance:.3f} m  {error:.2f} px"
            )
            lines.extend(pose_lines(rvec, tvec, prefix="  "))

        if lines:
            colour = (0, 255, 0)
        else:
            found = (
                0 if marker_ids is None else len(marker_ids)
            )
            lines = [
                f"markers seen {found}, none of them wanted: "
                + ", ".join(
                    f"{marker_id}" for marker_id in sorted(self.frames)
                )
            ]
            colour = (0, 0, 255)

        draw_readout(annotated, lines, colour)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoTFPublisher()

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

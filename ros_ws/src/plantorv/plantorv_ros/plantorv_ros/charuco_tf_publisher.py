#!/usr/bin/env python3

"""Detect a ChArUco board in the camera stream and publish its frame.

The node estimates the board pose from the colour image and the
camera's own ``camera_info``, then broadcasts it as a TF transform from
the optical frame of the image to ``board_frame``. RViz can then show
the board axes live, which is what makes it possible to judge where the
board should be placed.

Alongside the transform the node publishes a marker array outlining the
board and an annotated image, both optional, so a bad or jittery
detection is visible instead of silently producing a wrong frame. The
annotated image carries the board axes drawn onto the colour frame
itself, lettered and with the pose printed as numbers, which is the
quickest way to tell a good placement from a bad one.

OpenCV puts the board origin on the outer corner of the first square,
with x along the short side, y along the long one and z out of the
board. ``origin_corner`` moves the published frame onto any of the
interpolated chessboard corners instead, addressed by the same id the
annotated image prints next to it.

The ChArUco API was rewritten in OpenCV 4.7, and which one is available
depends on the interpreter the node runs under: ROS 2 Humble ships
OpenCV 4.5 system-wide, while the project virtualenv has 4.10. Both are
supported here, behind BOARD_API.
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


# OpenCV 4.7 replaced the free ChArUco functions with CharucoDetector
# and CharucoBoard. Detection and pose estimation are the only places
# the two differ, so pick the API once here.
BOARD_API = "new" if hasattr(cv2.aruco, "CharucoDetector") else "legacy"


class CharucoTFPublisher(Node):
    def __init__(self):
        super().__init__("charuco_tf_publisher")

        # Topics
        self.declare_parameter(
            "rgb_topic",
            "/camera/color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/camera/color/camera_info",
        )

        # Board geometry. squares_x/squares_y count the chessboard
        # squares, not the markers, and the two lengths are in metres.
        self.declare_parameter("squares_x", 5)
        self.declare_parameter("squares_y", 7)
        self.declare_parameter("square_length", 0.04)
        self.declare_parameter("marker_length", 0.02)
        self.declare_parameter("dictionary", "DICT_6X6_250")

        # Boards generated with OpenCV < 4.6 have the marker layout
        # shifted by one square. Set this when reusing such a printout,
        # otherwise the pose is off by half a square.
        self.declare_parameter("legacy_pattern", False)

        # A pose from very few corners is unreliable, so ignore those
        # detections instead of publishing a jumping frame.
        self.declare_parameter("min_corners", 6)

        # Output
        self.declare_parameter("board_frame", "charuco_board")

        # Which point of the board the published frame sits on. -1 keeps
        # the OpenCV origin, the outer corner of the first square. Any
        # other value is the id of an interpolated chessboard corner, as
        # printed next to that corner on the annotated image.
        self.declare_parameter("origin_corner", -1)

        # Length of the drawn axes in metres. Zero or less falls back to
        # two squares, which suits most boards.
        self.declare_parameter("axis_length", 0.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("publish_markers", True)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter(
            "camera_info_topic"
        ).value

        squares_x = int(self.get_parameter("squares_x").value)
        squares_y = int(self.get_parameter("squares_y").value)
        self.square_length = float(
            self.get_parameter("square_length").value
        )
        marker_length = float(
            self.get_parameter("marker_length").value
        )
        dictionary_name = self.get_parameter("dictionary").value

        self.min_corners = int(self.get_parameter("min_corners").value)
        self.board_frame = self.get_parameter("board_frame").value

        axis_length = float(self.get_parameter("axis_length").value)
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        self.publish_markers = bool(
            self.get_parameter("publish_markers").value
        )

        if marker_length >= self.square_length:
            raise ValueError(
                "marker_length must be smaller than square_length "
                f"(got {marker_length} >= {self.square_length})"
            )

        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(
                f"Unknown ArUco dictionary '{dictionary_name}'"
            )

        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )

        legacy_pattern = bool(
            self.get_parameter("legacy_pattern").value
        )

        if BOARD_API == "new":
            self.board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                self.square_length,
                marker_length,
                dictionary,
            )
            self.board.setLegacyPattern(legacy_pattern)
            self.detector = cv2.aruco.CharucoDetector(self.board)
            self.dictionary = dictionary
            self.detector_params = None
        else:
            self.board = cv2.aruco.CharucoBoard_create(
                squares_x,
                squares_y,
                self.square_length,
                marker_length,
                dictionary,
            )
            self.detector = None
            self.dictionary = dictionary
            self.detector_params = (
                cv2.aruco.DetectorParameters_create()
            )

            # OpenCV 4.5 only knows the pre-4.6 marker layout, so a
            # board generated by a newer OpenCV is read with its origin
            # off by one square.
            if not legacy_pattern:
                self.get_logger().warn(
                    f"OpenCV {cv2.__version__} implements the pre-4.6 "
                    "ChArUco layout only; legacy_pattern:=false is "
                    "ignored. Run the node under an interpreter with "
                    "OpenCV >= 4.7 for a board generated by one."
                )

        self.axis_length = (
            axis_length
            if axis_length > 0.0
            else 2.0 * self.square_length
        )

        # Board size in metres, used for the outline marker.
        self.board_width = squares_x * self.square_length
        self.board_height = squares_y * self.square_length

        # A ChArUco board carries one marker per white square, and its
        # ids start at 0, so an id at or above this count cannot belong
        # to the board described by the parameters.
        self.marker_count = (squares_x * squares_y) // 2

        # One-shot logging, so the parent frame and any board mismatch
        # are reported once instead of every frame.
        self.logged_parent_frame = False
        self.logged_id_mismatch = False

        # Offset of the published frame from the OpenCV board origin,
        # expressed in board coordinates.
        self.origin_offset = self.resolve_origin_offset(
            int(self.get_parameter("origin_corner").value)
        )

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # Seed for the iterative solver, see estimate_pose.
        self.last_rvec = None
        self.last_tvec = None

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
                MarkerArray, "~/board_markers", 1
            )

        self.get_logger().info(
            f"Looking for a {squares_x}x{squares_y} ChArUco board "
            f"({dictionary_name}, square {self.square_length} m, "
            f"marker {marker_length} m) on {self.rgb_topic} "
            f"using the {BOARD_API} OpenCV {cv2.__version__} API"
        )

    def chessboard_corners(self) -> np.ndarray:
        """Return the board's interpolated corners, in board coordinates."""
        if BOARD_API == "new":
            return np.asarray(
                self.board.getChessboardCorners(), dtype=np.float64
            )

        return np.asarray(
            self.board.chessboardCorners, dtype=np.float64
        )

    def resolve_origin_offset(self, origin_corner: int) -> np.ndarray:
        """Turn an origin_corner id into an offset in board coordinates."""
        if origin_corner < 0:
            return np.zeros(3, dtype=np.float64)

        corners = self.chessboard_corners()

        if origin_corner >= len(corners):
            raise ValueError(
                f"origin_corner {origin_corner} is out of range; this "
                f"board has {len(corners)} corners, so the ids run 0 to "
                f"{len(corners) - 1}"
            )

        offset = corners[origin_corner].reshape(3)

        self.get_logger().info(
            f"Publishing the frame on corner {origin_corner}, "
            f"{offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f} m from "
            "the OpenCV board origin"
        )

        return offset

    def apply_origin_offset(self, rvec, tvec):
        """Move a board pose onto the configured origin corner.

        The rotation is unchanged: the offset is a translation inside
        the board plane, so both frames keep the same axes.
        """
        t = np.asarray(tvec, dtype=np.float64).reshape(3)

        if not self.origin_offset.any():
            return rvec, t

        R, _ = cv2.Rodrigues(rvec)

        return rvec, t + R @ self.origin_offset

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
                f"Publishing {self.board_frame} as a child of "
                f"'{msg.header.frame_id}', the frame the driver stamps "
                f"on {self.rgb_topic}. Set that as the RViz fixed "
                "frame, or make sure the driver publishes TF to it."
            )

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = self.detect_board(gray)

        self.check_board_match(marker_ids)

        pose = self.estimate_pose(charuco_corners, charuco_ids)

        # The pose from the solver is that of the OpenCV board origin;
        # everything downstream works on the frame the user asked for.
        frame_pose = (
            None if pose is None else self.apply_origin_offset(*pose)
        )

        if frame_pose is not None:
            self.tf_broadcaster.sendTransform(
                transform_from_pose(
                    msg.header, self.board_frame, *frame_pose
                )
            )

            if self.marker_pub is not None:
                self.publish_board_markers(msg.header)
        else:
            self.get_logger().warn(
                "No usable ChArUco detection",
                throttle_duration_sec=2.0,
            )

        if self.debug_pub is not None:
            self.publish_debug(
                msg.header,
                frame,
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,
                pose,
                frame_pose,
            )

    def check_board_match(self, marker_ids):
        """Warn when the detected markers cannot be this board.

        A board described by the wrong squares_x/squares_y still solves
        to a pose, only against the wrong object points, which puts the
        origin somewhere inside the printout instead of on its corner.
        Marker ids beyond the end of the board are the clearest sign of
        that, so report them once.
        """
        if self.logged_id_mismatch or marker_ids is None:
            return

        if len(marker_ids) == 0:
            return

        highest = int(np.max(marker_ids))

        if highest < self.marker_count:
            return

        self.logged_id_mismatch = True
        self.get_logger().error(
            f"Detected marker id {highest}, but the configured board "
            f"holds only {self.marker_count} markers, ids 0 to "
            f"{self.marker_count - 1}. squares_x, squares_y or "
            "dictionary do not match the printed board, so the pose is "
            "solved against the wrong corners and its origin will not "
            "sit on a board corner."
        )

    def detect_board(self, gray):
        """Detect markers and interpolated ChArUco corners in a frame.

        Returns the same four values under both OpenCV APIs:
        (charuco_corners, charuco_ids, marker_corners, marker_ids).
        """
        if BOARD_API == "new":
            return self.detector.detectBoard(gray)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.detector_params,
        )

        if marker_ids is None or len(marker_ids) == 0:
            return None, None, marker_corners, marker_ids

        _, charuco_corners, charuco_ids = (
            cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                self.board,
            )
        )

        return charuco_corners, charuco_ids, marker_corners, marker_ids

    def estimate_pose(self, charuco_corners, charuco_ids):
        """Return (rvec, tvec) of the board, or None when unreliable."""
        if charuco_ids is None or len(charuco_ids) < self.min_corners:
            return None

        # Reuse the previous pose as the starting point, so consecutive
        # frames converge to the same one of the two solutions a
        # near-planar target admits.
        use_guess = (
            self.last_rvec is not None and self.last_tvec is not None
        )

        rvec_guess = np.copy(self.last_rvec) if use_guess else None
        tvec_guess = np.copy(self.last_tvec) if use_guess else None

        if BOARD_API == "new":
            object_points, image_points = self.board.matchImagePoints(
                charuco_corners, charuco_ids
            )

            if object_points is None or len(object_points) < 4:
                return None

            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                rvec_guess,
                tvec_guess,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                self.camera_matrix,
                self.dist_coeffs,
                rvec_guess,
                tvec_guess,
                useExtrinsicGuess=use_guess,
            )

        if not ok:
            return None

        self.last_rvec = rvec
        self.last_tvec = tvec

        return rvec, tvec

    def publish_board_markers(self, header):
        """Outline the board and label it, in the board's own frame."""
        markers = MarkerArray()

        # The OpenCV board origin sits on the board's own top-left
        # corner, with x to the right and y down along the printout, so
        # the outline spans the whole board from there. origin_corner
        # moves the frame, so the outline moves the other way.
        x0 = -self.origin_offset[0]
        y0 = -self.origin_offset[1]
        x1 = x0 + self.board_width
        y1 = y0 + self.board_height

        outline = Marker()
        outline.header.stamp = header.stamp
        outline.header.frame_id = self.board_frame
        outline.ns = "charuco_board"
        outline.id = 0
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.scale.x = 0.004
        outline.color = ColorRGBA(r=0.1, g=0.9, b=0.3, a=1.0)
        outline.pose.orientation.w = 1.0

        for corner_x, corner_y in (
            (float(x0), float(y0)),
            (float(x1), float(y0)),
            (float(x1), float(y1)),
            (float(x0), float(y1)),
            (float(x0), float(y0)),
        ):
            outline.points.append(
                Point(x=corner_x, y=corner_y, z=0.0)
            )

        label = Marker()
        label.header.stamp = header.stamp
        label.header.frame_id = self.board_frame
        label.ns = "charuco_board"
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.text = self.board_frame
        label.scale.z = 0.03
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.pose.orientation.w = 1.0
        label.pose.position.x = float((x0 + x1) / 2.0)
        label.pose.position.y = float(y0 - 0.03)
        label.pose.position.z = 0.0

        markers.markers.append(outline)
        markers.markers.append(label)

        self.marker_pub.publish(markers)

    def draw_board_outline(self, image, rvec, tvec):
        """Project the board's own edges, as the parameters describe it.

        The quad only lines up with the printed board when squares_x,
        squares_y and square_length match it. When it does not, the
        pose is being solved against the wrong geometry, which is what
        puts the origin somewhere other than a board corner.

        The pose passed in is the board pose, not the published frame,
        so the quad is drawn from the OpenCV origin regardless of
        origin_corner.
        """
        edges = np.float32([
            [0.0, 0.0, 0.0],
            [self.board_width, 0.0, 0.0],
            [self.board_width, self.board_height, 0.0],
            [0.0, self.board_height, 0.0],
        ]).reshape(-1, 3)

        projected, _ = cv2.projectPoints(
            edges,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )

        if not np.all(np.isfinite(projected)):
            return

        cv2.polylines(
            image,
            [np.int32(projected).reshape(-1, 2)],
            True,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def reprojection_error(
        self, charuco_corners, charuco_ids, rvec, tvec
    ) -> float:
        """Return the RMS reprojection error of the detected corners."""
        corners = self.chessboard_corners()
        ids = np.asarray(charuco_ids).reshape(-1)

        object_points = corners[ids].reshape(-1, 3).astype(np.float32)

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )

        observed = np.asarray(
            charuco_corners, dtype=np.float64
        ).reshape(-1, 2)

        residuals = projected.reshape(-1, 2) - observed

        return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    def publish_debug(
        self,
        header,
        frame,
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
        pose,
        frame_pose,
    ):
        """Publish the frame with the detection drawn on top.

        ``pose`` is the board pose from the solver and ``frame_pose``
        the published one, which differ when origin_corner is set.
        """
        annotated = frame.copy()

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                annotated, marker_corners, marker_ids
            )

        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                annotated,
                charuco_corners,
                charuco_ids,
                (255, 0, 255),
            )

        corner_count = 0 if charuco_ids is None else len(charuco_ids)

        if frame_pose is not None:
            board_rvec, board_tvec = pose
            rvec, tvec = frame_pose

            # The yellow quad first, so the axes stay on top of it.
            self.draw_board_outline(annotated, board_rvec, board_tvec)

            draw_axes(
                annotated,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                self.axis_length,
            )

            error = self.reprojection_error(
                charuco_corners, charuco_ids, board_rvec, board_tvec
            )

            distance = float(
                np.linalg.norm(np.asarray(tvec, dtype=np.float64))
            )

            lines = [
                f"corners {corner_count}  "
                f"distance {distance:.3f} m  "
                f"error {error:.2f} px"
            ]
            lines.extend(pose_lines(rvec, tvec))
            colour = (0, 255, 0)
        else:
            lines = [f"corners {corner_count}  no pose"]
            colour = (0, 0, 255)

        draw_readout(annotated, lines, colour)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CharucoTFPublisher()

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

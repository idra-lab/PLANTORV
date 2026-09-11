#!/usr/bin/env python3

"""Shared pose maths and image overlay for the marker pose nodes.

``charuco_tf_publisher`` and ``aruco_tf_publisher`` solve for different
targets but publish and draw the result the same way, so the frame
conversions and the annotation live here.
"""

import math

import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped

# drawFrameAxes paints x red, y green and z blue, in BGR.
AXIS_LABELS = (
    ("X", (0, 0, 255)),
    ("Y", (0, 255, 0)),
    ("Z", (255, 0, 0)),
)


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple:
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return x, y, z, w


def rotation_matrix_to_euler_degrees(R: np.ndarray) -> tuple:
    """Return the (roll, pitch, yaw) of a rotation matrix, in degrees.

    Fixed-axis XYZ convention, matching what RViz shows for a TF frame.
    """
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))

    if abs(R[2, 0]) < 0.99999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: roll and yaw are not separable, so put all the
        # rotation in roll.
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def transform_from_pose(header, child_frame, rvec, tvec) -> TransformStamped:
    """Build the transform of a solved pose, in the image's own frame.

    The parent is whatever frame the driver stamped on the image, so
    the caller never has to name it.
    """
    R, _ = cv2.Rodrigues(rvec)
    x, y, z, w = rotation_matrix_to_quaternion(R)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    transform = TransformStamped()
    transform.header.stamp = header.stamp
    transform.header.frame_id = header.frame_id
    transform.child_frame_id = child_frame

    transform.transform.translation.x = float(t[0])
    transform.transform.translation.y = float(t[1])
    transform.transform.translation.z = float(t[2])

    transform.transform.rotation.x = x
    transform.transform.rotation.y = y
    transform.transform.rotation.z = z
    transform.transform.rotation.w = w

    return transform


def draw_axes(
    image, camera_matrix, dist_coeffs, rvec, tvec, axis_length, label=None
):
    """Draw a pose as lettered axes, optionally named at the origin."""
    cv2.drawFrameAxes(
        image, camera_matrix, dist_coeffs, rvec, tvec, axis_length
    )

    axes = np.float32([
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ]).reshape(-1, 3)

    tips, _ = cv2.projectPoints(
        axes, rvec, tvec, camera_matrix, dist_coeffs
    )

    height, width = image.shape[:2]

    for tip, (name, colour) in zip(tips.reshape(-1, 2), AXIS_LABELS):
        if not np.all(np.isfinite(tip)):
            continue

        x = int(round(tip[0])) + 6
        y = int(round(tip[1])) - 6

        # An axis pointing away from the camera can project outside the
        # frame, where putText would silently drop the letter.
        x = max(0, min(width - 20, x))
        y = max(20, min(height - 4, y))

        cv2.putText(
            image,
            name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )

    if label is None:
        return

    origin, _ = cv2.projectPoints(
        np.float32([[0.0, 0.0, 0.0]]),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    if not np.all(np.isfinite(origin)):
        return

    point = origin.reshape(2)

    cv2.putText(
        image,
        label,
        (
            max(0, min(width - 10, int(round(point[0])) + 8)),
            max(14, min(height - 4, int(round(point[1])) + 20)),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_readout(image, lines, colour):
    """Print a few lines over a darkened band at the top of the image."""
    if not lines:
        return

    # Shrink the text until the longest line fits the image width,
    # so a narrow frame or a long frame name does not get clipped.
    width = image.shape[1]
    longest = max(lines, key=len)
    scale = 0.7

    while scale > 0.35:
        (text_width, _), _ = cv2.getTextSize(
            longest, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
        )

        if text_width <= width - 20:
            break

        scale -= 0.05

    step = int(round(40 * scale))

    # A marker origin often lands near the top-left corner, right under
    # the readout, so darken a band behind the text to keep both
    # legible.
    band_height = min(image.shape[0], step * len(lines) + step // 2)
    band = image[0:band_height, :]
    cv2.addWeighted(band, 0.35, np.zeros_like(band), 0.0, 0.0, band)

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, step + index * step),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            2,
            cv2.LINE_AA,
        )


def pose_lines(rvec, tvec, prefix="") -> list:
    """Describe a pose as the two lines the readout prints."""
    R, _ = cv2.Rodrigues(rvec)
    roll, pitch, yaw = rotation_matrix_to_euler_degrees(R)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    return [
        f"{prefix}xyz {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f} m",
        f"{prefix}rpy {roll:+7.1f} {pitch:+7.1f} {yaw:+7.1f} deg",
    ]


def quaternion_average(quaternions) -> np.ndarray:
    """Average (x, y, z, w) quaternions into one.

    Uses the dominant eigenvector of the accumulated outer products,
    which is the standard rotation average and does not care how the
    samples are ordered. Signs are aligned first, because q and -q are
    the same rotation but would cancel in the sum.
    """
    Q = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)

    signs = np.sign(Q @ Q[0])
    signs[signs == 0.0] = 1.0
    Q = Q * signs[:, None]

    values, vectors = np.linalg.eigh(Q.T @ Q)
    average = vectors[:, int(np.argmax(values))]

    if average[3] < 0.0:
        average = -average

    return average / np.linalg.norm(average)


def quaternion_angle_degrees(first, second) -> float:
    """Return the angle between two (x, y, z, w) quaternions, in degrees."""
    dot = abs(
        float(
            np.dot(
                np.asarray(first, dtype=np.float64).reshape(4),
                np.asarray(second, dtype=np.float64).reshape(4),
            )
        )
    )

    return math.degrees(2.0 * math.acos(min(1.0, dot)))

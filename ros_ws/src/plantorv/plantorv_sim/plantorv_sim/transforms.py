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

"""The pose arithmetic the fake grasp needs.

Attaching an object means expressing its world pose in the tool frame, and
releasing it means undoing that. Both are one composition and one inverse, so
they are written out here rather than pulling in a transform library.

Poses are ``geometry_msgs/Pose``; quaternions are ``(x, y, z, w)`` tuples, the
order the messages use.
"""

from typing import Tuple

from geometry_msgs.msg import Pose

Quaternion = Tuple[float, float, float, float]
Vector = Tuple[float, float, float]


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Return the rotation ``a`` followed by, in ``a``'s frame, ``b``."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_inverse(q: Quaternion) -> Quaternion:
    """Return the inverse of a unit quaternion."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_rotate(q: Quaternion, v: Vector) -> Vector:
    """Rotate the vector ``v`` by the quaternion ``q``."""
    rotated = quat_multiply(quat_multiply(q, (v[0], v[1], v[2], 0.0)), quat_inverse(q))
    return rotated[0], rotated[1], rotated[2]


def pose_parts(pose: Pose) -> Tuple[Vector, Quaternion]:
    """Split a Pose into its translation and its rotation."""
    p, o = pose.position, pose.orientation
    return (p.x, p.y, p.z), (o.x, o.y, o.z, o.w)


def make_pose(translation: Vector, rotation: Quaternion) -> Pose:
    """Build a Pose out of a translation and a rotation."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = translation
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = rotation
    return pose


def compose(parent: Pose, child: Pose) -> Pose:
    """Return ``child``, given relative to ``parent``, in ``parent``'s frame."""
    pt, pr = pose_parts(parent)
    ct, cr = pose_parts(child)
    offset = quat_rotate(pr, ct)
    return make_pose(
        (pt[0] + offset[0], pt[1] + offset[1], pt[2] + offset[2]),
        quat_multiply(pr, cr),
    )


def relative_to(reference: Pose, pose: Pose) -> Pose:
    """Return ``pose``, given in some frame, expressed in ``reference``."""
    rt, rr = pose_parts(reference)
    pt, pr = pose_parts(pose)
    inverse = quat_inverse(rr)
    delta = (pt[0] - rt[0], pt[1] - rt[1], pt[2] - rt[2])
    return make_pose(quat_rotate(inverse, delta), quat_multiply(inverse, pr))


def transform_to_pose(transform) -> Pose:
    """Convert a ``geometry_msgs/Transform`` into the equivalent Pose."""
    return make_pose(
        (transform.translation.x, transform.translation.y, transform.translation.z),
        (
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ),
    )

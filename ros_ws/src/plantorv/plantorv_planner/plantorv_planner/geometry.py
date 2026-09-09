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

"""Grasp geometry, and the one measure the planner optimises.

Everything the planner does to a cube it does from straight above, so the only
pose it ever has to construct is a top-down one, and the only freedom in it is
the rotation about the vertical.
"""

import math
from typing import Iterable, List, Sequence

from geometry_msgs.msg import Pose, Quaternion


def top_down_quaternion(yaw: float) -> Quaternion:
    """Return the orientation of a tool pointing straight down at ``yaw``.

    The tool's z axis points along world -z and its x axis is turned by ``yaw``
    about the vertical, which is ``Rz(yaw) * Rx(pi)``. Written out rather than
    composed, because it is a one-liner and this is the only rotation in the
    package.
    """
    half = yaw / 2.0
    return Quaternion(x=math.cos(half), y=math.sin(half), z=0.0, w=0.0)


def yaw_of(orientation: Quaternion) -> float:
    """Return the rotation about z of a quaternion, radians."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def square_symmetric_yaw(yaw: float) -> float:
    """Fold a yaw into ``[-pi/4, pi/4]``.

    A cube looks the same every quarter turn, so a cube lying at 80 degrees is
    better approached as one lying at -10 than by winding the wrist most of the
    way round. Only sound for objects with four-fold symmetry, which is what
    this scene has.
    """
    folded = math.remainder(yaw, math.pi / 2.0)
    return folded


def pose_above(x: float, y: float, z: float, yaw: float = 0.0) -> Pose:
    """A top-down tool pose at a point."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation = top_down_quaternion(yaw)
    return pose


def raised(pose: Pose, height: float) -> Pose:
    """The same pose, ``height`` metres higher."""
    lifted = Pose()
    lifted.position.x = pose.position.x
    lifted.position.y = pose.position.y
    lifted.position.z = pose.position.z + height
    lifted.orientation = pose.orientation
    return lifted


def joint_path_length(positions: Iterable[Sequence[float]]) -> float:
    """Total joint-space length of a trajectory, in radians.

    This is what "optimal" means for the planner: of several plans that all
    reach the goal, the shortest one through joint space is the one that moves
    the arm least, and in this cell that is also reliably the quickest and the
    least likely to swing the elbow over the trays. It is a crude measure --
    it weights every joint the same -- but it is monotone in the thing that
    matters and costs nothing to evaluate.
    """
    points: List[Sequence[float]] = list(positions)
    total = 0.0
    for before, after in zip(points, points[1:]):
        total += math.sqrt(sum((b - a) ** 2 for a, b in zip(before, after)))
    return total

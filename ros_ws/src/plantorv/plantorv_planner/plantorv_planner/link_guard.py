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

__maintainers__ = ["Enrico Saccon", "Tommaso Faraci"]

"""Does the whole arm clear the furniture, not just the tool.

This is the check that was missing. The workspace guard in cartesian_client
tests where ``tool0`` goes: a height floor and a reach annulus, both about one
point. It passed a move whose tool path was exactly right and whose forearm
came at the table, because reaching 0.46 m out with base_link at 0.885 puts
the arm near full extension and nearly horizontal, and in that posture the
elbow hangs well below the tool.

So this walks the arm's own skeleton instead. For each configuration along a
path it computes the origin of every joint frame from the robot description,
joins them into a polyline, and tests that polyline against the boxes the
description gives the table and the pedestal, inflated by a link radius.

Both halves come out of ``/robot_description``, which is the same URDF the
controllers and TF use. Nothing here reads scene.yaml or trusts a second copy
of the layout.

What it is not:

    It is not a collision checker. Link geometry is a polyline with a radius,
    not the meshes; obstacles are the boxes in the description, not the world.
    It will not catch the arm hitting something nobody described, and it does
    not check the arm against itself -- joint limits and the SRDF's disabled
    pairs are what covered that, and self-collision on a UR3 needs the wrist
    folded much further than anything here asks for.

    It is a cheap, explicit answer to one question that had no answer: does
    any part of this arm go through the table on the way. That question is
    what put a real arm at risk.
"""

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import String

from plantorv_planner.errors import PlanningError

#: Frames whose origins make up the arm's skeleton, base outwards.
#:
#: Starts at the shoulder, not at base_link. The segment from base_link to the
#: shoulder runs straight up out of the pedestal the arm is bolted to, so it
#: touches the pedestal's own box by construction and there is nothing to
#: learn from checking it.
ARM_FRAMES = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
)

#: Per obstacle: the first ARM_FRAMES index whose segments are tested against
#: it, and how much of ``link_radius`` to use as the shell.
#:
#: **Which links.** The table is checked against the whole arm. The pedestal is
#: not, and the reason is geometric rather than a fudge: the arm is bolted to
#: its top face, so the shoulder sits 0.15 m directly above it and the segment
#: out to the upper arm hugs it. Inflating that face upward by a link radius
#: puts the shoulder permanently inside the obstacle and refuses every move.
#: From the forearm onwards the question is real, so those are checked.
#:
#: **How much margin.** The pedestal gets a thinner shell, and this is measured
#: rather than guessed: on a normal pick of the nearest block the forearm
#: passes 59.8 mm from the pedestal's top front corner. A 60 mm shell clips
#: that by 0.2 mm and refuses the pick. The forearm is about 40 mm in radius,
#: so the true clearance there is near 20 mm -- tight, and a consequence of how
#: near the blocks sit to the base, but not a collision.
#:
#: So the pedestal is checked at three quarters of the radius, which leaves
#: 5 mm of margin over the link itself and still stops a real approach. The
#: table keeps the full shell: it is the large flat thing the arm has actually
#: been driven at, and there is no legitimate reason to pass within 60 mm of it.
OBSTACLES = {
    "table_link": {"from_frame": 0, "scale": 1.0},
    "stand_link": {"from_frame": 2, "scale": 0.75},
}


@dataclass
class Box:
    """An axis-aligned-in-its-own-frame box, placed in the world."""

    name: str
    centre: Tuple[float, float, float]
    half: Tuple[float, float, float]
    #: World-to-box rotation, as three rows. Identity for everything here, but
    #: carried so a tilted table would not silently be checked as upright.
    rows: Tuple[Tuple[float, float, float], ...]

    #: First ARM_FRAMES index whose segments are tested against this box.
    from_frame: int = 0
    #: Fraction of ``link_radius`` used as this box's shell.
    scale: float = 1.0

    def penetration(self, point, radius: float) -> float:
        """How far inside the inflated box a point is, in metres. 0 if outside."""
        local = [
            sum(self.rows[i][k] * (point[k] - self.centre[k]) for k in range(3))
            for i in range(3)
        ]
        # Distance from the box surface, negative inside.
        outside = [abs(local[i]) - self.half[i] for i in range(3)]
        if all(v <= 0.0 for v in outside):
            # Inside the box proper; report the depth to the nearest face.
            return radius - max(outside)
        gap = math.sqrt(sum(max(v, 0.0) ** 2 for v in outside))
        return max(0.0, radius - gap)


class LinkGuard:
    """Tests whole configurations, and paths between them, against the furniture.

    Parameters
    ----------
    node : rclpy.node.Node
        The node the ``/robot_description`` subscription is created on.
    joint_names : sequence of str
        The actuated joints, in the order configurations are given in.
    callback_group : rclpy.callback_groups.CallbackGroup
        Reentrant, since callers block waiting for the description.
    """

    def __init__(self, node, joint_names: Sequence[str], callback_group):
        self.node = node
        self.joint_names = list(joint_names)
        self._joints: Optional[Dict] = None
        self._boxes: List[Box] = []
        # robot_state_publisher latches the description, so a late subscriber
        # still receives it -- but only with matching durability.
        node.create_subscription(
            String,
            "/robot_description",
            self._on_description,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
            callback_group=callback_group,
        )

    # -- the description -------------------------------------------------

    def _on_description(self, msg: String) -> None:
        try:
            self._parse(msg.data)
        except Exception as error:  # noqa: BLE001 - a bad URDF must not kill the node
            self.node.get_logger().error(f"could not read /robot_description: {error}")

    def _parse(self, urdf: str) -> None:
        root = ElementTree.fromstring(urdf)
        joints = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            joints[element.get("name")] = {
                "type": element.get("type"),
                "parent": element.find("parent").get("link"),
                "child": element.find("child").get("link"),
                "xyz": _triple(origin.get("xyz") if origin is not None else None),
                "rpy": _triple(origin.get("rpy") if origin is not None else None),
                "axis": _triple(axis.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
            }
        self._joints = joints

        boxes = []
        for element in root.findall("link"):
            if element.get("name") not in OBSTACLES:
                continue
            for collision in element.findall("collision"):
                box = collision.find("geometry/box")
                if box is None:
                    continue
                size = _triple(box.get("size"))
                pose = self._frame(element.get("name"), {})
                if pose is None:
                    continue
                boxes.append(
                    Box(
                        name=element.get("name"),
                        centre=(pose[0][3], pose[1][3], pose[2][3]),
                        half=(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0),
                        rows=tuple(tuple(pose[i][k] for k in range(3)) for i in range(3)),
                        from_frame=OBSTACLES[element.get("name")]["from_frame"],
                        scale=OBSTACLES[element.get("name")]["scale"],
                    )
                )
        self._boxes = boxes
        self.node.get_logger().info(
            "link guard reading "
            + ", ".join(
                f"{b.name} {tuple(round(2 * h, 3) for h in b.half)} at "
                f"({b.centre[0]:.2f}, {b.centre[1]:.2f}, {b.centre[2]:.2f}) "
                f"from {ARM_FRAMES[b.from_frame]} at {b.scale:.2f} radius"
                for b in boxes
            )
        )

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until the description has arrived and yielded obstacles."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._joints is not None and self._boxes:
                return True
            time.sleep(0.1)
        return False

    # -- kinematics ------------------------------------------------------

    def _frame(self, link: str, values: Dict[str, float]):
        """World transform of ``link``, or None if it is not in the tree."""
        by_child = {j["child"]: (n, j) for n, j in self._joints.items()}
        chain = []
        current = link
        while current in by_child:
            name, joint = by_child[current]
            chain.append((name, joint))
            current = joint["parent"]
        transform = _identity()
        for name, joint in reversed(chain):
            transform = _multiply(transform, _translation(joint["xyz"]))
            transform = _multiply(transform, _rotation(joint["rpy"]))
            if joint["type"] in ("revolute", "continuous"):
                transform = _multiply(
                    transform, _about_axis(joint["axis"], values.get(name, 0.0))
                )
        return transform

    def skeleton(self, configuration: Sequence[float]) -> List[Tuple[float, float, float]]:
        """The origins of ARM_FRAMES for one configuration, in world."""
        if self._joints is None:
            raise PlanningError("no /robot_description yet; is robot_state_publisher up?")
        values = dict(zip(self.joint_names, (float(v) for v in configuration)))
        points = []
        for frame in ARM_FRAMES:
            transform = self._frame(frame, values)
            if transform is None:
                raise PlanningError(f"'{frame}' is not in the robot description")
            points.append((transform[0][3], transform[1][3], transform[2][3]))
        return points

    # -- the check -------------------------------------------------------

    def check_configuration(self, configuration: Sequence[float], radius: float) -> None:
        """Raise if any part of the arm's skeleton is inside the furniture."""
        points = self.skeleton(configuration)
        for index, (start, end) in enumerate(zip(points, points[1:])):
            for sample in _along(start, end, step=0.02):
                for box in self._boxes:
                    if index < box.from_frame:
                        continue
                    shell = radius * box.scale
                    depth = box.penetration(sample, shell)
                    if depth > 0.0:
                        # Depth into the shell, not distance to the box: a
                        # point can be inside the box itself, where a distance
                        # would come out negative and read as nonsense.
                        raise PlanningError(
                            f"{ARM_FRAMES[index]} to {ARM_FRAMES[index + 1]} reaches "
                            f"{depth * 1000:.1f} mm into the {shell * 1000:.0f} mm "
                            f"clearance around {box.name}, at "
                            f"({sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f})"
                        )

    def check_path(
        self, configurations: Sequence[Sequence[float]], radius: float
    ) -> None:
        """Raise if any configuration along a path fails.

        Every point of the trajectory is tested, not only its ends. A joint
        move is a straight line in joint space and the arm's path through the
        world is not straight, so passing at both ends says nothing about the
        middle -- which is exactly where a swing through the table would be.
        """
        if not self._boxes:
            raise PlanningError(
                "the link guard has no obstacles; /robot_description never arrived "
                "or describes no table"
            )
        for index, configuration in enumerate(configurations):
            try:
                self.check_configuration(configuration, radius)
            except PlanningError as error:
                where = index / max(len(configurations) - 1, 1)
                raise PlanningError(f"{error} ({where * 100:.0f}% along the move)")


# -- small matrix helpers ------------------------------------------------


def _triple(text: Optional[str], default=(0.0, 0.0, 0.0)):
    if not text:
        return default
    return tuple(float(v) for v in text.split())


def _identity():
    return [[1.0 if i == k else 0.0 for k in range(4)] for i in range(4)]


def _multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _translation(xyz):
    m = _identity()
    m[0][3], m[1][3], m[2][3] = xyz
    return m


def _rotation(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _about_axis(axis, angle: float):
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return _identity()
    x, y, z = x / norm, y / norm, z / norm
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _along(start, end, step: float):
    """Points along a segment at no more than ``step`` spacing, ends included."""
    length = math.sqrt(sum((b - a) ** 2 for a, b in zip(start, end)))
    count = max(1, int(math.ceil(length / max(step, 1e-4))))
    return [
        tuple(a + (b - a) * index / count for a, b in zip(start, end))
        for index in range(count + 1)
    ]

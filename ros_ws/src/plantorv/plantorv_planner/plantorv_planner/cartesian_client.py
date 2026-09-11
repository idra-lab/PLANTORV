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

"""Straight tool moves, driven by cartesian_motion_controller.

This replaces ``MoveItClient.plan_cartesian``, and it replaces it for one
reason: MoveIt's ``/compute_cartesian_path`` solves inverse kinematics
separately at every waypoint and then hopes that consecutive solutions are
near each other. They are not always, and when they are not the controller
interpolates between two different arm configurations in joint space and the
forearm sweeps through whatever lies between them. ``jump_threshold`` in
moveit_client is a heuristic against that, and it is the reason a traverse
that should have cost 2 rad once cost 23.5.

cartesian_motion_controller cannot do this. It never solves inverse
kinematics: it integrates a simulated twin of the arm forward from the
measured joint positions towards the target pose, so the configuration it
produces is continuous with the one the arm is already in. There is no branch
to jump to.

What it also cannot do is plan. It is a control law, not a planner, and there
are three consequences that shape everything below.

**It follows a target, so the target has to move.** Publish one distant pose
and the arm goes there along whatever curve the PD gains produce, which is not
the straight line the caller asked for. So ``move_linear`` interpolates the
line itself and streams the intermediate poses at ``rate`` Hz. That
interpolation is the part MoveIt used to do.

**It does not check collisions before it moves, and it cannot.** The
controller decides the arm's configuration as it goes, so there is nothing to
check in advance: the tool path is known, the joint path is not. What is
checked in advance is therefore only the tool path, against a height floor and
a reach annulus around the arm's base -- see ``check_path`` -- and that says
nothing about the elbow.

That gap put a real forearm at the table. What closes it is a check that runs
*during* the move instead of before it: every cycle, the arm's measured
configuration goes to a LinkGuard, and the move is abandoned the moment any
link's shell touches the furniture. It stops the arm roughly a link radius
short of contact rather than predicting the swing, which is less than a
planner would give and enough to prevent the thing that happened.

**It reports nothing.** A target the arm cannot reach is not refused; the
solver converges as near as it can and stays there, straining, which looks
exactly like a slow move. So ``move_linear`` watches the measured pose, and
gives up on a target the arm stops approaching rather than waiting out the
timeout.

Every call blocks the caller until the arm has converged or the attempt has
failed, the same as MoveItClient, and it is safe for the same reason: the node
that owns this puts it in a reentrant callback group and spins a
multi-threaded executor, so the joint states and transforms it waits for
arrive on other threads.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener

from plantorv_planner.errors import PlanningError

#: Joint motion below this, in radians, is encoder noise rather than travel.
#: Without it the measured path length of a stationary arm grows all day.
TRAVEL_DEADBAND = 1e-5


@dataclass
class Motion:
    """What a Cartesian move cost.

    ``path_length`` is measured, not planned: the joint travel the arm
    actually did while the move was running, in radians. That keeps the field
    comparable with the planned figure MoveIt reports for the same move, which
    is the number worth watching while both back ends are in the tree.
    """

    path_length: float = 0.0
    #: Always zero. Nothing is planned, so nothing is spent planning.
    planning_time: float = 0.0
    #: Straight-line tool distance commanded, in metres.
    distance: float = 0.0

    def __add__(self, other: "Motion") -> "Motion":
        return Motion(
            path_length=self.path_length + other.path_length,
            planning_time=self.planning_time + other.planning_time,
            distance=self.distance + other.distance,
        )


class CartesianClient:
    """Streams tool poses to cartesian_motion_controller.

    Parameters
    ----------
    node : rclpy.node.Node
        The node the interfaces are created on.
    controller : str
        Name of the controller, which is also the namespace of its topics.
    tool_link : str
        The frame whose pose is commanded. Must be the controller's
        ``end_effector_link``, or the arm goes to the wrong place.
    planning_frame : str
        Frame the caller's poses are in.
    controller_frame : str
        The controller's ``robot_base_link``. It refuses a target in any other
        frame rather than transforming one, so poses are transformed here when
        the two differ. On this cell it is the planning frame, because the
        workcell URDF is rooted there and that saves transforming every target.
    arm_base_frame : str
        The link the arm is actually mounted on, which is what ``check_path``
        measures reach from. Not the same thing as ``controller_frame``, and
        the reason this is a separate parameter: with the controller working in
        ``world``, measuring reach from ``controller_frame`` measures it from
        the middle of the desk, where every pose in the cell is over a metre
        from the origin and the guard refuses the first move of every run.
    joint_names : sequence of str
        The arm's joints, in the order path length is measured over.
    switcher : ControllerSwitcher
        Used to make this controller the active one before commanding it.
    guard : LinkGuard, optional
        Checked against the arm's measured configuration on every cycle of a
        move. See ``_check_links``.
    callback_group : rclpy.callback_groups.CallbackGroup
        Must be reentrant, for the reason in the module docstring.
    """

    def __init__(
        self,
        node,
        controller: str,
        tool_link: str,
        planning_frame: str,
        controller_frame: str,
        arm_base_frame: str,
        joint_names: Sequence[str],
        switcher,
        callback_group,
        guard=None,
    ):
        self.node = node
        self.controller = controller
        self.tool_link = tool_link
        self.planning_frame = planning_frame
        self.controller_frame = controller_frame
        self.arm_base_frame = arm_base_frame
        self.joint_names = list(joint_names)
        self.switcher = switcher
        self.guard = guard
        self.link_radius = 0.06

        self.target_publisher = node.create_publisher(
            PoseStamped, f"/{controller}/target_frame", 3
        )
        self.buffer = Buffer()
        # spin_thread is left false deliberately. The listener puts its
        # subscriptions in a reentrant group of its own, so they are served
        # while this class blocks; asking it for a thread would spin the node a
        # second time, alongside the executor that already does.
        self.listener = TransformListener(self.buffer, node, spin_thread=False)

        self._travel = 0.0
        self._last_positions: Optional[List[float]] = None
        self._base_offset: Optional[Pose] = None
        node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10, callback_group=callback_group
        )

    # -- state -----------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        """Keep a running total of how far the arm has moved.

        Accumulated here rather than sampled by the move loop, so that the
        figure does not depend on how fast the loop happens to run.
        """
        lookup = dict(zip(msg.name, msg.position))
        positions = [lookup.get(name) for name in self.joint_names]
        if any(value is None for value in positions):
            return
        if self._last_positions is not None:
            step = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(positions, self._last_positions))
            )
            if step > TRAVEL_DEADBAND:
                self._travel += step
        self._last_positions = [float(value) for value in positions]

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until the controller, the joint states and TF are all there."""
        deadline = time.monotonic() + timeout
        if not self.switcher.wait_until_ready(timeout=timeout):
            return False
        while time.monotonic() < deadline:
            if self._last_positions is not None and self._can_see_tool():
                base = self._base_position()
                # Logged because the reach guard is measured from here, and a
                # wrong frame does not fail until the first move is refused
                # for a distance that looks nothing like the arm.
                self.node.get_logger().info(
                    f"reach measured from '{self.arm_base_frame}' at "
                    f"({base.x:.3f}, {base.y:.3f}, {base.z:.3f}) in "
                    f"'{self.planning_frame}'"
                )
                return True
            time.sleep(0.1)
        return False

    def _can_see_tool(self) -> bool:
        try:
            self.buffer.lookup_transform(self.planning_frame, self.tool_link, Time())
        except TransformException:
            return False
        return True

    def current_pose(self) -> Pose:
        """Where the tool is now, in the planning frame."""
        try:
            transform = self.buffer.lookup_transform(
                self.planning_frame, self.tool_link, Time()
            )
        except TransformException as error:
            raise PlanningError(
                f"no transform from '{self.planning_frame}' to '{self.tool_link}': {error}"
            )
        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation = transform.transform.rotation
        return pose

    def _base_position(self):
        """The arm's base, in the planning frame. Fixed, so looked up once."""
        if self._base_offset is None:
            try:
                transform = self.buffer.lookup_transform(
                    self.planning_frame, self.arm_base_frame, Time()
                )
            except TransformException as error:
                raise PlanningError(
                    f"no transform from '{self.planning_frame}' to "
                    f"'{self.arm_base_frame}': {error}"
                )
            base = Pose()
            base.position.x = transform.transform.translation.x
            base.position.y = transform.transform.translation.y
            base.position.z = transform.transform.translation.z
            self._base_offset = base
        return self._base_offset.position

    # -- the guard that stands in for collision checking -----------------

    def check_path(
        self,
        start: Pose,
        waypoints: Sequence[Pose],
        min_z: float,
        min_reach: float,
        max_reach: float,
        step: float,
    ) -> None:
        """Refuse a tool path that leaves the volume the arm may work in.

        Sampled along the line at ``step``, not only at the ends. The height
        floor and the outer reach would not need it -- a chord between two
        points above a plane stays above it, and a chord inside a ball stays
        inside -- but the inner bound is the surface of a hole, and a line
        between two points outside a hole can pass straight through it. Which
        is what a traverse across the middle of the table would do.

        ``step`` is the old ``cartesian_step``: the same spacing MoveIt used to
        check waypoints at, now the spacing this checks samples at.

        This is a tool-path check and nothing more. It says where the tool may
        go; it says nothing about where the rest of the arm goes to put it
        there.
        """
        base = self._base_position()
        previous = start
        for goal in waypoints:
            for sample in _samples(previous, goal, step):
                if sample.position.z < min_z:
                    raise PlanningError(
                        f"the path reaches z = {sample.position.z:.3f}, below the floor "
                        f"of {min_z:.3f} in '{self.planning_frame}'"
                    )
                reach = math.sqrt(
                    (sample.position.x - base.x) ** 2
                    + (sample.position.y - base.y) ** 2
                    + (sample.position.z - base.z) ** 2
                )
                if reach > max_reach:
                    raise PlanningError(
                        f"the path reaches {reach:.3f} m from the arm's base, past the "
                        f"{max_reach:.3f} m limit"
                    )
                if reach < min_reach:
                    raise PlanningError(
                        f"the path passes {reach:.3f} m from the arm's base, inside the "
                        f"{min_reach:.3f} m limit"
                    )
            previous = goal

    # -- motion ----------------------------------------------------------

    def move_linear(
        self,
        waypoints: Sequence[Pose],
        speed: float,
        accel: float,
        rot_speed: float,
        rate: float,
        pos_tolerance: float,
        rot_tolerance: float,
        timeout: float,
        stall_time: float,
        start: Optional[Pose] = None,
    ) -> Motion:
        """Move the tool along straight lines through ``waypoints``.

        The first line starts where the arm measures itself to be. Every line
        after that starts at the pose the one before it was commanded to,
        rather than at the pose the arm converged to, so the path the tool is
        asked to follow is exactly the polyline the caller passed in and does
        not accumulate the convergence tolerance at each corner.

        Raises PlanningError, and leaves the arm holding still where it is, if
        the tool does not converge or stops approaching the target.
        """
        if not waypoints:
            return Motion()

        self.switcher.ensure(self.controller)
        here = start if start is not None else self.current_pose()

        before = self._travel
        total = Motion()
        try:
            for goal in waypoints:
                total = total + self._segment(
                    here,
                    goal,
                    speed=speed,
                    accel=accel,
                    rot_speed=rot_speed,
                    rate=rate,
                    pos_tolerance=pos_tolerance,
                    rot_tolerance=rot_tolerance,
                    timeout=timeout,
                    stall_time=stall_time,
                )
                here = goal
        except PlanningError:
            self.hold()
            raise
        total.path_length = self._travel - before
        return total

    def hold(self) -> None:
        """Target the pose the arm is at, so that it stops trying to move.

        Called when a move fails. The controller has no idea the goal has been
        abandoned and would go on pulling towards an unreachable target.
        """
        try:
            self._publish(self.current_pose())
        except PlanningError:
            # Nothing better is available. Leaving the old target standing is
            # bad, but so is raising over the error that got us here.
            self.node.get_logger().error("could not read the tool pose to hold at")

    def _segment(
        self,
        start: Pose,
        goal: Pose,
        speed: float,
        accel: float,
        rot_speed: float,
        rate: float,
        pos_tolerance: float,
        rot_tolerance: float,
        timeout: float,
        stall_time: float,
    ) -> Motion:
        """Stream one straight line, then wait for the arm to arrive."""
        distance = _distance(start.position, goal.position)
        angle = _angle_between(start.orientation, goal.orientation)
        period = 1.0 / max(rate, 1.0)
        duration = _duration(distance, angle, speed, accel, rot_speed)

        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            fraction = 1.0 if duration <= 0.0 else min(elapsed / duration, 1.0)
            self._check_links()
            self._publish(_interpolate(start, goal, _eased(fraction)))
            if fraction >= 1.0:
                break
            time.sleep(period)

        self._settle(goal, period, pos_tolerance, rot_tolerance, timeout, stall_time)
        return Motion(distance=distance)

    def _settle(
        self,
        goal: Pose,
        period: float,
        pos_tolerance: float,
        rot_tolerance: float,
        timeout: float,
        stall_time: float,
    ) -> None:
        """Hold the target and wait for the tool to reach it.

        Convergence is a control transient, so the tool arrives some time after
        the target stops moving, and how long depends on the gains. Waiting is
        therefore bounded two ways: by ``timeout``, and -- because the solver
        will sit forever an inch short of a pose the arm cannot make -- by how
        long it goes without getting any closer.
        """
        deadline = time.monotonic() + timeout
        closest = float("inf")
        improved_at = time.monotonic()

        while True:
            self._check_links()
            self._publish(goal)
            current = self.current_pose()
            position_error = _distance(current.position, goal.position)
            rotation_error = _angle_between(current.orientation, goal.orientation)
            if position_error <= pos_tolerance and rotation_error <= rot_tolerance:
                return

            now = time.monotonic()
            if position_error < closest - pos_tolerance / 10.0:
                closest = position_error
                improved_at = now
            elif now - improved_at > stall_time:
                raise PlanningError(
                    f"the tool stopped {position_error * 1000:.1f} mm short of "
                    f"({goal.position.x:.3f}, {goal.position.y:.3f}, {goal.position.z:.3f}) "
                    f"and is no longer approaching it; the pose is probably out of reach"
                )
            if now > deadline:
                raise PlanningError(
                    f"the tool did not converge within {timeout:.1f} s; "
                    f"{position_error * 1000:.1f} mm and {rotation_error:.3f} rad remain"
                )
            time.sleep(period)

    def _check_links(self) -> None:
        """Abandon the move if the arm has reached into the furniture.

        Reactive, not predictive. The forward-dynamics solver picks the
        configuration and does not say beforehand what it will pick, so the
        only configuration available to check is the one the arm is in. The
        guard inflates each link by a radius, so touching the shell means
        being about that far short of contact, which is the margin this buys.
        """
        if self.guard is None or self._last_positions is None:
            return
        self.guard.check_configuration(self._last_positions, self.link_radius)

    def _publish(self, pose: Pose) -> None:
        """Send one target pose, in the frame the controller insists on."""
        message = PoseStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = self.controller_frame
        if self.controller_frame == self.planning_frame:
            message.pose = pose
        else:
            try:
                transform = self.buffer.lookup_transform(
                    self.controller_frame, self.planning_frame, Time()
                )
            except TransformException as error:
                raise PlanningError(
                    f"no transform from '{self.planning_frame}' to "
                    f"'{self.controller_frame}': {error}"
                )
            message.pose = do_transform_pose(pose, transform)
        self.target_publisher.publish(message)


# -- geometry ------------------------------------------------------------


def _distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _dot(a: Quaternion, b: Quaternion) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w


def _angle_between(a: Quaternion, b: Quaternion) -> float:
    """Smallest rotation taking one orientation to the other, in radians.

    Measured from the relative quaternion rather than as ``2 * acos(dot)``,
    which is the obvious form and is worst exactly where this is used most.
    ``acos`` loses half its significant digits as its argument approaches 1,
    so the obvious form reports tens of nanoradians between an orientation and
    itself -- and every convergence test in a settled move asks for the angle
    between two orientations that are nearly the same. ``atan2`` is well
    conditioned there, and gives a flat zero when the two agree.

    ``abs`` on the real part folds the double cover, so a quaternion and its
    negation read as the same orientation, which they are.
    """
    w = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z
    x = a.w * b.x - a.x * b.w - a.y * b.z + a.z * b.y
    y = a.w * b.y + a.x * b.z - a.y * b.w - a.z * b.x
    z = a.w * b.z - a.x * b.y + a.y * b.x - a.z * b.w
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w))


def _slerp(a: Quaternion, b: Quaternion, fraction: float) -> Quaternion:
    """Shortest-arc interpolation between two orientations.

    Component-wise interpolation would do here, since consecutive poses in a
    move differ by a few thousandths of a radian, but it would stop being true
    the moment a move turned the wrist any distance, and being wrong only for
    large inputs is the worst way to be wrong.
    """
    dot = _dot(a, b)
    # Negate one end if the pair is more than a quarter turn apart, so the
    # interpolation takes the short way round.
    if dot < 0.0:
        b = Quaternion(x=-b.x, y=-b.y, z=-b.z, w=-b.w)
        dot = -dot
    if dot > 0.9995:
        result = Quaternion(
            x=a.x + fraction * (b.x - a.x),
            y=a.y + fraction * (b.y - a.y),
            z=a.z + fraction * (b.z - a.z),
            w=a.w + fraction * (b.w - a.w),
        )
        return _normalised(result)
    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    first = math.sin((1.0 - fraction) * theta) / sin_theta
    second = math.sin(fraction * theta) / sin_theta
    return Quaternion(
        x=first * a.x + second * b.x,
        y=first * a.y + second * b.y,
        z=first * a.z + second * b.z,
        w=first * a.w + second * b.w,
    )


def _normalised(q: Quaternion) -> Quaternion:
    norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
    if norm == 0.0:
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return Quaternion(x=q.x / norm, y=q.y / norm, z=q.z / norm, w=q.w / norm)


def _interpolate(start: Pose, goal: Pose, fraction: float) -> Pose:
    """A pose ``fraction`` of the way along the line from one to the other."""
    pose = Pose()
    pose.position.x = start.position.x + fraction * (goal.position.x - start.position.x)
    pose.position.y = start.position.y + fraction * (goal.position.y - start.position.y)
    pose.position.z = start.position.z + fraction * (goal.position.z - start.position.z)
    pose.orientation = _slerp(start.orientation, goal.orientation, fraction)
    return pose


def _eased(fraction: float) -> float:
    """Raised cosine, so the target starts and stops at rest.

    A constant-speed target would ask for an instant change of velocity at
    both ends of the line, and the controller answers a step in the target
    with a lurch. This is smooth in acceleration, which a trapezoidal profile
    is not, and the arm follows it with no tuning; the cost is that the move
    takes pi/2 as long as a trapezoid at the same peak speed.
    """
    return 0.5 * (1.0 - math.cos(math.pi * fraction))


def _duration(
    distance: float, angle: float, speed: float, accel: float, rot_speed: float
) -> float:
    """How long to spend on a line, given what it may not exceed.

    The eased profile peaks at ``pi * distance / (2 * duration)`` in speed and
    ``pi**2 * distance / (2 * duration**2)`` in acceleration; this is the
    shortest duration that keeps both, and the rotation rate, inside their
    limits.
    """
    duration = 0.0
    if distance > 1e-9:
        duration = max(
            duration,
            math.pi * distance / (2.0 * speed),
            math.pi * math.sqrt(distance / (2.0 * accel)),
        )
    if angle > 1e-9:
        duration = max(duration, math.pi * angle / (2.0 * rot_speed))
    return duration


def _samples(start: Pose, goal: Pose, step: float) -> List[Pose]:
    """The line from one pose to the other, at ``step`` spacing, ends included."""
    distance = _distance(start.position, goal.position)
    count = max(1, int(math.ceil(distance / max(step, 1e-4))))
    return [_interpolate(start, goal, index / count) for index in range(count + 1)]

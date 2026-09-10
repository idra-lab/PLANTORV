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

"""Joint-space moves, sent straight to the trajectory controller.

This is what `home` uses instead of MoveIt, and the argument for it is short:
going to a known configuration is not a search. The start is on
``/joint_states``, the end is six numbers in planner.yaml, and the move is the
straight line between them in joint space. There is nothing to plan.

What that buys, beyond removing a dependency:

**The path is the same every time.** OMPL is a sampling planner; two runs from
the same state produce different routes, and the mitigation in moveit_client
was to ask for six plans and keep the shortest, which narrows the spread
without removing it. Here every joint moves monotonically from where it is to
where it is going, so the arm's route is a function of its start alone. An
operator can watch it once and know it.

**It can be read before it is run.** ``describe`` returns the trajectory as
text and ``move_to_configuration`` takes ``dry_run``, so the exact motion can
be printed and checked before anything is commanded. That is not possible with
a planner whose output changes on the next call.

**When it fails, it says what failed.** There is one action, one goal, and one
error code, rather than a planning request, a planning scene, a controller
manager and a trajectory execution manager each with their own opinion.

What it does not do is find a way round anything. If the straight line in
joint space sweeps a link through the table, the move is refused, not
rerouted -- that would be planning, and the point of this class is that there
is none. The refusal is the useful part: every configuration of the
trajectory is handed to a LinkGuard before anything is sent, so a bad move is
rejected while the arm is still stationary.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from plantorv_planner.errors import PlanningError

#: Joint motion below this, in radians, is encoder noise rather than travel.
TRAVEL_DEADBAND = 1e-5

#: Points emitted per trajectory. The controller interpolates between them, so
#: this only has to be fine enough that the profile below is what it follows
#: rather than a spline of its own choosing.
POINTS = 60


@dataclass
class JointMotion:
    """What a joint-space move cost."""

    #: Radians of joint travel, measured from /joint_states.
    path_length: float = 0.0
    #: Always zero. Nothing is planned.
    planning_time: float = 0.0
    #: Seconds the trajectory was built to take.
    duration: float = 0.0


class JointTrajectoryClient:
    """Moves the arm to a joint configuration, with no planner in between.

    Parameters
    ----------
    node : rclpy.node.Node
        The node the interfaces are created on.
    controller : str
        Trajectory controller name; its action is
        ``/<controller>/follow_joint_trajectory``.
    joint_names : sequence of str
        The arm's joints, in the order configurations are given in.
    switcher : ControllerSwitcher
        Used to activate the trajectory controller before commanding it.
    guard : LinkGuard, optional
        Tests every configuration of the trajectory against the furniture
        before it is sent. Left out, nothing is checked and the caller is
        warned once per move.
    callback_group : rclpy.callback_groups.CallbackGroup
        Must be reentrant: this blocks the caller while the controller runs.
    """

    def __init__(self, node, controller: str, joint_names: Sequence[str], switcher,
                 callback_group, guard=None):
        self.node = node
        self.guard = guard
        self.controller = controller
        self.joint_names = list(joint_names)
        self.switcher = switcher
        self.client = ActionClient(
            node,
            FollowJointTrajectory,
            f"/{controller}/follow_joint_trajectory",
            callback_group=callback_group,
        )
        self._positions: Optional[Dict[str, float]] = None
        self._travel = 0.0
        self._last: Optional[List[float]] = None
        node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10, callback_group=callback_group
        )

    # -- state -----------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        lookup = dict(zip(msg.name, msg.position))
        self._positions = lookup
        current = [lookup.get(name) for name in self.joint_names]
        if any(value is None for value in current):
            return
        current = [float(v) for v in current]
        if self._last is not None:
            step = math.sqrt(sum((a - b) ** 2 for a, b in zip(current, self._last)))
            if step > TRAVEL_DEADBAND:
                self._travel += step
        self._last = current

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until the controller's action server and joint states are up."""
        if not self.client.wait_for_server(timeout_sec=timeout):
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._last is not None:
                return True
            time.sleep(0.1)
        return False

    def current_positions(self) -> List[float]:
        """The arm's configuration now, in ``joint_names`` order."""
        if self._positions is None:
            raise PlanningError("no /joint_states yet; is the controller running?")
        missing = [n for n in self.joint_names if n not in self._positions]
        if missing:
            raise PlanningError(f"joints missing from /joint_states: {missing}")
        return [float(self._positions[n]) for n in self.joint_names]

    # -- the trajectory --------------------------------------------------

    def build(
        self, target: Sequence[float], speed: float, accel: float
    ) -> JointTrajectory:
        """The straight line in joint space from here to ``target``.

        Timed by the joint that has furthest to go, so every joint arrives
        together and none exceeds ``speed`` or ``accel``. The profile is a
        raised cosine, which starts and ends at rest; a constant-rate ramp
        would ask the controller for a step change in velocity at both ends.
        """
        if len(target) != len(self.joint_names):
            raise PlanningError(
                f"target has {len(target)} values, expected {len(self.joint_names)}"
            )
        start = self.current_positions()
        deltas = [float(t) - s for s, t in zip(start, target)]
        widest = max(abs(d) for d in deltas)

        duration = 0.0
        if widest > 1e-9:
            duration = max(
                math.pi * widest / (2.0 * speed),
                math.pi * math.sqrt(widest / (2.0 * accel)),
            )
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        if duration <= 0.0:
            # Already there. One point, so the controller has something valid.
            trajectory.points = [_point(start, [0.0] * len(start), 0.05)]
            return trajectory

        for index in range(1, POINTS + 1):
            fraction = index / POINTS
            eased = 0.5 * (1.0 - math.cos(math.pi * fraction))
            rate = (math.pi / (2.0 * duration)) * math.sin(math.pi * fraction)
            positions = [s + eased * d for s, d in zip(start, deltas)]
            velocities = [rate * d for d in deltas]
            trajectory.points.append(_point(positions, velocities, fraction * duration))
        # End exactly on target and at rest, whatever the arithmetic above did.
        trajectory.points[-1] = _point(list(target), [0.0] * len(target), duration)
        return trajectory

    def describe(self, target: Sequence[float], speed: float, accel: float) -> str:
        """The move as text, for reading before it is run."""
        start = self.current_positions()
        trajectory = self.build(target, speed, accel)
        duration = _seconds(trajectory.points[-1].time_from_start)
        lines = [
            f"joint move over {duration:.2f} s, {len(trajectory.points)} points, "
            f"limits {speed:.2f} rad/s and {accel:.2f} rad/s^2",
        ]
        for name, a, b in zip(self.joint_names, start, target):
            lines.append(
                f"  {name:22s} {math.degrees(a):+8.2f} deg -> {math.degrees(float(b)):+8.2f} deg"
                f"   ({math.degrees(float(b) - a):+7.2f})"
            )
        return "\n".join(lines)

    # -- execution -------------------------------------------------------

    def move_to_configuration(
        self,
        target: Sequence[float],
        speed: float,
        accel: float,
        timeout: float = 60.0,
        dry_run: bool = False,
        link_radius: float = 0.06,
    ) -> JointMotion:
        """Move to ``target`` and wait. Raises PlanningError if it cannot.

        With ``dry_run`` the trajectory is built and logged and nothing is
        sent, which is how to see what a move would do before doing it.
        """
        self.node.get_logger().info(self.describe(target, speed, accel))
        trajectory = self.build(target, speed, accel)
        duration = _seconds(trajectory.points[-1].time_from_start)

        # Checked before anything is sent, so a bad move is refused with the
        # arm still standing still. Every point, not just the ends: the line is
        # straight in joint space and curved in the world, so clear ends say
        # nothing about the middle.
        if self.guard is not None:
            self.guard.check_path([p.positions for p in trajectory.points], link_radius)
            self.node.get_logger().info(
                f"link guard: all {len(trajectory.points)} configurations clear "
                f"of the furniture by more than {link_radius * 1000:.0f} mm"
            )
        else:
            self.node.get_logger().warn(
                "no link guard; nothing is checking what the arm passes through"
            )

        if dry_run:
            self.node.get_logger().warn("dry_run: nothing sent to the controller")
            return JointMotion(duration=duration)

        self.switcher.ensure(self.controller)
        if not self.client.server_is_ready() and not self.client.wait_for_server(
            timeout_sec=10.0
        ):
            raise PlanningError(
                f"'{self.controller}' has no follow_joint_trajectory action; "
                f"is the controller loaded and active?"
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        before = self._travel

        handle = _await(self.client.send_goal_async(goal), 10.0)
        if handle is None:
            raise PlanningError("the trajectory controller did not answer the goal")
        if not handle.accepted:
            raise PlanningError("the trajectory controller rejected the trajectory")

        wrapper = _await(handle.get_result_async(), duration + timeout)
        if wrapper is None:
            handle.cancel_goal_async()
            raise PlanningError(
                f"the trajectory did not finish within {duration + timeout:.1f} s"
            )
        result = wrapper.result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise PlanningError(
                f"the trajectory controller reported error {result.error_code}"
                + (f": {result.error_string}" if result.error_string else "")
            )
        return JointMotion(path_length=self._travel - before, duration=duration)


def _point(positions, velocities, seconds: float) -> JointTrajectoryPoint:
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]
    point.velocities = [float(v) for v in velocities]
    point.time_from_start = _duration(seconds)
    return point


def _duration(seconds: float) -> DurationMsg:
    return DurationMsg(sec=int(seconds), nanosec=int(round((seconds % 1.0) * 1e9)))


def _seconds(duration: DurationMsg) -> float:
    return duration.sec + duration.nanosec / 1e9


def _await(future, timeout: float):
    """Wait for a future without blocking the executor that completes it."""
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    return future.result() if future.done() else None

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

"""The known actions, and what each one decomposes into.

The behaviour tree names an action and an object; everything about how the arm
gets there lives here. Four actions are known -- ``home``, ``move_to``,
``pick`` and ``place`` -- and the last two are the only interesting ones.

Every move the arm makes between objects is Cartesian, and the shape is always
the same: traverse along the transit plane to a point directly above the
target, descend straight onto it, and lift straight back to the plane. Nothing
between two objects is left to a sampling planner, so the path the arm takes is
the same on every run and the same on the real cell as in the simulator, which
is the point -- an operator can watch it once and know what it will do.

Those straight moves are driven by ``cartesian_motion_controller``, through
``CartesianClient``, rather than by MoveIt's ``/compute_cartesian_path``. The
reason is in that client's docstring: MoveIt solved inverse kinematics at each
waypoint independently and consecutive solutions could land in different arm
configurations, which is what once put the forearm through the stand. The
controller integrates a simulated twin of the arm forward from where the arm
is, so there is no second configuration for it to jump to. Set the
``motion_backend`` parameter to ``moveit`` to put the old path back and compare
the two.

That the arm cannot swing through the table falls out of the geometry rather
than out of collision checking: both ends of a traverse are on the transit
plane, and a straight line between two points of equal height stays at that
height. The plane itself is set high enough to clear the tray rims with a cube
hanging under the tool; see ``transit_height`` in planner_node.

**That geometry is now the only thing keeping the arm out of the desk.** The
controller checks nothing, so ``_traverse`` no longer trusts a planner to
refuse a line that starts below the plane: it lifts the tool onto the plane
first, and every commanded tool path is checked against a height floor and a
reach annulus before it is sent. None of that watches the elbow, which is the
part that has been through the table before.

``home`` is the exception and stays a joint-space move: it is a fixed, known
configuration rather than a point in the workspace, and it leaves the tool
above the transit plane, so the invariant still holds afterwards. It is also
the only action that needs the trajectory controller, so it is the only one
that pays for a controller switch.

It no longer goes through MoveIt. Going to a known configuration is not a
search -- the start is on /joint_states, the end is six numbers, and the move
is the straight line between them -- so it is sent to the trajectory
controller directly; see joint_client.py. Nothing here plans, and nothing here
checks collisions either.

Adding an action means adding a method and an entry in ``DISPATCH``. The
behaviour tree needs no change to call it.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from geometry_msgs.msg import Pose

from plantorv_planner.cartesian_client import CartesianClient
from plantorv_planner.errors import PlanningError
from plantorv_planner.geometry import pose_above, square_symmetric_yaw, yaw_of
from plantorv_planner.moveit_client import MoveItClient, Plan


@dataclass
class ActionOutcome:
    """What an action did."""

    message: str
    path_length: float = 0.0
    planning_time: float = 0.0

    def __add__(self, other: "ActionOutcome") -> "ActionOutcome":
        """Accumulate the cost of the steps an action is made of."""
        return ActionOutcome(
            message=self.message or other.message,
            path_length=self.path_length + other.path_length,
            planning_time=self.planning_time + other.planning_time,
        )


@dataclass
class Held:
    """The object currently in the fake gripper."""

    name: str
    height: float


class ActionLibrary:
    """Turns a named action into planned, executed motion.

    Parameters
    ----------
    node : rclpy.node.Node
        Used for parameters, logging, and the service calls to the world model
        and the fake grasp.
    moveit : MoveItClient or None
        Kept only for the ``moveit`` motion_backend and the straight-lift
        fallback in ``pick``. None when the planner runs without MoveIt, which
        is the default.
    joints : JointTrajectoryClient
        The joint-space back end: ``home``. Sends a trajectory straight to the
        controller, with nothing planning anything.
    cartesian : CartesianClient
        The straight-line back end, which is most of the motion.
    params : dict
        The contents of ``config/planner.yaml``, already read off the node.
    look_up : Callable[[str], object]
        Resolves an object name to the world model's answer.
    grasp : Callable[[str, bool], None]
        Attaches or releases an object; raises on failure.
    gripper : Callable[[str], str] or None
        Sends one command to the gripper and returns what it said. None when
        no gripper is configured, and then the gripper actions refuse rather
        than pretend.
    """

    def __init__(
        self,
        node,
        moveit: Optional[MoveItClient],
        cartesian: CartesianClient,
        joints,
        params: Dict,
        look_up: Callable,
        grasp: Callable,
        gripper: Optional[Callable] = None,
    ):
        self.node = node
        self.moveit = moveit
        self.cartesian = cartesian
        self.cartesian.link_radius = float(params["link_radius"])
        self.joints = joints
        self.params = params
        self.look_up = look_up
        self.grasp = grasp
        self.gripper = gripper
        self.held: Optional[Held] = None

    # -- the actions -----------------------------------------------------

    def execute(
        self, action_name: str, target: str, pose: Optional[Pose], report: Callable
    ) -> ActionOutcome:
        """Run one known action. Raises PlanningError if it cannot."""
        handler = self.DISPATCH.get(action_name)
        if handler is None:
            raise PlanningError(
                f"unknown action '{action_name}'; known actions are "
                f"{', '.join(sorted(self.DISPATCH))}"
            )
        return handler(self, target, pose, report)

    def _home(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        """Straight to the taught configuration, with no planner involved.

        The only action that is a joint-space move, and the only one that does
        not need a target worked out from the scene: home is six numbers. It is
        the straight line to them, timed so no joint exceeds its limit, sent to
        the trajectory controller as one goal.
        """
        report("home", 0.0)
        motion = self.joints.move_to_configuration(
            self.params["home_positions"],
            speed=self.params["joint_speed"] * self.params["velocity_scaling"],
            accel=self.params["joint_accel"] * self.params["acceleration_scaling"],
            dry_run=bool(self.params["dry_run"]),
            link_radius=self.params["link_radius"],
        )
        report("home", 1.0)
        return ActionOutcome("at home", motion.path_length, motion.planning_time)

    def _move_to(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        """Put the tool over something, at a height.

        Three ways to say where:

            a target alone, and the height is the object's approach height,
            worked out from the scene;

            a pose alone, and it is taken as given;

            both, and the target names the column while the pose gives only
            the height. That last one is what a tree uses to stand over a
            block at a height chosen by hand rather than derived -- useful
            while the gripper's length is still being established, since the
            derived height assumes tool0 is the gripping point and it is not.

        The motion is the same in every case, and it is the shape the whole
        library uses: up to the transit plane if not already on it, across at
        that height, then straight down. Never diagonally towards the table.
        """
        goal = pose if not target else self._hover_pose(target)
        if goal is None:
            raise PlanningError("move_to needs either a target object or a pose")
        if target and pose is not None and pose.position.z > 0.0:
            goal = pose_above(
                goal.position.x, goal.position.y, pose.position.z, yaw_of(goal.orientation)
            )
        report("move", 0.0)
        # Same shape as pick and place: along the plane, then straight down.
        outcome = self._traverse(goal) + self._go_straight(
            [goal], tolerance=self.params["cartesian_transit_tolerance"]
        )
        report("move", 1.0)
        where = target or "the given pose"
        return ActionOutcome(f"tool at {where}", outcome.path_length, outcome.planning_time)

    # def _pick(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
    #     if not target:
    #         raise PlanningError("pick needs the name of an object")
    #     if self.held is not None:
    #         raise PlanningError(f"already holding {self.held.name}")

    #     report("locate", 0.0)
    #     obj = self._require(target)
    #     grasp_pose = self._grasp_pose(obj)
    #     above = self._transit_over(grasp_pose)

    #     report("approach", 0.2)
    #     total = self._traverse(grasp_pose)

    #     report("descend", 0.4)
    #     # Tight: the tool is about to take hold of the cube.
    #     total += self._go_straight([grasp_pose])

    #     report("grasp", 0.6)
    #     self.grasp(target, True)
    #     self.held = Held(name=target, height=float(obj.dimensions.z))

    #     report("retreat", 0.8)
    #     try:
    #         total += self._go_straight(
    #             [above], tolerance=self.params["cartesian_transit_tolerance"]
    #         )
    #     except PlanningError:
    #         # The object is in the hand either way; leaving the arm down there
    #         # would be worse than a joint-space lift.
    #         self.node.get_logger().warn("straight retreat failed, lifting through joint space")
    #         total += self._go_to(above)

    #     report("done", 1.0)
    #     return ActionOutcome(f"picked {target}", total.path_length, total.planning_time)

    # def _place(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
    #     if self.held is None:
    #         raise PlanningError("nothing is held, so there is nothing to place")
    #     if not target and pose is None:
    #         raise PlanningError("place needs a target container or a pose")

    #     report("locate", 0.0)
    #     release = (
    #         pose if pose is not None and not target else self._release_pose(self._require(target))
    #     )
    #     above = self._transit_over(release)

    #     report("approach", 0.2)
    #     total = self._traverse(release)

    #     report("descend", 0.4)
    #     # Tight: the cube is let go from here, and where it lands follows.
    #     total += self._go_straight([release])

    #     report("release", 0.6)
    #     placed = self.held.name
    #     self.grasp(placed, False)
    #     self.held = None

    #     report("retreat", 0.8)
    #     total += self._go_straight(
    #         [above], tolerance=self.params["cartesian_transit_tolerance"]
    #     )

    #     report("done", 1.0)
    #     return ActionOutcome(f"placed {placed} in {target}", total.path_length, total.planning_time)

    def _open_gripper(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        """Open the gripper, and wait for it to say it has."""
        return self._gripper_command("open", report)

    def _close_gripper(self, target: str, pose: Optional[Pose], report: Callable) -> ActionOutcome:
        """Close the gripper, and wait for it to say it has.

        Closing is not the same as grasping. The driver reports that the
        gripper moved, not that anything is held; it has no force feedback
        wired up here and this action does not check whether a block is in it.
        A tree that closes and then lifts is assuming, not verifying.
        """
        return self._gripper_command("close", report)

    def _gripper_command(self, command: str, report: Callable) -> ActionOutcome:
        """Send one command and wait. The arm does not move."""
        if self.gripper is None:
            raise PlanningError(
                "no gripper is configured; launch against the real arm with "
                "gripper:=true, or set use_gripper in planner.yaml"
            )
        report(command, 0.0)
        said = self.gripper(command)
        report(command, 1.0)
        return ActionOutcome(said or f"gripper {command}")

    DISPATCH: Dict[str, Callable] = {
        "home": _home,
        "move_to": _move_to,
        # "pick": _pick,
        # "place": _place,
        "open_gripper": _open_gripper,
        "close_gripper": _close_gripper,
    }

    # -- poses -----------------------------------------------------------

    def _require(self, name: str):
        """Ask the world model where something is."""
        found = self.look_up(name)
        if found is None or not found.found:
            raise PlanningError(f"the world model does not know an object called '{name}'")
        return found

    def _grasp_pose(self, obj) -> Pose:
        """Where the tool sits to grasp an object from above.

        The tool frame is put just over the object's top face, so that the
        object hangs under it once attached, and turned to line up with it --
        folded to the nearest quarter turn, since the objects are cubes.
        """
        position = obj.pose.pose.position
        yaw = square_symmetric_yaw(yaw_of(obj.pose.pose.orientation))
        top = position.z + obj.dimensions.z / 2.0
        return pose_above(position.x, position.y, top + self.params["tool_gap"], yaw)

    def _hover_pose(self, name: str) -> Pose:
        """A pose over an object, at the height the approach starts from."""
        obj = self._require(name)
        position = obj.pose.pose.position
        top = position.z + obj.dimensions.z / 2.0
        height = top + self.params["tool_gap"] + self.params["approach_distance"]
        return pose_above(position.x, position.y, height)

    def _release_pose(self, container) -> Pose:
        """Where the tool sits to drop the held object into a container.

        The object is let go just above the container's rim rather than lowered
        into it: the fake grasp has no compliance, and the last centimetres of
        a 10 cm cube into a tray of its own size are exactly where a rigid
        placement jams.
        """
        position = container.pose.pose.position
        rim = position.z + container.dimensions.z / 2.0
        height = rim + self.params["drop_gap"] + self.params["tool_gap"] + self.held.height
        return pose_above(position.x, position.y, height)

    # -- motion ----------------------------------------------------------

    def _plan_to_joints(self, joint_values: List[float]) -> Plan:
        return self.moveit.plan_to_joint_goal(
            joint_names=self.params["joint_names"],
            joint_values=joint_values,
            planner_id=self.params["planner_id"],
            attempts=self.params["planning_attempts"],
            allowed_time=self.params["allowed_planning_time"],
            velocity_scaling=self.params["velocity_scaling"],
            acceleration_scaling=self.params["acceleration_scaling"],
        )

    def _plan_to_pose(self, pose: Pose) -> Plan:
        joint_values = self.moveit.inverse_kinematics(pose, self.params["joint_names"])
        return self._plan_to_joints(joint_values)

    def _transit_over(self, pose: Pose) -> Pose:
        """The point on the transit plane directly above ``pose``."""
        return pose_above(
            pose.position.x,
            pose.position.y,
            self.params["transit_height"],
            yaw_of(pose.orientation),
        )

    def _traverse(self, goal: Pose) -> ActionOutcome:
        """Move along the transit plane to the point above ``goal``.

        Both ends are on the plane, so the straight line between them cannot
        descend, and this is the move that carries the arm across the table.

        Every action arranges to leave the tool on the plane, so the caller is
        normally there already. When it is not -- the first action after the
        arm has been left somewhere odd -- the tool is lifted straight up onto
        the plane first, and only then does the traverse run.

        That lift is the difference from the MoveIt version, and it is the one
        place where losing collision checking would have cost something. There,
        a traverse starting from down beside a cube was refused, and the refusal
        was the signal to fall back to a joint-space plan. Nothing refuses it
        now: the controller would drag the tool sideways at cube height,
        through every cube between here and the target. So the precondition is
        established rather than detected.
        """
        above = self._transit_over(goal)
        outcome = ActionOutcome("")
        here = self.cartesian.current_pose()
        plane = float(self.params["transit_height"])
        if here.position.z < plane - self.params["cartesian_pos_tolerance"]:
            self.node.get_logger().info(
                f"the tool is at z = {here.position.z:.3f}, below the transit plane at "
                f"{plane:.3f}; lifting onto the plane before traversing"
            )
            outcome += self._go_straight(
                [
                    pose_above(
                        here.position.x,
                        here.position.y,
                        plane,
                        yaw_of(here.orientation),
                    )
                ],
                tolerance=self.params["cartesian_transit_tolerance"],
            )
        return outcome + self._go_straight(
            [above], tolerance=self.params["cartesian_transit_tolerance"]
        )

    def _go_to(self, pose: Pose) -> ActionOutcome:
        """Plan freely to a pose and execute, through MoveIt.

        The last resort, and the only move whose path is not predictable. It
        exists because an arm that has failed a straight lift while holding a
        cube is in a worse place than one that took an odd route out of it.

        Unavailable without MoveIt, and deliberately not replaced: reaching a
        pose from an arbitrary state is the one thing here that genuinely is a
        planning problem, and a bad substitute for it would be worse than
        refusing. The caller is told so and the goal aborts with the arm where
        it is.
        """
        if self.moveit is None:
            raise PlanningError(
                "the straight move failed and there is no free-space fallback "
                "without MoveIt; the arm is holding still. Send `home` to "
                "recover, which is a joint-space move and does not need one"
            )
        plan = self._plan_to_pose(pose)
        self.moveit.execute(plan)
        return ActionOutcome("", plan.path_length, plan.planning_time)

    def _go_straight(
        self, waypoints: List[Pose], tolerance: Optional[float] = None
    ) -> ActionOutcome:
        """Move the tool in a straight line through ``waypoints``.

        The commanded path is checked before any of it is sent, because the
        controller will not check it and will not refuse it.

        ``tolerance`` is how near the tool has to get before the move is
        finished, and it is worth being deliberate about. The controller
        converges on a target asymptotically, so the last millimetre costs far
        more time than the first centimetre -- most of a move's wall time can
        go into closing a gap that does not matter. It matters when the tool is
        about to take hold of something or let go of it, and it does not matter
        at all at the end of a traverse, which finishes in open air on the
        transit plane with the next descent aimed at an absolute pose that
        corrects any error left over. So free-space moves pass
        ``transit_tolerance`` and only the two moves that end at an object use
        the tight one, which is the default here.
        """
        if self.params["motion_backend"] == "moveit":
            if self.moveit is None:
                raise PlanningError(
                    "motion_backend is 'moveit' but the planner was started without it; "
                    "launch with use_moveit:=true or set motion_backend to 'cartesian'"
                )
            return self._go_straight_via_moveit(waypoints)

        here = self.cartesian.current_pose()
        self.cartesian.check_path(
            here,
            waypoints,
            min_z=self.params["workspace_min_z"],
            min_reach=self.params["workspace_min_reach"],
            max_reach=self.params["workspace_max_reach"],
            step=self.params["cartesian_step"],
        )
        # The scaling factors are what a per-goal override in ExecuteAction
        # sets, so they keep meaning the same thing here as they did to MoveIt:
        # a fraction of the fastest this cell is allowed to move.
        motion = self.cartesian.move_linear(
            waypoints,
            speed=self.params["cartesian_speed"] * self.params["velocity_scaling"],
            accel=self.params["cartesian_accel"] * self.params["acceleration_scaling"],
            rot_speed=self.params["cartesian_rot_speed"] * self.params["velocity_scaling"],
            rate=self.params["cartesian_rate"],
            pos_tolerance=(
                self.params["cartesian_pos_tolerance"] if tolerance is None else tolerance
            ),
            rot_tolerance=self.params["cartesian_rot_tolerance"],
            timeout=self.params["cartesian_timeout"],
            stall_time=self.params["cartesian_stall_time"],
            start=here,
        )
        return ActionOutcome("", motion.path_length, motion.planning_time)

    def _go_straight_via_moveit(self, waypoints: List[Pose]) -> ActionOutcome:
        """The old straight move, kept to compare the two back ends."""
        plan = self.moveit.plan_cartesian(
            waypoints,
            velocity_scaling=self.params["velocity_scaling"],
            acceleration_scaling=self.params["acceleration_scaling"],
            step=self.params["cartesian_step"],
            min_fraction=self.params["cartesian_min_fraction"],
        )
        self.moveit.execute(plan)
        return ActionOutcome("", plan.path_length, plan.planning_time)

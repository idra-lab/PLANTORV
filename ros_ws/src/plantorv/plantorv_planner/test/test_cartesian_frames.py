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

"""Which frame each part of the Cartesian client asks TF for.

Three frames meet in this client and two of them are easy to swap. The
controller accepts targets only in its own ``robot_base_link``, which on this
cell is ``world`` so that no target has to be transformed. The reach guard has
to measure from the link the arm is bolted to. Point the guard at the
controller's frame and it measures from the middle of the desk, where every
pose in the cell is over a metre out and the first move of every run is
refused for a distance that looks nothing like a UR3.

So these tests are about frame names rather than numbers.
"""

import math

import pytest
from geometry_msgs.msg import Pose, TransformStamped

from plantorv_planner.cartesian_client import CartesianClient
from plantorv_planner.errors import PlanningError
from plantorv_planner.geometry import pose_above

PLANNING = "world"
CONTROLLER = "world"
ARM_BASE = "base_link"
TOOL = "tool0"

# Where the workcell puts the arm: middle of the desk's long side, on a stand.
BASE_IN_WORLD = (0.0, 0.45, 1.20)


class FakeLogger:
    def info(self, *args):
        pass

    def error(self, *args):
        pass


class FakeClock:
    def now(self):
        return self

    def to_msg(self):
        from builtin_interfaces.msg import Time

        return Time()


class FakeNode:
    def create_publisher(self, *args, **kwargs):
        return FakePublisher()

    def create_subscription(self, *args, **kwargs):
        return None

    def get_clock(self):
        return FakeClock()

    def get_logger(self):
        return FakeLogger()


class FakePublisher:
    def __init__(self):
        self.sent = []

    def publish(self, message):
        self.sent.append(message)


class FakeBuffer:
    """Records what was asked for, and answers from a fixed table."""

    def __init__(self):
        self.asked = []
        self.known = {
            (PLANNING, ARM_BASE): BASE_IN_WORLD,
            (PLANNING, TOOL): (0.1124, 0.1514, 1.5136),
            (PLANNING, PLANNING): (0.0, 0.0, 0.0),
        }

    def lookup_transform(self, target, source, when):
        self.asked.append((target, source))
        if (target, source) not in self.known:
            from tf2_ros import LookupException

            raise LookupException(f"no {target} -> {source}")
        x, y, z = self.known[(target, source)]
        transform = TransformStamped()
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.w = 1.0
        return transform


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "plantorv_planner.cartesian_client.TransformListener",
        lambda buffer, node, **kwargs: None,
    )
    made = CartesianClient(
        FakeNode(),
        controller="cartesian_motion_controller",
        tool_link=TOOL,
        planning_frame=PLANNING,
        controller_frame=CONTROLLER,
        arm_base_frame=ARM_BASE,
        joint_names=["a", "b"],
        switcher=None,
        callback_group=None,
    )
    made.buffer = FakeBuffer()
    return made


def test_reach_is_measured_from_the_arm_base_not_the_controller_frame(client):
    """The bug this file exists for.

    With the controller working in `world`, asking TF for the controller frame
    returns the identity and reach comes out measured from the desk's centre.
    """
    client.check_path(
        pose_above(0.0, 0.36, 1.32),
        [pose_above(-0.36, 0.36, 1.32)],
        min_z=1.01,
        min_reach=0.10,
        max_reach=0.485,
        step=0.005,
    )
    assert (PLANNING, ARM_BASE) in client.buffer.asked
    assert (PLANNING, CONTROLLER) not in client.buffer.asked


def test_the_home_pose_is_inside_the_guard(client):
    """Measured from base_link it is 0.447 m out; from the world origin, 1.525."""
    home = (0.1124, 0.1514, 1.5136)
    assert math.dist(BASE_IN_WORLD, home) == pytest.approx(0.4474, abs=1e-3)
    assert math.dist((0.0, 0.0, 0.0), home) == pytest.approx(1.5253, abs=1e-3)
    # And the client agrees, which it did not when it used the controller frame.
    client.check_path(
        Pose(position=type(Pose().position)(x=home[0], y=home[1], z=home[2])),
        [pose_above(-0.36, 0.36, 1.32)],
        min_z=1.01,
        min_reach=0.10,
        max_reach=0.485,
        step=0.005,
    )


def test_a_pose_genuinely_out_of_reach_is_still_refused(client):
    with pytest.raises(PlanningError, match="from the arm's base"):
        client.check_path(
            pose_above(0.0, 0.36, 1.32),
            [pose_above(0.9, 0.36, 1.32)],
            min_z=1.01,
            min_reach=0.10,
            max_reach=0.485,
            step=0.005,
        )


def test_a_path_driven_into_the_desk_is_refused(client):
    # Over cube_1 rather than over the stand: a descent on the centreline
    # passes 0.099 m from base_link and is refused for reach before it ever
    # gets low enough to be refused for height. Both refusals are true, but
    # this test is about the floor.
    with pytest.raises(PlanningError, match="below the floor"):
        client.check_path(
            pose_above(-0.36, 0.36, 1.32),
            [pose_above(-0.36, 0.36, 0.95)],
            min_z=1.01,
            min_reach=0.10,
            max_reach=0.485,
            step=0.005,
        )


def test_the_tool_pose_comes_from_the_tool_link(client):
    pose = client.current_pose()
    assert (PLANNING, TOOL) in client.buffer.asked
    assert pose.position.z == pytest.approx(1.5136)


def test_a_target_in_the_controller_frame_is_published_untransformed(client):
    # planning_frame == controller_frame here, so there is nothing to transform
    # and TF must not be consulted for it.
    before = list(client.buffer.asked)
    client._publish(pose_above(0.1, 0.2, 1.30))
    assert client.buffer.asked == before
    sent = client.target_publisher.sent[-1]
    assert sent.header.frame_id == CONTROLLER
    assert sent.pose.position.z == pytest.approx(1.30)


def test_a_missing_arm_base_transform_says_which_frame_is_missing(client):
    client.buffer.known.pop((PLANNING, ARM_BASE))
    with pytest.raises(PlanningError, match=ARM_BASE):
        client.check_path(
            pose_above(0.0, 0.36, 1.32),
            [pose_above(-0.36, 0.36, 1.32)],
            min_z=1.01,
            min_reach=0.10,
            max_reach=0.485,
            step=0.005,
        )

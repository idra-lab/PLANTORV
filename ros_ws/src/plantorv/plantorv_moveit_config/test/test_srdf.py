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

"""The SRDF, checked without starting move_group.

move_group reads this file at launch; a mistake in it costs a full bring-up to
find. These tests cost nothing and catch the two kinds that actually happen:
XML that does not parse, and a group or a joint name that has drifted away from
what the planner and the controllers use.
"""

import os
import xml.etree.ElementTree as ElementTree

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRDF = os.path.join(HERE, "config", "ur3_workcell.srdf")

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@pytest.fixture
def srdf():
    return ElementTree.parse(SRDF).getroot()


def test_it_is_well_formed_xml(srdf):
    assert srdf.tag == "robot"


def test_the_planning_group_is_the_one_everything_else_names(srdf):
    groups = {group.get("name") for group in srdf.findall("group")}
    assert "ur_manipulator" in groups


def test_the_group_runs_from_the_base_to_the_tool(srdf):
    chain = srdf.find("group[@name=\'ur_manipulator\']/chain")
    assert chain.get("base_link") == "base_link"
    # The frame the fake grasp attaches objects to.
    assert chain.get("tip_link") == "tool0"


def test_home_names_every_arm_joint(srdf):
    state = srdf.find("group_state[@name=\'home\']")
    assert state is not None
    named = [joint.get("name") for joint in state.findall("joint")]
    assert named == ARM_JOINTS


def test_home_matches_the_pose_the_arm_starts_in(srdf):
    """The SRDF and plantorv_sim's initial_positions.yaml state the same pose.

    They are read by different programs and neither would complain about the
    other, so the arm would simply start somewhere other than `home`.
    """
    import yaml

    state = srdf.find("group_state[@name=\'home\']")
    from_srdf = {joint.get("name"): float(joint.get("value")) for joint in state.findall("joint")}

    initial = os.path.join(
        os.path.dirname(HERE), "plantorv_sim", "config", "initial_positions.yaml"
    )
    with open(initial, encoding="utf-8") as handle:
        from_yaml = yaml.safe_load(handle)

    for joint in ARM_JOINTS:
        assert from_srdf[joint] == pytest.approx(from_yaml[joint], abs=1e-6)


def test_the_mounting_stack_is_excluded_from_collision_checking(srdf):
    """The desk, the stand and the base touch by construction.

    Everything else about the desk stays checked: the arm is 20 cm above a
    surface it can drive into, and that is the point of modelling it.
    """
    disabled = {
        frozenset((pair.get("link1"), pair.get("link2")))
        for pair in srdf.findall("disable_collisions")
    }
    assert frozenset(("table_link", "stand_link")) in disabled
    assert frozenset(("stand_link", "base_link_inertia")) in disabled
    # The forearm over the desk is exactly the collision worth keeping.
    assert frozenset(("table_link", "forearm_link")) not in disabled


def test_the_robot_name_matches_the_description(srdf):
    """MoveIt pairs the SRDF with the URDF by name, and says little if it cannot."""
    urdf = os.path.join(
        os.path.dirname(HERE), "plantorv_sim", "urdf", "ur3_workcell.urdf.xacro"
    )
    described = ElementTree.parse(urdf).getroot().get("name")
    assert srdf.get("name") == described

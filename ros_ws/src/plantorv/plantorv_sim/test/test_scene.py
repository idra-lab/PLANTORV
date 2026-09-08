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

"""The workcell layout, checked against the arm that has to reach it.

The numbers in scene.yaml are easy to change and hard to sanity-check by eye,
and the failure mode -- a tray 5 cm outside a UR3's 500 mm reach -- shows up as
an unexplained IK failure much later. These tests are the check.
"""

import math
import os
import pathlib
import xml.etree.ElementTree as ElementTree

import pytest

from plantorv_sim.scene import TYPE_CUBE, TYPE_TRAY, Scene

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_FILE = os.path.join(HERE, "config", "scene.yaml")

# Reach of a UR3, wrist included, from the shoulder axis.
UR3_REACH = 0.5


@pytest.fixture
def scene():
    return Scene.from_file(SCENE_FILE)


def test_the_desk_is_the_size_it_is_meant_to_be(scene):
    table = scene.get("table")
    assert table.dimensions == (2.0, 1.0, 1.0)
    assert scene.table_height == pytest.approx(1.0)


def test_the_robot_stands_on_the_desk_at_the_middle_of_its_long_side(scene):
    stand = scene.get("robot_stand")
    x, y = scene.mount_xy
    assert x == pytest.approx(0.0)
    # On the long side, which runs along x, so at the +y edge or near it.
    assert 0.0 < y <= scene.get("table").dimensions[1] / 2.0
    # It rests on the desk top, and the arm on the stand.
    assert stand.position[2] - stand.dimensions[2] / 2.0 == pytest.approx(scene.table_height)
    assert scene.mount_height == pytest.approx(stand.top_z)
    assert scene.mount_height - scene.table_height == pytest.approx(0.20)


def test_the_cubes_sit_on_the_desk(scene):
    cubes = scene.of_type(TYPE_CUBE)
    assert cubes
    for cube in cubes:
        assert cube.dimensions == (0.10, 0.10, 0.10)
        assert cube.position[2] - cube.dimensions[2] / 2.0 == pytest.approx(scene.table_height)


def test_there_are_two_trays_in_the_middle_of_the_desk(scene):
    trays = scene.of_type(TYPE_TRAY)
    assert len(trays) == 2
    for tray in trays:
        assert abs(tray.position[0]) < 0.5
        assert abs(tray.position[1]) < 0.25
        assert tray.position[2] - tray.dimensions[2] / 2.0 == pytest.approx(scene.table_height)


@pytest.mark.parametrize("kind", [TYPE_CUBE, TYPE_TRAY])
def test_everything_the_arm_works_on_is_within_its_reach(scene, kind):
    base_x, base_y = scene.mount_xy
    base_z = scene.mount_height
    for obj in scene.of_type(kind):
        # The tool has to get above the object, not merely to its centre.
        target_z = obj.top_z
        distance = math.dist((base_x, base_y, base_z), (obj.position[0], obj.position[1], target_z))
        assert distance < UR3_REACH, f"{obj.name} is {distance:.3f} m from the base"


def test_nothing_overlaps_anything_else(scene):
    objects = [o for o in scene if o.name != "table"]
    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            overlaps = all(
                abs(first.position[axis] - second.position[axis])
                < (first.dimensions[axis] + second.dimensions[axis]) / 2.0
                for axis in range(3)
            )
            assert not overlaps, f"{first.name} and {second.name} occupy the same space"


def test_the_generated_world_is_valid_sdf_and_holds_the_trays(scene):
    root = ElementTree.fromstring(scene.world_sdf())
    models = {model.get("name") for model in root.iter("model")}
    assert {tray.name for tray in scene.of_type(TYPE_TRAY)} <= models
    # The cubes are spawned at run time and the desk comes with the robot, so
    # neither belongs in the world file.
    assert not {cube.name for cube in scene.of_type(TYPE_CUBE)} & models
    assert "table" not in models


def test_the_committed_world_matches_the_scene(scene):
    """The world file is generated; this is what catches it going stale."""
    with open(os.path.join(HERE, "worlds", "plantorv_table.world"), encoding="utf-8") as handle:
        assert handle.read() == scene.world_sdf()


@pytest.mark.parametrize(
    "pattern", ["urdf/*.xacro", "worlds/*.world", "package.xml"]
)
def test_the_xml_files_are_well_formed(pattern):
    """Parse the descriptions here, rather than finding out at launch.

    xacro reads the file as XML before it substitutes anything, so an XML
    mistake in a comment -- a pair of hyphens, say -- fails the launch with a
    line number and nothing else. This is the same check, one test run earlier.
    """
    paths = sorted(pathlib.Path(HERE).glob(pattern))
    assert paths, f"nothing matches {pattern}"
    for path in paths:
        ElementTree.parse(path)


def test_the_root_element_uses_no_substitution_arguments():
    """xacro cannot resolve $(arg ...) on the element that declares the args.

    It evaluates the root element's attributes before it walks the children,
    so <robot name="$(arg name)"> fails with "Undefined substitution argument"
    for every caller that does not pass name:= on the command line. The fix is
    a literal, and this is the test that keeps it one.
    """
    for path in sorted(pathlib.Path(HERE).glob("urdf/*.xacro")):
        root = ElementTree.parse(path).getroot()
        for attribute, value in root.attrib.items():
            assert "$(arg" not in value, f"{path.name}: {attribute}={value}"


def test_a_spawnable_cube_is_valid_sdf_with_mass(scene):
    cube = scene.of_type(TYPE_CUBE)[0]
    root = ElementTree.fromstring(scene.model_sdf(cube))
    model = root.find("model")
    assert model.get("name") == cube.name
    assert model.find("static").text == "false"
    assert model.find("link/inertial/mass") is not None

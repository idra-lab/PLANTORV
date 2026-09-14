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

# How far the shoulder axis sits above base_link on a UR3, its d1. The
# reach above is a radius about that axis, so anything compared against
# it has to be measured from there: base_link is 0.152 m lower, and
# measuring from it charges height against the horizontal budget. The
# planner's guard makes the same distinction, through arm_base_frame.
SHOULDER_HEIGHT = 0.1519

# The planner's own config, one package over. Read rather than copied, because
# the numbers below are the ones that have to agree with this layout, and a
# copy of them here would agree with it for ever.
PLANNER_YAML = os.path.join(
    os.path.dirname(HERE), "plantorv_planner", "config", "planner.yaml"
)


@pytest.fixture
def scene():
    return Scene.from_file(SCENE_FILE)


def test_the_table_is_the_size_it_is_meant_to_be(scene):
    table = scene.get("table")
    assert table.dimensions == (1.60, 0.80, 0.75)
    assert scene.table_height == pytest.approx(0.75)


def test_the_robot_is_the_origin_and_the_table_is_in_front_of_it(scene):
    # The frame is pinned to the robot, not to the table, so this is what
    # everything else in the cell is measured from.
    x, y = scene.mount_xy
    assert (x, y) == (pytest.approx(0.0), pytest.approx(0.0))
    # Middle of the long side, which runs along x.
    assert scene.table_xy[0] == pytest.approx(0.0)
    # And in front, at +y, entirely so.
    assert scene.table_near_edge > 0.0


def test_the_pedestal_stands_on_the_floor_and_carries_the_arm(scene):
    stand = scene.get("robot_stand")
    # On the floor, not on the table: the two are independent.
    assert stand.position[2] - stand.dimensions[2] / 2.0 == pytest.approx(0.0)
    assert stand.dimensions[2] == pytest.approx(0.885)
    assert scene.mount_height == pytest.approx(stand.top_z)
    assert scene.mount_height == pytest.approx(0.885)


def test_the_table_is_pushed_up_to_the_pedestal_and_does_not_overlap_it(scene):
    stand = scene.get("robot_stand")
    behind_stand = scene.mount_xy[1] + stand.dimensions[1] / 2.0
    assert scene.table_near_edge >= behind_stand, "the table cuts into the pedestal"
    # And not left standing off in the distance either.
    assert scene.table_near_edge - behind_stand < 0.05


def test_the_arm_reaches_over_the_table_rather_than_standing_on_it(scene):
    """base_link sits above the working surface, which is what allows a
    top-down grasp at all.

    No upper bound on the clearance, because more of it is not a problem in
    itself -- but it is not free either. Every centimetre the base rises is a
    centimetre further the arm has to reach down, and the reach it costs is
    counted against the same 0.5 m budget as the reach across the table. At
    13.5 cm -- the stand was measured with a tape at 0.885, not the 0.975
    once assumed -- the release pose over a tray is 0.4587 m out, which is
    the closest reachable pose in this cell comes to the limit.
    """
    clearance = scene.mount_height - scene.table_height
    assert clearance > 0.0
    assert clearance == pytest.approx(0.135)


def test_the_blocks_sit_on_the_table(scene):
    cubes = scene.of_type(TYPE_CUBE)
    assert cubes
    for cube in cubes:
        # 3 x 3 x 6 cm: taller than they are wide, so not cubes despite the
        # type name. Standing, not lying down.
        assert cube.dimensions == (0.03, 0.03, 0.06)
        assert cube.position[2] - cube.dimensions[2] / 2.0 == pytest.approx(scene.table_height)


def test_there_are_two_trays_on_the_table_in_front_of_the_robot(scene):
    trays = scene.of_type(TYPE_TRAY)
    assert len(trays) == 2
    for tray in trays:
        assert abs(tray.position[0]) < 0.5
        # In front of the robot and on the table, not behind or beside it.
        assert tray.position[1] > scene.table_near_edge
        assert tray.position[2] - tray.dimensions[2] / 2.0 == pytest.approx(scene.table_height)


def test_the_trays_are_the_size_they_were_measured_to_be(scene):
    """27.5 x 40 x 2.5 cm outside, with the 40 cm side along the table."""
    for tray in scene.of_type(TYPE_TRAY):
        width, depth, height = tray.dimensions
        assert (width, depth, height) == (
            pytest.approx(0.400),
            pytest.approx(0.275),
            pytest.approx(0.025),
        )
        # The long side runs along x. Turned the other way the pair would not
        # match picture.jpg, and would reach 0.55 m out instead of 0.40.
        assert width > depth


def test_the_two_trays_touch_and_do_not_overlap(scene):
    blue, red = (scene.get("tray_blue"), scene.get("tray_red"))
    left = blue.position[0] + blue.dimensions[0] / 2.0
    right = red.position[0] - red.dimensions[0] / 2.0
    assert left == pytest.approx(right, abs=1e-6), "the trays do not meet"
    # And they are side by side rather than one behind the other.
    assert blue.position[1] == pytest.approx(red.position[1])


def test_the_trays_stand_clear_of_the_pedestal_by_the_measured_gap(scene):
    stand = scene.get("robot_stand")
    face = scene.mount_xy[1] + stand.dimensions[1] / 2.0
    near = min(t.position[1] - t.dimensions[1] / 2.0 for t in scene.of_type(TYPE_TRAY))
    assert near - face == pytest.approx(0.15, abs=1e-6)


def test_the_blocks_stand_between_the_pedestal_and_the_trays(scene):
    """Which is the whole reason the 15 cm gap is measured from the pedestal.

    Leave the trays 15 cm from the base axis instead and this strip is 1 cm
    wide, with nowhere for a block to stand.
    """
    trays_start = min(t.position[1] - t.dimensions[1] / 2.0 for t in scene.of_type(TYPE_TRAY))
    for block in scene.of_type(TYPE_CUBE):
        far = block.position[1] + block.dimensions[1] / 2.0
        near = block.position[1] - block.dimensions[1] / 2.0
        assert far < trays_start, f"{block.name} is inside or past the trays"
        assert near > scene.table_near_edge, f"{block.name} hangs off the table edge"


def test_everything_the_arm_works_on_is_on_the_table(scene):
    """A tray or a cube placed off the edge would fall, or hang in mid air."""
    table = scene.get("table")
    tx, ty = scene.table_xy
    for obj in scene.of_type(TYPE_CUBE) + scene.of_type(TYPE_TRAY):
        for axis, centre, extent in (
            (0, tx, table.dimensions[0]),
            (1, ty, table.dimensions[1]),
        ):
            low = obj.position[axis] - obj.dimensions[axis] / 2.0
            high = obj.position[axis] + obj.dimensions[axis] / 2.0
            assert centre - extent / 2.0 <= low and high <= centre + extent / 2.0, (
                f"{obj.name} hangs off the table on axis {axis}"
            )


def test_the_arm_can_reach_down_onto_every_block(scene):
    """A block is grasped from its top face, so that is what has to be in reach."""
    base = (*scene.mount_xy, scene.mount_height)
    for obj in scene.of_type(TYPE_CUBE):
        target = (obj.position[0], obj.position[1], obj.top_z)
        distance = math.dist(base, target)
        assert distance < UR3_REACH, f"{obj.name} is {distance:.3f} m from the base"


def test_the_arm_can_reach_the_height_it_releases_a_block_from(scene, planner_params):
    """For a tray, what has to be in reach is the release pose, not the rim.

    The planner lets a carried block go from above the rim rather than lowering
    it in, so the pose it actually commands sits a block's height and two
    clearances higher than the tray itself. Measuring to the rim would hold the
    layout to a pose the planner never asks for -- and, as the next test
    records, one this arm could not make.
    """
    base = (*scene.mount_xy, scene.mount_height)
    block_height = scene.of_type(TYPE_CUBE)[0].dimensions[2]
    for tray in scene.of_type(TYPE_TRAY):
        release_z = (
            tray.top_z
            + planner_params["drop_gap"]
            + planner_params["tool_gap"]
            + block_height
        )
        distance = math.dist(base, (tray.position[0], tray.position[1], release_z))
        assert distance < UR3_REACH, f"{tray.name} release is {distance:.3f} m from the base"


def test_the_arm_cannot_reach_down_to_the_tray_rims(scene):
    """A known limit of this layout, recorded so it is not discovered late.

    This was written against a base_link height of 0.975, since corrected to
    the measured 0.885. The pedestal has, in effect, already come down: the
    tray rims that used to sit at or just past a UR3's reach are now closer
    to 0.40-0.47 m out, comfortably inside it. Lowering into a tray and
    opening the gripper there may now be possible where it was not before --
    worth trying on the real cell rather than assumed from this docstring,
    since it is exactly the thing this test used to rule out.
    """
    base = (*scene.mount_xy, scene.mount_height)
    for tray in scene.of_type(TYPE_TRAY):
        rim = math.dist(base, (tray.position[0], tray.position[1], tray.top_z))
        assert rim >= UR3_REACH - 0.005, (
            f"{tray.name} rim is {rim:.4f} m from the base, now comfortably in "
            f"reach -- lowering into the tray has become possible and this test "
            f"and the release strategy in actions.py should be revisited"
        )


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


@pytest.fixture
def planner_params():
    """The planner's parameters, or a skip if the package is not beside us."""
    if not os.path.exists(PLANNER_YAML):
        pytest.skip("plantorv_planner is not in this source tree")
    import yaml

    with open(PLANNER_YAML) as handle:
        return yaml.safe_load(handle)["planner"]["ros__parameters"]


def _planner_poses(scene, params):
    """Every tool pose the planner builds for this layout, as (name, point).

    The planner derives these from the object poses and its own clearances, so
    they are what has to fit -- not the object positions, which are what
    test_everything_the_arm_works_on_is_within_its_reach already checks.
    """
    transit = params["transit_height"]
    gap = params["tool_gap"]
    approach = params["approach_distance"]
    drop = params["drop_gap"]
    cube_height = scene.of_type(TYPE_CUBE)[0].dimensions[2]

    poses = []
    for obj in scene.of_type(TYPE_CUBE):
        x, y = obj.position[0], obj.position[1]
        poses.append((f"{obj.name} transit", (x, y, transit)))
        poses.append((f"{obj.name} approach", (x, y, obj.top_z + gap + approach)))
        poses.append((f"{obj.name} grasp", (x, y, obj.top_z + gap)))
    for obj in scene.of_type(TYPE_TRAY):
        x, y = obj.position[0], obj.position[1]
        poses.append((f"{obj.name} transit", (x, y, transit)))
        poses.append((f"{obj.name} approach", (x, y, obj.top_z + gap + approach)))
        poses.append((f"{obj.name} release", (x, y, obj.top_z + drop + gap + cube_height)))
    return poses


def _base(scene):
    x, y = scene.mount_xy
    return (x, y, scene.mount_height)


def _shoulder(scene):
    """Where the arm actually reaches from, which is what UR3_REACH is about."""
    x, y = scene.mount_xy
    return (x, y, scene.mount_height + SHOULDER_HEIGHT)


def test_every_pose_the_planner_builds_is_inside_its_own_reach_guard(scene, planner_params):
    """The outer bound of the workspace guard has to clear the whole task.

    Since the Cartesian back end arrived, nothing checks collisions and this
    guard is what refuses a bad tool target. Set it under the furthest pose the
    task needs and the robot fails the task instead; set it over the arm's
    reach and it stops guarding anything.
    """
    shoulder = _shoulder(scene)
    limit = planner_params["workspace_max_reach"]

    # A little over UR3_REACH is allowed: the quoted 500 mm is the
    # working radius, and tool0 was measured 0.506 m from the shoulder on
    # this arm at full stretch. Much over it and the guard is guarding
    # nothing again.
    assert limit <= UR3_REACH + 0.02, (
        "the guard is well outside the arm's reach, so it guards nothing"
    )

    for name, point in _planner_poses(scene, planner_params):
        distance = math.dist(shoulder, point)
        assert distance < limit, (
            f"{name} is {distance:.4f} m from the shoulder, past {limit}"
        )


def test_every_pose_the_planner_builds_is_above_the_desk(scene, planner_params):
    floor = planner_params["workspace_min_z"]
    assert floor >= scene.table_height, "the floor of the guard is inside the desk"
    for name, point in _planner_poses(scene, planner_params):
        assert point[2] >= floor, f"{name} is at z = {point[2]:.4f}, below {floor}"


def test_no_traverse_passes_inside_the_inner_reach_guard(scene, planner_params):
    """A traverse is a straight line, and it may pass over the arm's base.

    The cubes sit either side of the stand, because the strip in front of it is
    too narrow for one, so the line from the leftmost to the rightmost crosses
    the centreline directly above base_link. The inner bound of the guard has
    to be under that distance or sort_cubes.xml is refused halfway through.

    Measured from the shoulder, as the guard measures it. A traverse
    crossing the centreline passes closest to the axis, not to the point
    on the floor under it, and the shoulder is where the arm's own
    geometry puts that axis.
    """
    shoulder = _shoulder(scene)
    limit = planner_params["workspace_min_reach"]
    plane = planner_params["transit_height"]
    # Only the things the arm actually traverses between. The table's centre
    # is 0.53 m out and its corners further; the arm never goes there, and
    # including them would make this test about the furniture.
    points = [
        (o.position[0], o.position[1], plane)
        for o in list(scene.of_type(TYPE_CUBE)) + list(scene.of_type(TYPE_TRAY))
    ]

    closest = min(
        math.dist(
            shoulder, tuple(a + step / 200.0 * (b - a) for a, b in zip(first, second))
        )
        for first in points
        for second in points
        for step in range(201)
    )
    assert closest > limit, (
        f"a traverse comes {closest:.4f} m from the base, inside the {limit} m guard"
    )


def test_a_position_measured_on_the_pendant_can_be_written_as_it_reads(scene):
    """`frame: base` means the numbers the UR pendant shows.

    Its Base feature frame is base_link turned by pi, and base_link is turned
    by robot.yaw relative to the scene, so the rotation between the two is
    yaw + pi. With the yaw of pi this cell uses, that is a full turn and the
    x and y are the same in both -- which is worth a test rather than a
    remembered coincidence, because it stops being true the moment the arm is
    bolted on at another angle.
    """
    assert scene.mount_yaw == pytest.approx(math.pi)
    for xy in ([-0.15, 0.25], [0.0, 0.30], [0.12, -0.05]):
        assert scene.to_world(xy, "base") == pytest.approx(tuple(xy))
    # And the default is the scene frame, unchanged.
    assert scene.to_world([0.2, 0.4]) == pytest.approx((0.2, 0.4))


def test_the_pendant_frame_follows_the_base_yaw(scene):
    """Turn the arm and the conversion has to turn with it."""
    turned = Scene.from_file(SCENE_FILE)
    turned.config["robot"]["yaw"] = 0.0
    # yaw 0 leaves only ur_description's pi between base and base_link, so a
    # pendant reading comes back negated rather than unchanged.
    assert turned.to_world([0.15, 0.25], "base") == pytest.approx((-0.15, -0.25))


def test_an_unknown_frame_is_refused_rather_than_assumed(scene):
    with pytest.raises(ValueError, match="pendant"):
        scene.to_world([0.0, 0.0], "tool")

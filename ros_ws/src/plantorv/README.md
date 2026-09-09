# PLANTORV on a UR3

The ROS 2 side of PLANTORV: a simulated workcell, a planner that executes known
manipulation actions, and a behaviour tree executor that reads `bt.xml`.

Tested target is **ROS 2 Humble** with **Gazebo Classic 11**.

```
                bt.xml
                  |
          plantorv_bt          ticks the tree, one goal at a time
                  |  ExecuteAction (action)
          plantorv_planner     turns "pick cube_1" into planned, executed motion
                  |  MoveGroup / GetCartesianPath / ExecuteTrajectory
             move_group        plantorv_moveit_config
                  |  FollowJointTrajectory
          gazebo_ros2_control  plantorv_sim
```

Alongside them, `plantorv_sim` also answers *where things are*
(`/get_object_pose`, `/list_objects`) and *what is in the hand*
(`/attach_object`). Those three interfaces are the seam: today they are backed
by Gazebo ground truth, and they are what the segmentation and depth pipeline
in the parent repository will eventually answer instead.

## The packages

| package | what it is |
|---|---|
| `plantorv_interfaces` | the `ExecuteAction` action and the three scene services |
| `plantorv_sim` | the workcell: desk, stand, trays, cubes, the UR3, and the nodes that keep Gazebo and MoveIt agreeing |
| `plantorv_moveit_config` | SRDF, kinematics, OMPL and controller configuration for the cell |
| `plantorv_planner` | the four known actions, planned with MoveIt |
| `plantorv_bt` | a subset of BehaviorTree.CPP 4, in Python, that runs `bt.xml` |
| `plantorv_bringup` | one launch file for the lot |

## The cell

Frame `world` is on the floor, at the centre of the desk top; x runs along the
long side, y towards the robot.

- a desk 200 x 100 cm, 1 m high;
- a UR3 at the middle of the long side, on a 20 cm stand standing on the desk,
  so `base_link` is 1.20 m above the floor;
- a blue tray and a red tray in the middle of the desk;
- three 10 cm cubes between the robot and the trays.

All of it comes from one file, [`plantorv_sim/config/scene.yaml`](plantorv_sim/config/scene.yaml).
Change a number there and regenerate the world:

```bash
python3 src/plantorv/plantorv_sim/scripts/generate_world.py
```

`plantorv_sim/test/test_scene.py` then checks the result, including that every
tray and every cube is still inside the UR3's 500 mm reach. That last check
matters: the arm's base is only 20 cm above the surface it works on, so the
usable workspace is small and easy to lay a tray just outside of.

## Building

```bash
ros_source                     # ROS 2 Humble
cd ros_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The dependencies not pulled in by `rosdep` from a bare Humble install are
`ros-humble-ur-description`, `ros-humble-moveit`, `ros-humble-gazebo-ros2-control`
and `ros-humble-gazebo-ros-pkgs`.

## Running

Everything, sorting the cubes by colour:

```bash
ros2 launch plantorv_bringup plantorv.launch.py
```

A specific tree:

```bash
ros2 launch plantorv_bringup plantorv.launch.py \
  tree:=$(ros2 pkg prefix plantorv_bt)/share/plantorv_bt/trees/pick_and_place.xml
```

The cell without a tree, to drive the planner by hand:

```bash
ros2 launch plantorv_bringup plantorv.launch.py run_tree:=false
ros2 action send_goal /execute_action plantorv_interfaces/action/ExecuteAction \
  "{action_name: pick, target: cube_1}"
ros2 action send_goal /execute_action plantorv_interfaces/action/ExecuteAction \
  "{action_name: place, target: tray_blue}"
```

## The planner

`plantorv_planner` serves one action, `ExecuteAction`, with four known action
names:

| action | what it does |
|---|---|
| `home` | to the home configuration |
| `move_to` | tool to a named object's approach height, or to a given pose |
| `pick` | approach, straight descent, grasp, straight retreat |
| `place` | approach, straight descent, release over the target, straight retreat |

Adding a fifth means a method and a line in `DISPATCH` in
[`actions.py`](plantorv_planner/plantorv_planner/actions.py); no tree and no
message changes.

**"Optimal" means shortest in joint space.** For each free-space move the
planner asks OMPL for `planning_attempts` plans (six by default) and executes
the one whose total joint travel is least. RRTConnect is fast but its output
varies a lot between runs, and this costs a fraction of a second to remove the
worst of that variation. Setting `planner_id` to `RRTstar` instead gets a
single asymptotically optimal search, which spends its whole time budget every
time. The descent onto an object and the retreat from it are Cartesian, not
free-space, so the tool comes down on a cube rather than swinging into it.

Grasping is faked, deliberately: there is no gripper. `pick` deletes the cube's
Gazebo model and attaches it to `tool0` in the planning scene, so the arm
carries it and keeps planning around it; `place` respawns it above the tray and
lets it drop. That gets the pick-and-place logic right before any gripper
dynamics are involved. The two gripper drivers already in this workspace
(`gripper_robotiq_2f85`, `gripper_soft_mgrip_p2`) are what replace it.

## The behaviour tree

`plantorv_bt` implements a subset of BehaviorTree.CPP 4 in Python, so the trees
are ordinary `bt.xml`, readable in Groot and portable to the C++ library:

```xml
<root BTCPP_format="4" main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence>
      <Home/>
      <RetryUntilSuccessful num_attempts="3">
        <Pick object="cube_1"/>
      </RetryUntilSuccessful>
      <Place target="tray_blue"/>
      <Home/>
    </Sequence>
  </BehaviorTree>
</root>
```

Available: `Sequence`, `SequenceWithMemory`, `ReactiveSequence`, `Fallback`,
`ReactiveFallback`, `Parallel`; `Inverter`, `ForceSuccess`, `ForceFailure`,
`RetryUntilSuccessful`, `Repeat`, `KeepRunningUntilFailure`, `Timeout`;
`SubTree`; and the leaves `Home`, `MoveTo`, `Pick`, `Place`, `DetectObjects`,
`MatchingTray`, `Log`, `SetBlackboard`, `PopFromList`, `Sleep`,
`AlwaysSuccess`, `AlwaysFailure`.

Not implemented: scripting expressions, pre/post conditions, typed ports. An
unknown tag is refused when the file is loaded, before the robot moves.

Two trees ship in [`plantorv_bt/trees/`](plantorv_bt/trees/):
`pick_and_place.xml` moves one named cube, and `sort_cubes.xml` detects every
cube and puts each in the tray of its own colour.

## Tests

The parts that do not need ROS are tested as plain Python, and that is most of
the logic worth testing:

```bash
cd ros_ws/src/plantorv
python3 -m pytest plantorv_bt/test plantorv_sim/test
```

`colcon test` runs the same files.

## Known rough edges

- **Reach.** A UR3 mounted 20 cm above its own work surface has very little
  room. The layout in `scene.yaml` fits, but there is not much margin, and a
  larger cube or a tray moved outwards will not.
- **IK.** The kinematics plugin is KDL, which is numerical and seeded from the
  current state, so a pick occasionally fails for no better reason than a bad
  seed. That is why the example trees wrap `Pick` in `RetryUntilSuccessful`.
  Switching `kinematics.yaml` to `trac_ik` or the UR analytic solver removes
  it.
- **Release, not placement.** The fake grasp drops objects from just above a
  tray's rim rather than lowering them in. A rigid 10 cm cube placed into a
  tray of nearly its own size jams; dropping does not.

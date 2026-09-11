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

"""The whole stack: simulator or real arm, MoveIt, planner, behaviour tree.

    ros2 launch plantorv_bringup plantorv.launch.py
    ros2 launch plantorv_bringup plantorv.launch.py tree:=/path/to/bt.xml
    ros2 launch plantorv_bringup plantorv.launch.py run_tree:=false

`real:=true` swaps Gazebo for the UR driver and moves the actual arm. It needs
robot_ip, and it needs the robot ready for it -- Remote Control mode, the
External Control URCap running on the pendant, and the speed slider down.
Everything above the hardware is unchanged: the same description, the same
`world` frame in the same place, the same planner, the same trees.

    ros2 launch plantorv_bringup plantorv.launch.py \\
      real:=true robot_ip:=192.168.1.102 run_tree:=false

Worth doing first, and moving nothing: the same path against mock hardware.

    ros2 launch plantorv_bringup plantorv.launch.py \\
      real:=true use_fake_hardware:=true

The last form brings the cell up without running anything, which is how to
drive the planner by hand::

    ros2 action send_goal /execute_action plantorv_interfaces/action/ExecuteAction \\
      "{action_name: pick, target: cube_1}"

The pieces come up in order rather than at once: move_group needs the robot
description and the controllers Gazebo brings with it, and starting it into an
empty world only produces a minute of warnings. The planner and the tree wait
for what they need on their own, so they are started with move_group.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def _include(package, launch_file, arguments=None, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), "launch", launch_file)
        ),
        launch_arguments=(arguments or {}).items(),
        condition=condition,
    )


def generate_launch_description():
    default_tree = os.path.join(
        get_package_share_directory("plantorv_bt"), "trees", "sort_cubes.xml"
    )

    arguments = [
        DeclareLaunchArgument("gui", default_value="true", description="Run the Gazebo client"),
        DeclareLaunchArgument("rviz", default_value="true", description="Run RViz with MoveIt"),
        DeclareLaunchArgument("ur_type", default_value="ur3"),
        DeclareLaunchArgument(
            "run_tree",
            default_value="true",
            description="Start the behaviour tree executor as well",
        ),
        DeclareLaunchArgument("tree", default_value=default_tree),
        DeclareLaunchArgument(
            "moveit_delay",
            default_value="6.0",
            description="Seconds to let Gazebo and the controllers settle first",
        ),
        DeclareLaunchArgument(
            "real",
            default_value="false",
            description="Drive the actual UR3 instead of Gazebo. Needs robot_ip.",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="0.0.0.0",
            description="IP of the UR control box. Only read when real:=true.",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="With real:=true, take the hardware launch path against mock "
            "hardware. Exercises everything except the arm.",
        ),
        DeclareLaunchArgument(
            "use_moveit",
            default_value="false",
            description="Start move_group. Off by default: `home` goes straight to the "
            "trajectory controller and the straight moves go to the Cartesian "
            "controller, so nothing in the normal cycle needs it.",
        ),
        DeclareLaunchArgument(
            "gripper",
            default_value="true",
            description="Start the Robotiq 2f85 driver alongside the arm.",
        ),
        DeclareLaunchArgument(
            "use_scaled_controller",
            default_value="false",
            description="Use ur_controllers/ScaledJointTrajectoryController on the real "
            "arm. Off because it segfaults ros2_control_node on this stack; see the "
            "note by trajectory_controller below.",
        ),
        DeclareLaunchArgument(
            "headless_mode",
            default_value="true",
            description="Driver pushes the control script itself. False waits for the "
            "External Control URCap program on the pendant.",
        ),
    ]

    real = LaunchConfiguration("real")

    # There is no /clock without Gazebo, and a node left on sim time with no
    # clock publisher simply stops: every timer waits forever for time to
    # start, including the one that ticks the behaviour tree.
    use_sim_time = PythonExpression(["'false' if '", real, "' == 'true' else 'true'"])

    # Which trajectory controller is actually running, which the planner has to
    # switch and, when it is running, move_group has to send to.
    #
    # joint_trajectory_controller everywhere, including on the real arm, and
    # not by preference.
    #
    # ur_controllers/ScaledJointTrajectoryController is the one that belongs
    # here: it reads the speed_scaling state interface and stretches a
    # trajectory when the pendant slider is below 100%, which makes the slider
    # a working speed limit rather than a source of following errors. But on
    # this stack it segfaults ros2_control_node the moment a trajectory
    # arrives, with the fault inside its own update():
    #
    #   #0 ur_controllers::ScaledJointTrajectoryController::update(...)
    #   Segmentation fault (Address not mapped to object [(nil)])
    #
    # It did so against mock hardware and again against the real arm, having
    # configured and activated cleanly both times. The description does declare
    # speed_scaling/speed_scaling_factor, which is the interface name the
    # controller defaults to, so a missing interface is not the explanation.
    #
    # The plain controller has run every trajectory asked of it in Gazebo, on
    # mock hardware and, with this change, on the arm. What is given up is
    # slider compensation: the robot still honours the slider, but the
    # controller does not know it has, so a slider below 100% shows up as
    # position error rather than a stretched trajectory. Safety therefore has
    # to come from the trajectory being slow in the first place, which
    # joint_speed in planner.yaml makes it -- 0.10 rad/s, about 6 degrees a
    # second.
    #
    # use_scaled_controller:=true puts the scaled one back, for testing whether
    # a newer ur_controllers has fixed it.
    trajectory_controller = PythonExpression(
        [
            "'scaled_joint_trajectory_controller' if ('",
            real,
            "' == 'true' and '",
            LaunchConfiguration("use_fake_hardware"),
            "' != 'true' and '",
            LaunchConfiguration("use_scaled_controller"),
            "' == 'true') else 'joint_trajectory_controller'",
        ]
    )

    simulation = _include(
        "plantorv_sim",
        "simulation.launch.py",
        {"gui": LaunchConfiguration("gui"), "ur_type": LaunchConfiguration("ur_type")},
        condition=UnlessCondition(real),
    )

    hardware = _include(
        "plantorv_sim",
        "hardware.launch.py",
        {
            "robot_ip": LaunchConfiguration("robot_ip"),
            "ur_type": LaunchConfiguration("ur_type"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "use_scaled_controller": LaunchConfiguration("use_scaled_controller"),
            "gripper": LaunchConfiguration("gripper"),
        },
        condition=IfCondition(real),
    )

    delayed = TimerAction(
        period=LaunchConfiguration("moveit_delay"),
        actions=[
            _include(
                "plantorv_moveit_config",
                "move_group.launch.py",
                {
                    "ur_type": LaunchConfiguration("ur_type"),
                    "use_sim_time": use_sim_time,
                    "sim_gazebo": PythonExpression(
                        ["'false' if '", real, "' == 'true' else 'true'"]
                    ),
                    "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
                    "trajectory_controller": trajectory_controller,
                },
                condition=IfCondition(LaunchConfiguration("use_moveit")),
            ),
            _include(
                "plantorv_moveit_config",
                "moveit_rviz.launch.py",
                {"ur_type": LaunchConfiguration("ur_type"), "use_sim_time": use_sim_time},
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'true' if ('",
                            LaunchConfiguration("rviz"),
                            "' == 'true' and '",
                            LaunchConfiguration("use_moveit"),
                            "' == 'true') else 'false'",
                        ]
                    )
                ),
            ),
            _include(
                "plantorv_planner",
                "planner.launch.py",
                {
                    "use_sim_time": use_sim_time,
                    "trajectory_controller": trajectory_controller,
                    "use_moveit": LaunchConfiguration("use_moveit"),
                    # Only when there is a real gripper on a real arm.
                    "use_gripper": PythonExpression(
                        [
                            "'true' if ('", real, "' == 'true' and '",
                            LaunchConfiguration("gripper"), "' == 'true' and '",
                            LaunchConfiguration("use_fake_hardware"), "' != 'true') "
                            "else 'false'",
                        ]
                    ),
                },
            ),
            _include(
                "plantorv_bt",
                "behaviour_tree.launch.py",
                {"tree": LaunchConfiguration("tree"), "use_sim_time": use_sim_time},
                condition=IfCondition(LaunchConfiguration("run_tree")),
            ),
        ],
    )

    return LaunchDescription(arguments + [simulation, hardware, delayed])

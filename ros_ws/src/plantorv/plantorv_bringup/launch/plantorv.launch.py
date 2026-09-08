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

"""The whole stack: simulator, MoveIt, planner, behaviour tree.

    ros2 launch plantorv_bringup plantorv.launch.py
    ros2 launch plantorv_bringup plantorv.launch.py tree:=/path/to/bt.xml
    ros2 launch plantorv_bringup plantorv.launch.py run_tree:=false

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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


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
    ]

    simulation = _include(
        "plantorv_sim",
        "simulation.launch.py",
        {"gui": LaunchConfiguration("gui"), "ur_type": LaunchConfiguration("ur_type")},
    )

    delayed = TimerAction(
        period=LaunchConfiguration("moveit_delay"),
        actions=[
            _include(
                "plantorv_moveit_config",
                "move_group.launch.py",
                {"ur_type": LaunchConfiguration("ur_type"), "use_sim_time": "true"},
            ),
            _include(
                "plantorv_moveit_config",
                "moveit_rviz.launch.py",
                {"ur_type": LaunchConfiguration("ur_type"), "use_sim_time": "true"},
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            _include("plantorv_planner", "planner.launch.py", {"use_sim_time": "true"}),
            _include(
                "plantorv_bt",
                "behaviour_tree.launch.py",
                {"tree": LaunchConfiguration("tree"), "use_sim_time": "true"},
                condition=IfCondition(LaunchConfiguration("run_tree")),
            ),
        ],
    )

    return LaunchDescription(arguments + [simulation, delayed])

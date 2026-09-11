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

"""RViz with the MotionPlanning display, wired to the same configuration."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from plantorv_moveit_config.moveit_config import build


def _setup(context, *args, **kwargs):
    moveit_config = build(
        sim_gazebo=LaunchConfiguration("sim_gazebo").perform(context),
        use_fake_hardware=LaunchConfiguration("use_fake_hardware").perform(context),
        ur_type=LaunchConfiguration("ur_type").perform(context),
    )
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    rviz_config = os.path.join(
        get_package_share_directory("plantorv_moveit_config"), "config", "moveit.rviz"
    )

    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                {"use_sim_time": use_sim_time},
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("sim_gazebo", default_value="true"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("ur_type", default_value="ur3"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            OpaqueFunction(function=_setup),
        ]
    )

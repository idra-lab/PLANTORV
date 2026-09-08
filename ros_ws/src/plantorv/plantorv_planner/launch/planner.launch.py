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

"""The planner node, on its own."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("plantorv_planner"), "config", "planner.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("planner_config", default_value=default_config),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="plantorv_planner",
                executable="planner",
                name="planner",
                output="screen",
                parameters=[
                    LaunchConfiguration("planner_config"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )

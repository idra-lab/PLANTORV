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

"""The behaviour tree executor.

    ros2 launch plantorv_bt behaviour_tree.launch.py tree:=/path/to/bt.xml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_tree = os.path.join(
        get_package_share_directory("plantorv_bt"), "trees", "sort_cubes.xml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tree", default_value=default_tree, description="The bt.xml to run"
            ),
            DeclareLaunchArgument("tick_rate", default_value="10.0"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument(
                "preview_seconds",
                default_value="120.0",
                description="Seconds a leaf shows its target before moving. "
                "0 moves straight away. A tree that sets preview_seconds "
                "itself overrides this.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="plantorv_bt",
                executable="bt_executor",
                name="bt_executor",
                output="screen",
                parameters=[
                    {
                        "tree_file": LaunchConfiguration("tree"),
                        "tick_rate": LaunchConfiguration("tick_rate"),
                        "autostart": LaunchConfiguration("autostart"),
                        "loop": LaunchConfiguration("loop"),
                        "preview_seconds": LaunchConfiguration("preview_seconds"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),
        ]
    )

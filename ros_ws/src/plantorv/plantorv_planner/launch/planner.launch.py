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
            # Overridden by plantorv_bringup, because the name differs between
            # the two back ends: Gazebo spawns joint_trajectory_controller and
            # the UR driver spawns scaled_joint_trajectory_controller. The
            # planner switches between this and the Cartesian controller, and a
            # switch naming a controller the manager has not loaded fails.
            DeclareLaunchArgument(
                "trajectory_controller", default_value="joint_trajectory_controller"
            ),
            DeclareLaunchArgument("use_moveit", default_value="false"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            # Height of the transit plane, in the planning frame. Settable
            # here so it can be tried without editing planner.yaml, but
            # raising it costs reach rather than buying it: base_link sits
            # at z = 0.885 (measured with a tape; it was assumed 0.975
            # before), so a plane at 1.5 is 0.615 m straight up from it,
            # past both workspace_max_reach and the UR3's own 0.5 m.
            DeclareLaunchArgument("transit_height", default_value="1.133"),
            DeclareLaunchArgument("use_gripper", default_value="false"),
            Node(
                package="plantorv_planner",
                executable="planner",
                name="planner",
                output="screen",
                parameters=[
                    LaunchConfiguration("planner_config"),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "trajectory_controller": LaunchConfiguration("trajectory_controller"),
                        "use_moveit": LaunchConfiguration("use_moveit"),
                        "dry_run": LaunchConfiguration("dry_run"),
                        "transit_height": LaunchConfiguration("transit_height"),
                        "use_gripper": LaunchConfiguration("use_gripper"),
                    },
                ],
            ),
        ]
    )

#!/usr/bin/env python3

"""Record the static colour camera's pose in ``world`` once and exit."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "output_file",
            default_value=os.path.join(
                get_package_share_directory("plantorv_bringup"),
                "config",
                "camera_to_world_transform.yaml",
            ),
            description="Where to save the world <- static camera transform.",
        ),
        DeclareLaunchArgument("timeout", default_value="30.0"),
        DeclareLaunchArgument(
            "source_frame", default_value="static_camera_color_optical_frame"
        ),
    ]

    recorder = Node(
        package="plantorv_ros",
        executable="save_camera_world_transform",
        name="save_camera_world_transform",
        output="screen",
        parameters=[{
            "source_frame": LaunchConfiguration("source_frame"),
            "output_file": LaunchConfiguration("output_file"),
            "timeout": LaunchConfiguration("timeout"),
        }],
        on_exit=Shutdown(),
    )
    return LaunchDescription(args + [recorder])

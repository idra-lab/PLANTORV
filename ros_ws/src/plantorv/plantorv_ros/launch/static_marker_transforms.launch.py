#!/usr/bin/env python3

"""Publish the recorded marker frames, with no camera and no detection.

This is the bringup half: include it and the frames recorded by
``save_marker_transforms.launch.py`` exist again, whether or not the
markers are in view.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from plantorv_ros.marker_transform_file import DEFAULT_FILE


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'input_file',
            default_value=DEFAULT_FILE,
            description='Recorded transforms to publish.',
        ),
        DeclareLaunchArgument(
            'frames',
            default_value="['']",
            description='Child frames to publish; empty means all.',
        ),
    ]

    publisher = Node(
        package='plantorv_ros',
        executable='static_marker_publisher',
        name='static_marker_publisher',
        output='screen',
        parameters=[{
            'input_file': LaunchConfiguration('input_file'),
            'frames': LaunchConfiguration('frames'),
        }],
    )

    return LaunchDescription(args + [publisher])

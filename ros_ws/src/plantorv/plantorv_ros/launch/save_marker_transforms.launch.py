#!/usr/bin/env python3

"""Record the marker frames once, then shut everything down.

Runs the camera and both detector nodes, waits for the frames to hold
still, averages them and writes the file. When the recorder exits the
whole launch goes down with it, camera included.

Place the board and the robot first, check the frames in
``charuco_viz.launch.py``, and only then record.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    Shutdown,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'output_file',
            default_value=os.path.join(
                get_package_share_directory('plantorv_bringup'),
                'config',
                'static_transforms.yaml',
            ),
            description='Where to write the recorded transforms.',
        ),
        DeclareLaunchArgument(
            'samples',
            default_value='30',
            description='Samples to average per frame.',
        ),
        DeclareLaunchArgument('timeout', default_value='30.0'),
        DeclareLaunchArgument(
            'settle_time',
            default_value='2.0',
            description='Ignore the first seconds, while the camera settles.',
        ),
        DeclareLaunchArgument(
            'max_translation_spread',
            default_value='0.012',
            description='Refuse a frame that wanders more, in metres.',
        ),
        DeclareLaunchArgument(
            'max_rotation_spread',
            default_value='3.0',
            description='Refuse a frame that wanders more, in degrees.',
        ),
        DeclareLaunchArgument(
            'outlier_sigma',
            default_value='2.5',
            description='Drop samples this far out; 0 keeps all.',
        ),
        DeclareLaunchArgument(
            'force',
            default_value='false',
            description='Write even an unsteady frame.',
        ),
        DeclareLaunchArgument(
            'frames',
            default_value="['charuco_board', 'robot_base_marker']",
            description='Child frames to record.',
        ),
        DeclareLaunchArgument(
            'parent_frame',
            default_value='camera_color_optical_frame',
            description='Frame to record them in.',
        ),
    ]

    # The detectors and the camera come from the visualization launch,
    # with RViz left out: nothing to watch here, the recorder reports
    # what it got.
    detectors = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('plantorv_ros'),
                'launch',
                'charuco_viz.launch.py',
            ])
        ),
        launch_arguments={'use_rviz': 'false'}.items(),
    )

    recorder = Node(
        package='plantorv_ros',
        executable='save_marker_transforms',
        name='save_marker_transforms',
        output='screen',
        parameters=[{
            'parent_frame': LaunchConfiguration('parent_frame'),
            'frames': LaunchConfiguration('frames'),
            'output_file': LaunchConfiguration('output_file'),
            'samples': LaunchConfiguration('samples'),
            'timeout': LaunchConfiguration('timeout'),
            'settle_time': LaunchConfiguration('settle_time'),
            'outlier_sigma': LaunchConfiguration('outlier_sigma'),
            'max_translation_spread': LaunchConfiguration(
                'max_translation_spread'
            ),
            'max_rotation_spread': LaunchConfiguration(
                'max_rotation_spread'
            ),
            'force': LaunchConfiguration('force'),
        }],
        on_exit=Shutdown(),
    )

    return LaunchDescription(args + [detectors, recorder])

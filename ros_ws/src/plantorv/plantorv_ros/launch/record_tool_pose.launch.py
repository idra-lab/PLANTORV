#!/usr/bin/env python3

"""Write down where the tool is now, as a frame that can be moved to later.

Put the arm where you want it -- by hand, by the pendant, by any means --
and run this. It averages the tool's pose in the planning frame and
writes it as ``taught_point``, which is then a frame like any other:

    ros2 launch plantorv_ros record_tool_pose.launch.py
    ros2 launch plantorv_ros static_marker_transforms.launch.py \\
        input_file:=$HOME/.ros/plantorv/taught_point.yaml

and a tree reaches it with the point at its own origin::

    <MoveToPoint x="0.0" y="0.0" z="0.0" frame="taught_point"/>

It is recorded under a new name on purpose. Written as ``tool0`` it
would be replayed as ``tool0``, and then two things would be claiming
the frame the robot description owns.

This is also the measurement that settles an argument between the
camera and the arm. Stand the tool where the camera says a thing is,
record it, and compare: ``ros2 run tf2_ros tf2_echo taught_point
move_to_target`` is the error, in metres, in the planning frame.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'output_file',
            default_value=os.path.expanduser(
                '~/.ros/plantorv/taught_point.yaml'
            ),
            description='Where to write the recorded pose. Kept away from '
                        'the bringup config by default, since this is a '
                        'measurement and not part of the cell.',
        ),
        DeclareLaunchArgument(
            'tool_frame',
            default_value='tool0',
            description='The frame to record. tool0 is what the planner '
                        'commands, so it is what a taught point should be.',
        ),
        DeclareLaunchArgument(
            'record_as',
            default_value='taught_point',
            description='The name it is written under.',
        ),
        DeclareLaunchArgument('parent_frame', default_value='world'),
        DeclareLaunchArgument('samples', default_value='30'),
        DeclareLaunchArgument('timeout', default_value='30.0'),
        # The arm is standing still and its joint states are steadier
        # than a marker seen across the room, so this can be tighter
        # than the default the camera recordings need.
        DeclareLaunchArgument('max_translation_spread', default_value='0.002'),
        DeclareLaunchArgument('max_rotation_spread', default_value='1.0'),
        DeclareLaunchArgument('force', default_value='false'),
    ]

    recorder = Node(
        package='plantorv_ros',
        executable='save_marker_transforms',
        name='record_tool_pose',
        output='screen',
        parameters=[{
            'parent_frame': LaunchConfiguration('parent_frame'),
            # Written out as a list literal rather than a one-element
            # Python list: a list holding a single substitution is
            # flattened to a plain string before the parameter is set,
            # and the node declares these as string arrays.
            'frames': PythonExpression(
                ["['", LaunchConfiguration('tool_frame'), "']"]
            ),
            'record_as': PythonExpression(
                ["['", LaunchConfiguration('record_as'), "']"]
            ),
            'output_file': LaunchConfiguration('output_file'),
            'samples': LaunchConfiguration('samples'),
            'timeout': LaunchConfiguration('timeout'),
            'max_translation_spread': LaunchConfiguration(
                'max_translation_spread'
            ),
            'max_rotation_spread': LaunchConfiguration('max_rotation_spread'),
            'force': LaunchConfiguration('force'),
            # Nothing to settle: the arm is already where it is.
            'settle_time': 0.0,
        }],
        on_exit=Shutdown(),
    )

    return LaunchDescription(args + [recorder])

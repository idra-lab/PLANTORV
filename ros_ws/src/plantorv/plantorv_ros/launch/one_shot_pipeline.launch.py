#!/usr/bin/env python3

"""Launch the Femto Mega and run the one-shot pipeline.

The camera is reached over USB by default. Pass ``use_network:=true`` to
reach it over Ethernet instead; see the package README for the host-side
network setup.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    Shutdown,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def camera_launch(condition, extra_arguments):
    """Include the Orbbec driver with the shared arguments plus extras."""
    launch_arguments = {
        'camera_name': LaunchConfiguration('camera_name'),
        'depth_registration': LaunchConfiguration('depth_registration'),
        'color_width': LaunchConfiguration('color_width'),
        'color_height': LaunchConfiguration('color_height'),
        'depth_width': LaunchConfiguration('depth_width'),
        'depth_height': LaunchConfiguration('depth_height'),
    }
    launch_arguments.update(extra_arguments)

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('orbbec_camera'),
                'launch',
                'femto_mega.launch.py',
            )
        ),
        condition=condition,
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    use_network = LaunchConfiguration('use_network')
    camera_name = LaunchConfiguration('camera_name')

    args = [
        # ------------------------------------------------------------------
        # Camera transport
        # ------------------------------------------------------------------
        DeclareLaunchArgument(
            'use_network',
            default_value='false',
            description='Reach the camera over Ethernet instead of USB.',
        ),
        DeclareLaunchArgument('camera_name', default_value='camera'),
        # Keep the D2C alignment off. With it on the driver rewrites
        # /camera/depth/image_raw into the colour frame, so the depth
        # image arrives at the colour resolution (1280x720). The
        # RGBDMapper in mapping/rgbd_mapper.py does that alignment
        # itself and needs the native 640x576 NFOV depth frame; a
        # pre-aligned one has no matching hardcoded calibration.
        DeclareLaunchArgument('depth_registration', default_value='false'),

        # The sizes the pipeline is calibrated for: colour at the sensor's
        # 1280x720, depth at the native 640x576 of NFOV unbinned. These are
        # the driver's own defaults, stated here because the pipeline
        # depends on them rather than merely tolerating them, and passed to
        # the node as well so a frame of another size is refused instead of
        # quietly mapped with the wrong intrinsics.
        DeclareLaunchArgument('color_width', default_value='1280'),
        DeclareLaunchArgument('color_height', default_value='720'),
        DeclareLaunchArgument('depth_width', default_value='640'),
        DeclareLaunchArgument('depth_height', default_value='576'),

        # Ethernet only. With enumerate_net_device set to false the driver
        # connects straight to net_device_ip:net_device_port instead of
        # relying on broadcast discovery, which is the reliable option on a
        # point-to-point link.
        DeclareLaunchArgument('enumerate_net_device', default_value='false'),
        DeclareLaunchArgument('net_device_ip', default_value='192.168.1.10'),
        DeclareLaunchArgument('net_device_port', default_value='8090'),

        # USB only. Both empty means "take the first camera found".
        DeclareLaunchArgument('serial_number', default_value=''),
        DeclareLaunchArgument('usb_port', default_value=''),

        # ------------------------------------------------------------------
        # Pipeline
        # ------------------------------------------------------------------
        DeclareLaunchArgument(
            'pipeline_script',
            default_value=os.path.expanduser('~/Projects/plantorv/samgpt.py'),
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value=os.path.expanduser('~/Projects/plantorv/output_ros'),
        ),
        DeclareLaunchArgument('sync_slop', default_value='0.02'),
        DeclareLaunchArgument('pointcloud_timeout', default_value='2.0'),

        # Topics are namespaced under camera_name by the Orbbec driver.
        DeclareLaunchArgument(
            'rgb_topic',
            default_value=['/', camera_name, '/color/image_raw'],
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value=['/', camera_name, '/depth/image_raw'],
        ),
        # depth_registered/points is the coloured cloud, which the driver
        # only publishes with enable_colored_point_cloud:=true (and that in
        # turn forces D2C). depth/points is the plain cloud published by
        # default.
        DeclareLaunchArgument(
            'pointcloud_topic',
            default_value=['/', camera_name, '/depth/points'],
        ),
    ]

    network_camera = camera_launch(
        IfCondition(use_network),
        {
            'enumerate_net_device': LaunchConfiguration('enumerate_net_device'),
            'net_device_ip': LaunchConfiguration('net_device_ip'),
            'net_device_port': LaunchConfiguration('net_device_port'),
        },
    )

    # An empty net_device_ip with network enumeration off keeps the driver
    # on USB enumeration only.
    usb_camera = camera_launch(
        UnlessCondition(use_network),
        {
            'enumerate_net_device': 'false',
            'net_device_ip': '',
            'net_device_port': '0',
            'serial_number': LaunchConfiguration('serial_number'),
            'usb_port': LaunchConfiguration('usb_port'),
        },
    )

    # The node waits for the first synchronized frames, so it can start
    # together with the driver. It is one-shot: when it exits, tear the
    # whole launch down so the camera stops too.
    pipeline = Node(
        package='plantorv_ros',
        executable='one_shot_pipeline',
        name='one_shot_pipeline',
        output='screen',
        parameters=[{
            # This launch file starts the driver itself, just above, so
            # the node must not start a second one.
            'start_camera': False,
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
            'pipeline_script': LaunchConfiguration('pipeline_script'),
            'output_dir': LaunchConfiguration('output_dir'),
            # One substitution producing "[1280, 720]", not a Python list
            # of two substitutions: launch_ros flattens the latter into a
            # single concatenated string ("1280720") and rclpy then
            # refuses it as the wrong parameter type, since the node
            # declares these as integer arrays.
            'rgb_size': PythonExpression([
                "[", LaunchConfiguration('color_width'), ", ",
                LaunchConfiguration('color_height'), "]",
            ]),
            'depth_size': PythonExpression([
                "[", LaunchConfiguration('depth_width'), ", ",
                LaunchConfiguration('depth_height'), "]",
            ]),
            'sync_slop': LaunchConfiguration('sync_slop'),
            'pointcloud_timeout': LaunchConfiguration('pointcloud_timeout'),
        }],
        on_exit=Shutdown(),
    )

    return LaunchDescription(args + [network_camera, usb_camera, pipeline])
    # return LaunchDescription([pipeline])

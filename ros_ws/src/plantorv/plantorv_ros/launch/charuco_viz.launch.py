#!/usr/bin/env python3

"""Launch the Femto Mega, the marker frame publishers and RViz.

This is the placement aid: point the camera at the scene, put the board
down, and watch its frame in RViz to judge whether the pose is stable
and the board is well seen from where it lies.

The ChArUco board frames the scene; the ArUco node frames the single
markers standing in it, such as the one on the robot base. Set
``use_aruco:=false`` to leave the second node out.

The recorded transforms are replayed alongside the live ones, so the
two can be compared in RViz: ``charuco_board`` is what the camera sees
right now, ``static_charuco_board`` what was recorded. Set
``use_static_transforms:=false`` to leave them out.

The camera is reached over USB by default. Pass ``use_network:=true``
to reach it over Ethernet instead; see the package README for the
host-side network setup.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def camera_launch(condition, extra_arguments):
    """Include the Orbbec driver with the shared arguments plus extras."""
    launch_arguments = {
        'camera_name': LaunchConfiguration('camera_name'),
        'enable_point_cloud': 'true',
        'enable_colored_point_cloud': LaunchConfiguration(
            'use_colored_cloud'
        ),
        'depth_registration': LaunchConfiguration('use_colored_cloud'),
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
    use_colored_cloud = LaunchConfiguration('use_colored_cloud')
    use_rviz = LaunchConfiguration('use_rviz')
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

        # ------------------------------------------------------------------
        # Point cloud
        # ------------------------------------------------------------------
        # The driver publishes the plain cloud on
        # <camera_name>/depth/points by default. The coloured cloud goes
        # to <camera_name>/depth_registered/points instead, and needs
        # D2C alignment, which is why depth_registration follows it.
        DeclareLaunchArgument(
            'use_colored_cloud',
            default_value='true',
            description='Publish the coloured cloud instead of the plain one.',
        ),

        # Ethernet only. See one_shot_pipeline.launch.py for why
        # enumeration is off by default.
        DeclareLaunchArgument('enumerate_net_device', default_value='false'),
        DeclareLaunchArgument('net_device_ip', default_value='192.168.1.10'),
        DeclareLaunchArgument('net_device_port', default_value='8090'),

        # USB only. Both empty means "take the first camera found".
        DeclareLaunchArgument('serial_number', default_value=''),
        DeclareLaunchArgument('usb_port', default_value=''),

        # ------------------------------------------------------------------
        # Board
        # ------------------------------------------------------------------
        DeclareLaunchArgument(
            'squares_x',
            default_value='5',
            description='Chessboard squares along x, not markers.',
        ),
        DeclareLaunchArgument(
            'squares_y',
            default_value='7',
            description='Chessboard squares along y, not markers.',
        ),
        DeclareLaunchArgument(
            'square_length',
            default_value='0.04',
            description='Square side in metres, as printed.',
        ),
        DeclareLaunchArgument(
            'marker_length',
            default_value='0.02',
            description='Marker side in metres, as printed.',
        ),
        DeclareLaunchArgument('dictionary', default_value='DICT_6X6_250'),
        DeclareLaunchArgument(
            'legacy_pattern',
            default_value='false',
            description='Set for boards generated with OpenCV < 4.6.',
        ),
        DeclareLaunchArgument('min_corners', default_value='6'),
        DeclareLaunchArgument('board_frame', default_value='charuco_board'),
        DeclareLaunchArgument(
            'origin_corner',
            default_value='-1',
            description=(
                'Chessboard corner id to put the frame on, as printed on '
                'the annotated image; -1 keeps the OpenCV board origin.'
            ),
        ),
        DeclareLaunchArgument(
            'axis_length',
            default_value='0.0',
            description='Drawn axis length in metres; 0 means two squares.',
        ),

        # ------------------------------------------------------------------
        # Single ArUco markers
        # ------------------------------------------------------------------
        DeclareLaunchArgument('use_aruco', default_value='true'),
        DeclareLaunchArgument(
            'aruco_marker_length',
            default_value='0.10',
            description=(
                'Side of the black square in metres, border included.'
            ),
        ),
        # Id 0 is byte-identical in DICT_7X7_50, _100, _250 and _1000,
        # so the smallest of the four reads a marker printed from any
        # of them, with the fewest candidates to confuse it.
        DeclareLaunchArgument(
            'aruco_dictionary',
            default_value='DICT_7X7_50',
        ),
        DeclareLaunchArgument(
            'aruco_marker_ids',
            default_value='[0]',
            description='Marker ids to publish a frame for.',
        ),
        DeclareLaunchArgument(
            'aruco_marker_frames',
            default_value="['robot_base_marker']",
            description='Frame name per id, in the same order.',
        ),
        DeclareLaunchArgument(
            'aruco_max_reprojection_error',
            default_value='4.0',
            description='Drop a marker pose worse than this, in pixels.',
        ),

        # ------------------------------------------------------------------
        # Recorded transforms
        # ------------------------------------------------------------------
        DeclareLaunchArgument('use_static_transforms', default_value='true'),
        DeclareLaunchArgument(
            'static_transforms_file',
            default_value=os.path.join(
                get_package_share_directory('plantorv_bringup'),
                'config',
                'static_transforms.yaml',
            ),
        ),
        # Everything in the file, the driver running or not. Its
        # camera_to_camera entry hangs the recorded subtree under the
        # live camera frame, and points that way round precisely so it
        # cannot fight the driver over who parents
        # camera_color_optical_frame.
        DeclareLaunchArgument(
            'static_transforms_frames',
            default_value="['']",
            description='Recorded entries to replay, by name or frame. '
                        'Empty means all of them.',
        ),

        # ------------------------------------------------------------------
        # Visualization
        # ------------------------------------------------------------------
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument(
            'rgb_topic',
            default_value=['/', camera_name, '/color/image_raw'],
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value=['/', camera_name, '/color/camera_info'],
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

    # An empty net_device_ip with network enumeration off keeps the
    # driver on USB enumeration only.
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

    charuco = Node(
        package='plantorv_ros',
        executable='charuco_tf_publisher',
        name='charuco_tf_publisher',
        output='screen',
        parameters=[{
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'squares_x': LaunchConfiguration('squares_x'),
            'squares_y': LaunchConfiguration('squares_y'),
            'square_length': LaunchConfiguration('square_length'),
            'marker_length': LaunchConfiguration('marker_length'),
            'dictionary': LaunchConfiguration('dictionary'),
            'legacy_pattern': LaunchConfiguration('legacy_pattern'),
            'min_corners': LaunchConfiguration('min_corners'),
            'board_frame': LaunchConfiguration('board_frame'),
            'origin_corner': LaunchConfiguration('origin_corner'),
            'axis_length': LaunchConfiguration('axis_length'),
        }],
    )

    aruco = Node(
        package='plantorv_ros',
        executable='aruco_tf_publisher',
        name='aruco_tf_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_aruco')),
        parameters=[{
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'marker_length': LaunchConfiguration('aruco_marker_length'),
            'dictionary': LaunchConfiguration('aruco_dictionary'),
            'marker_ids': LaunchConfiguration('aruco_marker_ids'),
            'marker_frames': LaunchConfiguration('aruco_marker_frames'),
            'max_reprojection_error': LaunchConfiguration(
                'aruco_max_reprojection_error'
            ),
        }],
    )

    use_static_transforms = LaunchConfiguration('use_static_transforms')

    static_transforms = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('plantorv_ros'),
                'launch',
                'static_marker_transforms.launch.py',
            ])
        ),
        condition=IfCondition(use_static_transforms),
        launch_arguments={
            'input_file': LaunchConfiguration('static_transforms_file'),
            'frames': LaunchConfiguration('static_transforms_frames'),
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        condition=IfCondition(use_rviz),
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('plantorv_ros'),
                'rviz',
                'charuco.rviz',
            ]),
        ],
    )

    return LaunchDescription(
        args
        + [
            network_camera,
            usb_camera,
            charuco,
            aruco,
            static_transforms,
            rviz,
        ]
    )

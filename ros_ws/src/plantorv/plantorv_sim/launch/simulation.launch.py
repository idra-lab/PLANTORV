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

"""Gazebo, the UR3 on its stand, its controllers, and the scene nodes.

This brings up the world and the robot but not MoveIt: the planner needs a
move_group whether or not there is a simulator, so that lives in
plantorv_moveit_config and the two are composed in plantorv_bringup.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_share = get_package_share_directory("plantorv_sim")
    gazebo_share = get_package_share_directory("gazebo_ros")

    arguments = [
        DeclareLaunchArgument("gui", default_value="true", description="Run gzclient"),
        DeclareLaunchArgument(
            "world",
            default_value=os.path.join(sim_share, "worlds", "plantorv_table.world"),
            description="Gazebo world; regenerate it with scripts/generate_world.py",
        ),
        DeclareLaunchArgument("ur_type", default_value="ur3"),
        DeclareLaunchArgument(
            "spawn_scene",
            default_value="true",
            description="Run scene_manager and world_model alongside the simulator",
        ),
    ]

    robot_description = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("plantorv_sim"), "urdf", "ur3_workcell.urdf.xacro"]),
            " ur_type:=",
            LaunchConfiguration("ur_type"),
            " sim_gazebo:=true",
        ]
    )

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_share, "launch", "gzserver.launch.py")),
        launch_arguments={"world": LaunchConfiguration("world"), "verbose": "true"}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_share, "launch", "gzclient.launch.py")),
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            # The xacro output has to be declared a string. Left bare, launch_ros
            # tries to parse the URDF as YAML to infer the parameter type, and
            # the first colon in the XML makes that fail.
            {
                "robot_description": ParameterValue(robot_description, value_type=str),
                "use_sim_time": True,
            },
        ],
    )

    # The description already places base_link on top of the stand, and the
    # stand on the table, so the model goes in at the world origin.
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-entity", "ur3_workcell",
            "-x", "0.0", "-y", "0.0", "-z", "0.0",
        ],
    )

    def spawner(controller):
        return Node(
            package="controller_manager",
            executable="spawner",
            output="screen",
            arguments=[controller, "--controller-manager", "/controller_manager"],
        )

    joint_state_broadcaster = spawner("joint_state_broadcaster")
    trajectory_controller = spawner("joint_trajectory_controller")

    # gazebo_ros2_control only exists once the model is in the world, and the
    # trajectory controller wants joint states before it starts, so the two
    # spawners are chained rather than raced.
    controllers = [
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_robot, on_exit=[joint_state_broadcaster])
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=joint_state_broadcaster, on_exit=[trajectory_controller])
        ),
    ]

    scene_nodes = [
        Node(
            package="plantorv_sim",
            executable=executable,
            name=executable,
            output="screen",
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(LaunchConfiguration("spawn_scene")),
        )
        for executable in ("world_model", "scene_manager")
    ]

    return LaunchDescription(
        arguments + [gzserver, gzclient, robot_state_publisher, spawn_robot] + controllers + scene_nodes
    )

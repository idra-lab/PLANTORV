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

"""The real UR3, its controllers, and nothing else.

The counterpart of simulation.launch.py, and deliberately the same shape: it
brings up the robot and the controllers but not MoveIt, so plantorv_bringup can
compose either one under the same planner and behaviour tree.

The robot description is the same ur3_workcell.urdf.xacro the simulator uses,
with sim_gazebo off. That is the point of keeping the table and the pedestal in
the description rather than in the Gazebo world: the real cell gets the same
`world` frame, at the same place, with the same table in it, so the planner's
poses and its workspace guard mean the same thing on both.

What is not here, and is not an oversight:

    Gazebo, obviously, and with it scene_manager. On the real cell there is
    nothing to spawn or delete, and the fake grasp has nothing to do. Only
    world_model runs, and it answers from scene.yaml rather than from
    /gazebo/model_states -- so the object poses are the declared ones, not
    sensed ones. That is fine for moving to a known place and wrong for
    picking anything up, which is what the perception pipeline is for.

Prerequisites on the robot itself, none of which this file can check:

    the arm is in Remote Control mode, if it is an e-Series;
    robot_ip resolves to the controller and the network is up;
    the pendant's speed slider is somewhere low.

With headless_mode true, which is the default, the driver sends the control
script to the robot itself and nothing has to be started on the pendant. Set it
false to use the External Control URCap program instead, and then that program
has to be loaded and running -- if it is not, the driver comes up, the
controllers activate, and trajectories are accepted and silently do nothing.
"""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_share = get_package_share_directory("plantorv_sim")
    controllers = os.path.join(sim_share, "config", "ur3_controllers_real.yaml")

    arguments = [
        DeclareLaunchArgument(
            "robot_ip", description="IP address of the UR control box. Required."
        ),
        DeclareLaunchArgument("ur_type", default_value="ur3"),
        DeclareLaunchArgument(
            "use_scaled_controller",
            default_value="false",
            description="Use ur_controllers/ScaledJointTrajectoryController. Off: it "
            "segfaults ros2_control_node on this stack.",
        ),
        DeclareLaunchArgument(
            "headless_mode",
            default_value="true",
            description="Let the driver push the control script itself, rather than "
            "waiting for the External Control URCap program on the pendant.",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Run the whole stack against mock hardware instead of the arm. "
            "The one way to exercise this launch file without moving anything.",
        ),
        DeclareLaunchArgument(
            "gripper",
            default_value="true",
            description="Start the Robotiq 2f85 driver. Off with use_fake_hardware, "
            "since there is no gripper to talk to.",
        ),
        DeclareLaunchArgument(
            "gripper_port",
            default_value="54321",
            description="TCP port the RS485 URCap listens on for the gripper.",
        ),
        DeclareLaunchArgument(
            "spawn_scene",
            default_value="true",
            description="Run world_model, which answers object poses from scene.yaml",
        ),
    ]

    robot_description = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("plantorv_sim"), "urdf", "ur3_workcell.urdf.xacro"]
            ),
            " ur_type:=",
            LaunchConfiguration("ur_type"),
            " sim_gazebo:=false",
            " use_fake_hardware:=",
            LaunchConfiguration("use_fake_hardware"),
            " robot_ip:=",
            LaunchConfiguration("robot_ip"),
            " headless_mode:=",
            LaunchConfiguration("headless_mode"),
        ]
    )
    description_param = {
        "robot_description": ParameterValue(robot_description, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[description_param],
    )

    # Unlike the simulated cell, nothing else is going to create a controller
    # manager, so this launch runs one.
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[description_param, controllers],
    )

    def spawner(name, extra=None):
        return Node(
            package="controller_manager",
            executable="spawner",
            output="screen",
            arguments=[name, "--controller-manager", "/controller_manager"] + (extra or []),
        )

    def spawn_cartesian_controller(context, *args, **kwargs):
        """Same trick as in simulation.launch.py, for the same reason.

        On Humble a controller reads robot_description from a parameter on its
        own node, and the URDF is xacro output, so it cannot live in the static
        controller yaml. The spawner loads a params file into a controller
        before configuring it.
        """
        urdf = robot_description.perform(context)
        handle = tempfile.NamedTemporaryFile(
            mode="w", prefix="cartesian_motion_controller_real_", suffix=".yaml", delete=False
        )
        yaml.safe_dump(
            {"cartesian_motion_controller": {"ros__parameters": {"robot_description": urdf}}},
            handle,
        )
        handle.close()
        return [
            spawner(
                "cartesian_motion_controller",
                extra=["--inactive", "--param-file", handle.name],
            )
        ]

    joint_state_broadcaster = spawner("joint_state_broadcaster")
    status = spawner("io_and_status_controller")
    # The plain controller unless asked otherwise. The scaled one segfaults
    # ros2_control_node on the first trajectory, on mock hardware and on the
    # arm alike; see the note in plantorv_bringup/launch/plantorv.launch.py.
    trajectory = spawner(
        PythonExpression(
            [
                "'scaled_joint_trajectory_controller' if ('",
                LaunchConfiguration("use_scaled_controller"),
                "' == 'true' and '",
                LaunchConfiguration("use_fake_hardware"),
                "' != 'true') else 'joint_trajectory_controller'",
            ]
        )
    )
    cartesian = OpaqueFunction(function=spawn_cartesian_controller)

    # Chained rather than raced, as in simulation: the trajectory controller
    # wants joint states before it starts.
    chain = [
        RegisterEventHandler(
            OnProcessExit(target_action=joint_state_broadcaster, on_exit=[status])
        ),
        RegisterEventHandler(OnProcessExit(target_action=status, on_exit=[trajectory])),
        RegisterEventHandler(OnProcessExit(target_action=trajectory, on_exit=[cartesian])),
    ]

    # The gripper talks over the robot's RS485 URCap rather than through
    # ros2_control: the node opens a socat tunnel to the control box and
    # drives the gripper over Modbus on the far side. So it is a plain node,
    # started alongside the driver, and it needs the same robot_ip.
    #
    # It needs socat on the PATH and the RS485 URCap running on the pendant.
    # Neither can be checked from here; if the URCap is not running the node
    # fails to open the serial device and exits.
    gripper = Node(
        package="gripper_robotiq_2f85",
        executable="robotiq_2f85",
        name="robotiq_2f85",
        output="screen",
        parameters=[
            {
                "host": LaunchConfiguration("robot_ip"),
                "port": ParameterValue(LaunchConfiguration("gripper_port"), value_type=int),
            }
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'true' if ('",
                    LaunchConfiguration("gripper"),
                    "' == 'true' and '",
                    LaunchConfiguration("use_fake_hardware"),
                    "' != 'true') else 'false'",
                ]
            )
        ),
    )

    world_model = Node(
        package="plantorv_sim",
        executable="world_model",
        name="world_model",
        output="screen",
        condition=IfCondition(LaunchConfiguration("spawn_scene")),
    )

    return LaunchDescription(
        arguments
        + [robot_state_publisher, control_node, joint_state_broadcaster, world_model, gripper]
        + chain
    )

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

"""move_group for the UR3 workcell."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from plantorv_moveit_config.moveit_config import build


#: The controller config/moveit_controllers.yaml is written against.
CONFIGURED_CONTROLLER = "joint_trajectory_controller"


def _retarget_controller(parameters: dict, controller: str) -> dict:
    """Point move_group's controller manager at ``controller``.

    config/moveit_controllers.yaml names one controller, and which one it has
    to be depends on what is underneath: Gazebo spawns
    ``joint_trajectory_controller``, the UR driver spawns
    ``scaled_joint_trajectory_controller``. move_group sends a trajectory to a
    controller by name and does not check that the name exists, so getting this
    wrong on the real arm produces a goal that is simply never executed.

    Renaming here rather than keeping two yaml files, so there is still one
    statement of how the controller is configured and only its name varies.

    MoveItConfigsBuilder hands back ``moveit_simple_controller_manager`` as one
    nested dict rather than as dotted keys, and the controller's own settings
    live under its name inside it. Renaming only the ``controller_names`` entry
    therefore leaves those settings behind under the old name, and move_group
    starts and then refuses every trajectory with "No action namespace
    specified for controller" -- which does not appear until something tries to
    execute, long after launch looked fine.
    """
    if controller == CONFIGURED_CONTROLLER:
        return parameters

    manager = parameters.get("moveit_simple_controller_manager")
    if not isinstance(manager, dict) or CONFIGURED_CONTROLLER not in manager:
        raise RuntimeError(
            f"cannot retarget move_group at '{controller}': "
            f"moveit_controllers.yaml no longer configures '{CONFIGURED_CONTROLLER}' "
            f"the way this function expects"
        )

    manager = dict(manager)
    manager[controller] = manager.pop(CONFIGURED_CONTROLLER)
    manager["controller_names"] = [controller]
    return {**parameters, "moveit_simple_controller_manager": manager}


def _setup(context, *args, **kwargs):
    moveit_config = build(
        sim_gazebo=LaunchConfiguration("sim_gazebo").perform(context),
        use_fake_hardware=LaunchConfiguration("use_fake_hardware").perform(context),
        ur_type=LaunchConfiguration("ur_type").perform(context),
    )
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    controller = LaunchConfiguration("trajectory_controller").perform(context)

    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                _retarget_controller(moveit_config.to_dict(), controller),
                {"use_sim_time": use_sim_time},
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("sim_gazebo", default_value="true"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("ur_type", default_value="ur3"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "trajectory_controller", default_value=CONFIGURED_CONTROLLER
            ),
            OpaqueFunction(function=_setup),
        ]
    )

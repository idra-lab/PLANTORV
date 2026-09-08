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

"""One place that assembles the MoveIt parameters.

Both launch files here need the same configuration, and getting the two
slightly out of step is the classic way to end up with RViz planning against a
different robot than move_group. They call this instead.
"""

from moveit_configs_utils import MoveItConfigsBuilder


def build(sim_gazebo: str = "true", use_fake_hardware: str = "false", ur_type: str = "ur3"):
    """Return the MoveIt configuration for the UR3 workcell.

    Parameters
    ----------
    sim_gazebo : str
        ``"true"`` when the arm is driven by gazebo_ros2_control.
    use_fake_hardware : str
        ``"true"`` to build the description against ros2_control's mock
        hardware, which is how the planner runs with no simulator at all.
    ur_type : str
        Which UR the description is built for. Everything here is dimensioned
        for the UR3; the argument exists so the same configuration can be
        pointed at a larger arm without editing.
    """
    return (
        MoveItConfigsBuilder("ur3_workcell", package_name="plantorv_moveit_config")
        .robot_description(
            mappings={
                "ur_type": ur_type,
                "sim_gazebo": sim_gazebo,
                "use_fake_hardware": use_fake_hardware,
            }
        )
        .robot_description_semantic(file_path="config/ur3_workcell.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            # The scene manager changes the world through
            # /apply_planning_scene, and the behaviour tree wants to see the
            # result, so the monitored scene is published.
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
        )
        .to_moveit_configs()
    )

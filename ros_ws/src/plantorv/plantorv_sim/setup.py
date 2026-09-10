from glob import glob

from setuptools import find_packages, setup

__maintainers__ = ["Enrico Saccon", "Tommaso Faraci"]

package_name = "plantorv_sim"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/urdf", glob("urdf/*.xacro")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Enrico Saccon",
    maintainer_email="enrico.saccon@unitn.it",
    description="Gazebo workcell for PLANTORV: a UR3 on a stand over a desk, two trays and cubes",
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "scene_manager = plantorv_sim.scene_manager:main",
            "world_model = plantorv_sim.world_model:main",
        ],
    },
)

from glob import glob

from setuptools import find_packages, setup

__maintainers__ = ["Enrico Saccon", "Davide De Martini", "Marco Roveri", "Davide Nardi"]

package_name = "plantorv_moveit_config"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", ".setup_assistant"]),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.srdf") + glob("config/*.rviz")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Enrico Saccon",
    maintainer_email="enrico.saccon@unitn.it",
    description="MoveIt 2 configuration for the PLANTORV UR3 workcell",
    license="Apache 2.0",
    entry_points={"console_scripts": []},
)

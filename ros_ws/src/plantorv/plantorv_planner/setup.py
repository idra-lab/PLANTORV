from glob import glob

from setuptools import find_packages, setup

__maintainers__ = ["Enrico Saccon", "Tommaso Faraci"]

package_name = "plantorv_planner"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Enrico Saccon",
    maintainer_email="enrico.saccon@unitn.it",
    description="Plans and executes the known manipulation actions of PLANTORV on a UR3",
    license="Apache 2.0",
    entry_points={
        "console_scripts": [
            "planner = plantorv_planner.planner_node:main",
        ],
    },
)

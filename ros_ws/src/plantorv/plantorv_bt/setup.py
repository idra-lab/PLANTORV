from glob import glob

from setuptools import find_packages, setup

__maintainers__ = ["Enrico Saccon", "Davide De Martini", "Marco Roveri", "Davide Nardi"]

package_name = "plantorv_bt"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/trees", glob("trees/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Enrico Saccon",
    maintainer_email="enrico.saccon@unitn.it",
    description="Runs BehaviorTree.CPP-style bt.xml files against the PLANTORV planner",
    license="Apache 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bt_executor = plantorv_bt.executor_node:main",
        ],
    },
)

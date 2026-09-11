from glob import glob

from setuptools import find_packages, setup

package_name = 'plantorv_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='enrico',
    maintainer_email='enrico.saccon@unitn.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgbd_capturer = plantorv_ros.rgbd_capturer:main', 
            'one_shot_pipeline = plantorv_ros.one_shot_pipeline:main',
            'charuco_tf_publisher = plantorv_ros.charuco_tf_publisher:main',
            'aruco_tf_publisher = plantorv_ros.aruco_tf_publisher:main',
            'save_marker_transforms = plantorv_ros.save_marker_transforms:main',
            'static_marker_publisher = plantorv_ros.static_marker_publisher:main'
        ],
    },
)

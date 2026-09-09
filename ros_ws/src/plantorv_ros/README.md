### Running the camera

Orbbec ROS2 repo: https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main

'''
ros2 launch orbbec_camera femto_mega.launch.py   depth_registration:=true
'''


'''
ros2 run plantorv_ros one_shot_pipeline \
    --ros-args \
    -p rgb_topic:=/camera/color/image_raw \
    -p depth_topic:=/camera/depth/image_raw \
    -p pointcloud_topic:=/camera/depth_registered/points \
    -p pipeline_script:=/home/enrico/Projects/plantorv/samgpt.py \
    -p output_dir:=/home/enrico/Projects/plantorv/output_ros
'''
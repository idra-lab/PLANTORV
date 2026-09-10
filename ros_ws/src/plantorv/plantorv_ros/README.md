## plantorv_ros

ROS 2 wrapper around the SAM/GPT pipeline: capture one synchronized
RGB-D pair plus a point cloud from the camera, write them to disk, and
run the existing `samgpt.py` pipeline on them.

Orbbec ROS 2 driver: https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main

### Running

`one_shot_pipeline.launch.py` starts the camera driver and the pipeline
node together. The pipeline node is one-shot: when it finishes, the
launch shuts the camera down as well.

Over USB, which is the default:

```
ros2 launch plantorv_ros one_shot_pipeline.launch.py \
    pipeline_script:=/home/enrico/Projects/plantorv/samgpt.py \
    output_dir:=/home/enrico/Projects/plantorv/output_ros
```

Over Ethernet:

```
ros2 launch plantorv_ros one_shot_pipeline.launch.py \
    use_network:=true \
    net_device_ip:=192.168.1.10 \
    pipeline_script:=/home/enrico/Projects/plantorv/samgpt.py \
    output_dir:=/home/enrico/Projects/plantorv/output_ros
```

`use_network` selects the transport. With `use_network:=false` the
driver enumerates USB devices only, and `serial_number` or `usb_port`
pick a specific camera when more than one is connected. With
`use_network:=true` the driver connects over Ethernet using
`net_device_ip` and `net_device_port`.

Topics are namespaced under `camera_name`, so `camera_name:=cam2` moves
`rgb_topic`, `depth_topic` and `pointcloud_topic` with it. Each of the
three can still be overridden individually.

### Camera over Ethernet

Set the host NIC to a static address on the camera's subnet (for example
`192.168.1.100/24`, mask `255.255.255.0`) and check that the camera
answers `ping 192.168.1.10`.

The launch file sets `enumerate_net_device:=false`, so the driver
connects straight to `net_device_ip`:`net_device_port` instead of
relying on broadcast discovery. That is the reliable option on a
point-to-point link. `net_device_port` must be `8090`; the driver's own
default of `0` is only valid when auto-discovery is used. To use
discovery instead, pass `enumerate_net_device:=true`.

If the camera has never been given a static address it still uses DHCP,
and `192.168.1.10` will not answer. Assign the address once with the
driver's Force IP function, which works over broadcast and so reaches
the camera even from a different subnet:

```
ros2 launch orbbec_camera femto_mega.launch.py \
    force_ip_enable:=true \
    force_ip_mac:=<camera MAC, printed on the label> \
    force_ip_address:=192.168.1.10 \
    force_ip_subnet_mask:=255.255.255.0 \
    force_ip_gateway:=192.168.1.1
```

`ros2 run orbbec_camera list_devices_node` reports the devices the SDK
can currently see, including their addresses.

### Running the parts separately

Camera only:

```
ros2 launch orbbec_camera femto_mega.launch.py \
    depth_registration:=true \
    enumerate_net_device:=false \
    net_device_ip:=192.168.1.10 \
    net_device_port:=8090
```

Pipeline only, against an already running camera:

```
ros2 run plantorv_ros one_shot_pipeline \
    --ros-args \
    -p rgb_topic:=/camera/color/image_raw \
    -p depth_topic:=/camera/depth/image_raw \
    -p pointcloud_topic:=/camera/depth_registered/points \
    -p pipeline_script:=/home/enrico/Projects/plantorv/samgpt.py \
    -p output_dir:=/home/enrico/Projects/plantorv/output_ros
```

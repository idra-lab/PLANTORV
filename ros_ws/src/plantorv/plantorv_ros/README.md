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

### ChArUco board placement aid

`charuco_viz.launch.py` starts the camera, the `charuco_tf_publisher`
node and RViz. The node estimates the pose of a ChArUco board from the
colour image and the camera's `camera_info`, and broadcasts it as a TF
transform from the colour optical frame to `board_frame`. Watching that
frame live is what tells you whether a candidate spot for the board is
seen well enough.

```
ros2 launch plantorv_ros charuco_viz.launch.py \
    squares_x:=5 \
    squares_y:=7 \
    square_length:=0.04 \
    marker_length:=0.02 \
    dictionary:=DICT_6X6_250
```

Over Ethernet:

```
ros2 launch plantorv_ros charuco_viz.launch.py \
    use_network:=true \
    net_device_ip:=192.168.1.10 \
    squares_x:=5 \
    squares_y:=7 \
    square_length:=0.04 \
    marker_length:=0.02
```

`use_network`, `net_device_ip`, `net_device_port`,
`enumerate_net_device`, `serial_number` and `usb_port` behave exactly
as in `one_shot_pipeline.launch.py`; see the two sections above for the
host-side network setup and the Force IP procedure.

`squares_x` and `squares_y` count the chessboard squares, not the
markers, and both lengths are in metres as measured on the printout.
Wrong lengths still give a pose, only at the wrong distance, so measure
the printed board rather than trusting the generator settings. Boards
generated with OpenCV older than 4.6 have their markers shifted by one
square; pass `legacy_pattern:=true` for those.

The launch also replays the recorded transforms, so the live frames
and the recorded ones stand side by side in RViz: `charuco_board` is
what the camera sees right now, `static_charuco_board` what was
recorded, and the two should sit on top of each other. Set
`use_static_transforms:=false` to leave them out, or point
`static_transforms_file:=...` somewhere else.

The `camera_to_camera` entry is deliberately not replayed here.
It publishes `camera_color_optical_frame` as a child of
`static_camera_color_optical_frame`, which is how a bringup with no
camera gets that frame; with the driver running it already has a
parent, `camera_color_frame`, and a frame cannot have two. The launch
publishes the identity bridge the other way round instead, from
`camera_color_optical_frame` to
`static_camera_color_optical_frame`, which hangs the recorded subtree
under the live camera. `static_transforms_frames` chooses what gets
replayed, by file key or by frame name.

The preloaded RViz config shows the TF tree, the board outline, two
point clouds and the annotated image.

Both clouds come from the driver, which publishes the plain one on
`/camera/depth/points` and the coloured one on
`/camera/depth_registered/points`. `use_colored_cloud` is on by
default, which turns on the coloured cloud and the D2C alignment it
needs; the plain cloud keeps being published either way, so the
`Scene cloud` display shows it coloured by height and the
`Coloured cloud` display shows the same scene in real colour. Turn one
of the two off in RViz to see the other clearly, or pass
`use_colored_cloud:=false` to skip the coloured one, which costs the
driver less.

Seeing the board frame sitting on the cloud is the second half of the
placement check: the axes should land on the surface the board rests
on, not floating above it or sunk into it.

Both cloud topics are hardcoded in the RViz config, so a non-default
`camera_name` needs the topic reset by hand in the RViz display.

The annotated image is the one to watch. It draws the board frame onto
the colour image itself: the three axes from the board origin, x red,
y green and z blue, each lettered at its tip, so the orientation is
readable without looking at the 3D view. Over them the readout gives
the corner count, the distance, the origin in metres and the
orientation as roll, pitch and yaw in degrees. `axis_length` sets the
drawn axis length in metres; it defaults to two squares.

### Where the frame sits

By default the frame sits where OpenCV puts the board origin: the outer
corner of the first square, with x along the short side of the board,
y along the long side and z out of the board towards the camera. That
point is a property of the board definition, not a choice.

`origin_corner` overrides it. The annotated image prints an `id=N`
label next to every interpolated chessboard corner, and passing that
number moves the published frame onto that corner:

```
ros2 launch plantorv_ros charuco_viz.launch.py origin_corner:=22
```

`-1`, the default, keeps the OpenCV origin. The ids run from 0 to
`(squares_x - 1) * (squares_y - 1) - 1`, so a 5x7 board has 24 corners
numbered 0 to 23, row by row from the origin; an id past the end is
rejected at startup with the valid range in the message.

Only the origin moves. The axes keep the board's own orientation,
since all the corners lie in the board plane, and the outline marker
follows the new origin. The pose is still solved from every detected
corner, so moving the origin costs no accuracy.

Read the placement off that image:

- Green text means the pose is valid; red text means no pose came out
  of that frame.
- A board that keeps losing its pose, or whose axes jitter or flip, is
  too far, too oblique or too poorly lit for the spot it is in.
- More detected corners is better. Below `min_corners` (6 by default)
  the node publishes nothing instead of a jumping frame.
- z should point out of the board towards the camera. If it points
  away, or the origin sits on the wrong corner, the board layout does
  not match `legacy_pattern`.
- The yellow quad is the board as the parameters describe it, drawn
  from the OpenCV origin whatever `origin_corner` says. It must sit on
  the printed board's own edges. If it is the wrong size, or shifted,
  then `squares_x`, `squares_y` or `square_length` do not match the
  printout, and the pose is being solved against the wrong geometry.
  That is what puts the origin somewhere inside the board instead of on
  a corner.
- `error` is the RMS reprojection error of the detected corners. A
  couple of tenths of a pixel is normal. Whole pixels mean the board
  description or the camera calibration is off, not that the board is
  badly placed.

### Which frame the board is published in

The node reads the frame the driver stamps on the colour image and
publishes the board as a child of it, so no frame name is configured
here. The Orbbec driver stamps `camera_color_optical_frame` on
`/camera/color/image_raw`, which is the frame the RViz config uses as
its fixed frame. The first frame the node processes is logged with the
parent it resolved:

```
Publishing charuco_board as a child of 'camera_color_optical_frame',
the frame the driver stamps on /camera/color/image_raw.
```

If that line names something else, the driver was started with a
different `camera_name` or frame configuration, and the RViz fixed
frame has to follow it.

The node also checks the detected marker ids against the configured
board and reports a mismatch once:

```
Detected marker id 43, but the configured board holds only 17 markers,
ids 0 to 16.
```

That means `squares_x`, `squares_y` or `dictionary` describe a smaller
board than the printed one.

Node only, against an already running camera:

```
ros2 run plantorv_ros charuco_tf_publisher \
    --ros-args \
    -p rgb_topic:=/camera/color/image_raw \
    -p camera_info_topic:=/camera/color/camera_info \
    -p squares_x:=5 \
    -p squares_y:=7 \
    -p square_length:=0.04 \
    -p marker_length:=0.02
```

It publishes `/charuco_tf_publisher/debug_image` and
`/charuco_tf_publisher/board_markers`; set `publish_debug_image` or
`publish_markers` to `false` to drop either.

The node works with both ChArUco APIs and picks one at startup, since
which one is available depends on the interpreter: `ros2 run` uses
`/usr/bin/python3`, where ROS 2 Humble provides OpenCV 4.5, while the
project virtualenv has 4.10. OpenCV 4.5 implements the pre-4.6 marker
layout only, so a board generated by a newer OpenCV needs the
virtualenv interpreter:

```
.venv/bin/python3 -m plantorv_ros.charuco_tf_publisher --ros-args ...
```

The startup log line reports which API and OpenCV version are in use.

### Single ArUco markers

`aruco_tf_publisher` publishes a frame per single ArUco marker, which
is how the robot base gets one. `charuco_viz.launch.py` starts it
alongside the board node; pass `use_aruco:=false` to leave it out.

```
ros2 launch plantorv_ros charuco_viz.launch.py \
    aruco_marker_length:=0.10 \
    aruco_dictionary:=DICT_7X7_50 \
    aruco_marker_ids:='[0]' \
    aruco_marker_frames:="['robot_base_marker']"
```

Those are also the defaults, so a 10 cm 7x7 marker with id 0 needs no
arguments at all. Id 0 is byte-identical in `DICT_7X7_50`, `_100`,
`_250` and `_1000`, so the default reads a marker printed from any of
the four; the smallest dictionary is the default because it has the
fewest candidates to confuse the detector with.

`aruco_marker_length` is the side of the black square in metres, the
printed border included but not the white margin around it. Measure
the printout: 10 cm of paper is not 10 cm of marker if the generator
added a quiet zone.

`aruco_marker_ids` and `aruco_marker_frames` line up entry by entry, so
several markers can be published at once:

```
ros2 launch plantorv_ros charuco_viz.launch.py \
    aruco_marker_ids:='[0, 4]' \
    aruco_marker_frames:="['robot_base_marker', 'gripper_marker']"
```

Any id left unnamed becomes `aruco_<id>`. Markers of the dictionary
that are not in `aruco_marker_ids` are detected and drawn, but no frame
is published for them.

OpenCV puts a marker's origin at its centre, with x right along the top
edge, y up and z out of the marker towards the camera. That is the
marker's frame, not the robot's: the transform from the marker to the
robot base belongs to the robot description, and
`static_transform_publisher` is the quick way to add it while
measuring.

```
ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 \
    --frame-id robot_base_marker --child-frame-id robot_base
```

A single marker is a four-point planar target, so its pose is weaker
than the board's: it flips between two solutions when seen at an
oblique angle and jitters more with distance. IPPE_SQUARE picks the
better solution and a VVS pass refines it, but a marker seen edge-on is
still a bad frame. `aruco_max_reprojection_error`, 4 px by default,
drops a pose that does not fit its own corners; set it to `0.0` to
publish whatever the solver returns.

Its annotated image is `/aruco_tf_publisher/debug_image`, listing each
published marker with its distance, reprojection error, position and
orientation, and the outlines go to
`/aruco_tf_publisher/marker_outlines`. The RViz config carries a
display for each.

### Recording the frames once

The detector nodes re-solve their frames on every image, which is what
you want while placing things. Once the board and the robot are where
they will stay, record the frames once and replay them at every
bringup: the camera then does not have to see the markers again, and
the frames stop jittering.

Place everything first and check it in `charuco_viz.launch.py`. Then
record:

```
ros2 launch plantorv_ros save_marker_transforms.launch.py
```

That starts the camera and both detectors, averages 30 samples of
`charuco_board` and `robot_base_marker` in
`camera_color_optical_frame`, writes the transform file and shuts the
whole launch down again.

The file it writes, and the one the replay reads, is
`plantorv_bringup`'s `config/static_transforms.yaml`, found through
`get_package_share_directory` rather than a hardcoded path. The
workspace installs with symlinks, so the installed share path leads
back to the source tree and a recording lands on the copy under
version control. Pass `output_file:=...` to write somewhere else.

The first `settle_time` seconds, two by default, are thrown away: the
camera starts with auto exposure and auto white balance still moving,
and those frames detect worse than the ones after them. Of what
follows, samples more than `outlier_sigma` (2.5) standard deviations
out are dropped, so one frame with a missed corner does not drag the
average. The translation is a median and the rotation an eigenvector
average, both robust to a bad sample the rejection missed.

What survives has to hold still, or the file is not written:

```
'charuco_board' moved 196.0 mm and 180.00 deg while sampling, over the
12.0 mm / 3.0 deg limit. Nothing was written.
```

The limits are loose enough for a single ArUco marker a metre out,
which really does wander a few millimetres and a degree or two on a
USB 2 connection, and for a board, which is steadier. They are tight
enough to catch the failures that matter. A spread of tens of
centimetres and 180 degrees means two nodes are publishing the same
frame: look for a detector, or a `static_marker_publisher`, left
running from an earlier session, since a latched static transform
outlives the launch that started it. Loosen the gate with
`max_translation_spread` and `max_rotation_spread`, or skip it with
`force:=true`.

The file is plain YAML and can be edited by hand:

```yaml
parent_frame: camera_color_optical_frame
recorded_at: '2026-09-10T17:17:44+00:00'
transforms:
  charuco_board:
    parent_frame: camera_color_optical_frame
    child_frame: charuco_board
    translation: {x: -0.100005, y: -0.140005, z: 0.285714}
    rotation: {x: 0.0, y: -0.0, z: 0.0, w: 1.0}
    samples: 20
    dropped_samples: 0
    distance_m: 0.32
    translation_spread_m: 1.2e-16
    rotation_spread_deg: 0.0
```

Everything below `rotation` is there to be read later, when you want to
know how good a recording was: how many samples it kept, how many it
threw away, how far the frame was and how much it wandered. All of it
is ignored on replay.

### Replaying them at bringup

```
ros2 launch plantorv_ros static_marker_transforms.launch.py
```

This publishes every transform in
`plantorv_bringup`'s `config/static_transforms.yaml` on `/tf_static`
and nothing else: no camera, no detection, no OpenCV. Include it from
a bringup launch to have the frames available from the start.

```python
IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare('plantorv_ros'),
            'launch',
            'static_marker_transforms.launch.py',
        ])
    ),
    launch_arguments={'input_file': '/path/to/marker_transforms.yaml'}.items(),
)
```

`frames:="['charuco_board']"` publishes only some of them, which is
useful when the robot frame comes from the robot description instead.

A recorded frame is only as true as the setup it was recorded in.
Moving the camera or the robot invalidates the file and nothing can
detect that from the file alone, so record it again after either one
moves.

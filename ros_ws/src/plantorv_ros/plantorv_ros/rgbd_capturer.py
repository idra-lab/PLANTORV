#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RGBDPointCloudNode(Node):
    def __init__(self):
        super().__init__('rgbd_pointcloud_node')

        # RGB + depth subscribers used by message_filters
        self.rgb_sub = Subscriber(
            self, Image,
            '/camera/color/image_raw',
            qos_profile=qos_profile_sensor_data
        )

        self.depth_sub = Subscriber(
            self, Image,
            '/camera/depth/image_raw',
            qos_profile=qos_profile_sensor_data
        )

        # Synchronize RGB and depth within 20 ms
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.02
        )
        self.sync.registerCallback(self.rgbd_callback)

        # Point cloud is intentionally NOT synchronized
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.pointcloud_callback,
            qos_profile_sensor_data
        )

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pointcloud = None

    def rgbd_callback(self, rgb_msg, depth_msg):
        """Called whenever a matching RGB/depth pair is found."""
        self.latest_rgb = rgb_msg
        self.latest_depth = depth_msg

        t_rgb = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        t_depth = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9

        self.get_logger().info(
            f'RGB-D pair: {t_rgb:.6f}, {t_depth:.6f} '
            f'(dt={abs(t_rgb - t_depth)*1000:.2f} ms)'
        )

        # Process/save your synchronized RGB + depth here.

    def pointcloud_callback(self, cloud_msg):
        """Called independently for every incoming point cloud."""
        self.latest_pointcloud = cloud_msg

        t_pcd = cloud_msg.header.stamp.sec + cloud_msg.header.stamp.nanosec * 1e-9

        # Process/save the streamed PointCloud2 here.
        self.get_logger().info(
            f'Received PCD at time: {t_pcd:.6}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = RGBDPointCloudNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
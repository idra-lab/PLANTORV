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

__maintainers__ = ["Enrico Saccon", "Tommaso Faraci"]

"""Loads a ``bt.xml`` and ticks it.

    ros2 run plantorv_bt bt_executor --ros-args -p tree_file:=/path/to/bt.xml

The tree is loaded once, at start-up, so a malformed file is an error before
the robot moves. Ticking happens on a timer: each tick walks the tree and
returns, and the leaves that are waiting on the planner return RUNNING rather
than blocking, which is what lets the same node also receive the planner's
replies.

The run can be driven from outside with two services, ``~/start`` and
``~/stop``, so that a tree can be re-run without restarting anything.
"""

import os

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from plantorv_bt.core import Blackboard, Status
from plantorv_bt.factory import BehaviorTreeFactory, TreeBuildError
from plantorv_bt.leaves import LEAVES, RosBridge


class BehaviorTreeExecutor(Node):
    """Ticks one behaviour tree against the planner and the world model."""

    def __init__(self):
        super().__init__("bt_executor")

        self.declare_parameter("tree_file", "")
        self.declare_parameter("tick_rate", 10.0)
        self.declare_parameter("autostart", True)
        # Wait for the planner and the world model before the first tick.
        # Without it the first Pick fails simply because move_group was slow.
        self.declare_parameter("wait_for_services", 120.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("shutdown_when_done", False)

        # How long a leaf that previews its target waits before acting,
        # so the wait can be set for a run without editing the tree. A
        # tree that says preview_seconds itself still wins.
        self.declare_parameter("preview_seconds", 120.0)

        tree_file = self.get_parameter("tree_file").value
        if not tree_file or not os.path.exists(tree_file):
            raise SystemExit(
                f"tree_file '{tree_file}' does not exist; pass one with "
                "--ros-args -p tree_file:=<path to bt.xml>"
            )

        self.bridge = RosBridge(self)
        self.bridge.preview_seconds = float(
            self.get_parameter("preview_seconds").value
        )
        self.factory = BehaviorTreeFactory(context=self.bridge)
        for tag, leaf in LEAVES.items():
            self.factory.register(tag, leaf)

        self.blackboard = Blackboard({"held_object": ""})
        try:
            self.tree = self.factory.create_from_file(tree_file, self.blackboard)
        except TreeBuildError as error:
            raise SystemExit(f"{error}")
        self.get_logger().info(f"loaded {tree_file}")

        self.running = False
        self.ticks = 0
        self.create_service(Trigger, "~/start", self._on_start)
        self.create_service(Trigger, "~/stop", self._on_stop)

        period = 1.0 / max(0.1, float(self.get_parameter("tick_rate").value))
        self.timer = self.create_timer(period, self._tick)

        if self.get_parameter("autostart").value:
            # Waiting has to happen off the executor thread, or the services
            # being waited for could never be discovered.
            import threading

            threading.Thread(target=self._start_when_ready, daemon=True).start()

    # -- running ---------------------------------------------------------

    def _start_when_ready(self) -> None:
        timeout = float(self.get_parameter("wait_for_services").value)
        if not self.bridge.wait_until_ready(timeout):
            self.get_logger().error(
                "the planner or the world model never appeared; start the tree "
                "by hand with the ~/start service once they are up"
            )
            return
        self.get_logger().info("planner and world model are up, starting the tree")
        self._start()

    def _start(self) -> None:
        self.tree.halt()
        self.ticks = 0
        self.running = True

    def _tick(self) -> None:
        if not self.running:
            return
        self.ticks += 1
        status = self.tree.tick()
        if status is Status.RUNNING:
            return

        self.running = False
        self.get_logger().info(f"tree finished: {status.value} after {self.ticks} ticks")
        self.tree.halt()

        if self.get_parameter("loop").value:
            self._start()
        elif self.get_parameter("shutdown_when_done").value:
            raise SystemExit(0 if status is Status.SUCCESS else 1)

    # -- services --------------------------------------------------------

    def _on_start(self, request, response):
        self._start()
        response.success = True
        response.message = "ticking"
        return response

    def _on_stop(self, request, response):
        self.running = False
        self.tree.halt()
        response.success = True
        response.message = "stopped, and any goal in flight cancelled"
        return response


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BehaviorTreeExecutor()
    except SystemExit as error:
        print(f"bt_executor: {error}")
        rclpy.shutdown()
        raise
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

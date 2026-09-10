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

"""Whichever ros2_control controller the next move needs.

Two controllers want the same command interfaces, and ros2_control gives them
to one at a time: ``joint_trajectory_controller`` for a joint-space move, which
is what ``home`` is and what MoveIt executes through, and
``cartesian_motion_controller`` for everything else. So every motion begins by
saying which of the two it needs, and this switches if that is not the one
already running.

Switching is cheap -- it lands on one control cycle of the manager -- but it is
not free of consequence: the Cartesian controller starts from wherever the arm
is (it sets its target to the measured pose on activation), while the
trajectory controller starts from whatever trajectory it is given. Neither
jumps, which is the only property that matters here.

``ensure`` remembers what it last asked for, so a run of Cartesian moves costs
one service call rather than one per move. It deliberately does not trust that
memory across a failure: a switch that does not report success clears it, and
the next move asks again.

When it does have to switch, it asks the manager what is running first, and
requests only the transitions that are actually needed. That is not an
optimisation. A switch is strict, and a strict switch that names a controller
already in the state it is being asked for is an error -- "Controller with name
'x' can not be deactivated since it is not active". The Cartesian controller is
spawned inactive, so a blind "activate the trajectory controller, deactivate
the other one" fails on the first ``home`` of every run, which is the first
node of every tree. Asking first is what makes it work at all.
"""

import time
from typing import Dict, List, Optional

from controller_manager_msgs.srv import ListControllers, SwitchController

from plantorv_planner.errors import PlanningError

#: The lifecycle state a controller is in when it holds the joints.
ACTIVE = "active"


class ControllerSwitcher:
    """Activates one of a set of mutually exclusive controllers.

    Parameters
    ----------
    node : rclpy.node.Node
        The node the service client is created on.
    known : sequence of str
        Every controller this may switch between. The ones not being activated
        are deactivated, which is what makes them mutually exclusive.
    callback_group : rclpy.callback_groups.CallbackGroup
        Must be reentrant: ``ensure`` blocks the caller while the manager
        answers, and the answer arrives on a callback.
    timeout : float
        Seconds to wait for the manager, per call.
    """

    def __init__(self, node, known, callback_group, timeout: float = 10.0):
        self.node = node
        self.known = list(known)
        self.timeout = timeout
        self.active: Optional[str] = None
        self.client = node.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
            callback_group=callback_group,
        )
        self.list_client = node.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
            callback_group=callback_group,
        )

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Block until the controller manager is there."""
        return self.client.wait_for_service(
            timeout_sec=timeout
        ) and self.list_client.wait_for_service(timeout_sec=timeout)

    def states(self) -> Dict[str, str]:
        """Lifecycle state of every controller the manager has loaded."""
        response = self._call(self.list_client, ListControllers.Request())
        if response is None:
            raise PlanningError("the controller manager did not answer list_controllers")
        return {entry.name: entry.state.lower() for entry in response.controller}

    def ensure(self, name: str) -> None:
        """Make ``name`` the active controller. Raises if it cannot."""
        if name not in self.known:
            raise PlanningError(
                f"'{name}' is not one of the controllers this planner switches between "
                f"({', '.join(self.known)})"
            )
        if self.active == name:
            return

        states = self.states()
        missing = [controller for controller in self.known if controller not in states]
        if missing:
            raise PlanningError(
                f"the controller manager has not loaded {', '.join(missing)}; "
                f"it knows {', '.join(sorted(states)) or 'nothing'}"
            )

        activate: List[str] = [] if states[name] == ACTIVE else [name]
        deactivate: List[str] = [
            controller
            for controller in self.known
            if controller != name and states[controller] == ACTIVE
        ]
        if not activate and not deactivate:
            # Already the only one of ours holding the joints. Nothing to ask
            # for -- and asking anyway would be the error described above.
            self.active = name
            return

        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        # STRICT: a move must not run because a switch half worked. Better to
        # abort the goal and say so than to command an arm whose controller is
        # not the one the motion was written for. Safe to be strict only
        # because every transition asked for above is one the manager can make.
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True

        # Forget first. If the call raises or times out we do not know what the
        # manager did, and assuming the old controller is still active is the
        # one assumption that could command the wrong one silently.
        self.active = None
        response = self._call(self.client, request)
        if response is None:
            raise PlanningError(f"the controller manager did not answer a switch to '{name}'")
        if not response.ok:
            raise PlanningError(
                f"the controller manager refused to activate '{name}' "
                f"(deactivating {', '.join(deactivate) or 'nothing'})"
            )

        self.active = name
        self.node.get_logger().info(f"controller: {name}")

    def _call(self, client, request):
        if not client.service_is_ready() and not client.wait_for_service(
            timeout_sec=self.timeout
        ):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + self.timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        return future.result() if future.done() else None

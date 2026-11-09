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

"""What the planner asks the controller manager for, and what it does not ask.

A strict switch that names a controller already in the state it is asked for is
refused, not ignored. The Cartesian controller is spawned inactive, so the
obvious implementation -- activate one, deactivate the rest -- fails on the
first `home` of every run, and the tree dies before the arm moves. These tests
are about the request, because the request is where that goes wrong.
"""

import pytest

from plantorv_planner.controllers import ControllerSwitcher
from plantorv_planner.errors import PlanningError

TRAJECTORY = "joint_trajectory_controller"
CARTESIAN = "cartesian_motion_controller"


class FakeLogger:
    def info(self, *args):
        pass


class FakeNode:
    """Just enough node for the switcher's constructor."""

    def create_client(self, *args, **kwargs):
        return None

    def get_logger(self):
        return FakeLogger()


@pytest.fixture
def switcher(monkeypatch):
    """A switcher whose service calls are recorded instead of sent."""
    made = ControllerSwitcher(
        FakeNode(), known=[TRAJECTORY, CARTESIAN], callback_group=None
    )
    made.requests = []
    made.reported = {}

    def states():
        return dict(made.reported)

    def call(client, request):
        made.requests.append(request)

        class Ok:
            ok = True

        return Ok()

    monkeypatch.setattr(made, "states", states)
    monkeypatch.setattr(made, "_call", call)
    return made


def test_activating_the_trajectory_controller_does_not_deactivate_an_inactive_one(switcher):
    """The bug this file exists for: the first `home` of every run."""
    switcher.reported = {TRAJECTORY: "inactive", CARTESIAN: "inactive"}
    switcher.ensure(TRAJECTORY)
    assert len(switcher.requests) == 1
    assert switcher.requests[0].activate_controllers == [TRAJECTORY]
    assert switcher.requests[0].deactivate_controllers == []


def test_switching_away_from_an_active_controller_deactivates_it(switcher):
    switcher.reported = {TRAJECTORY: "active", CARTESIAN: "inactive"}
    switcher.ensure(CARTESIAN)
    assert switcher.requests[0].activate_controllers == [CARTESIAN]
    assert switcher.requests[0].deactivate_controllers == [TRAJECTORY]


def test_asking_for_the_one_already_running_asks_the_manager_for_nothing(switcher):
    switcher.reported = {TRAJECTORY: "inactive", CARTESIAN: "active"}
    switcher.ensure(CARTESIAN)
    assert switcher.requests == []
    assert switcher.active == CARTESIAN


def test_the_second_move_in_a_row_costs_no_switch(switcher):
    switcher.reported = {TRAJECTORY: "active", CARTESIAN: "inactive"}
    switcher.ensure(CARTESIAN)
    switcher.ensure(CARTESIAN)
    assert len(switcher.requests) == 1


def test_a_refused_switch_is_not_remembered_as_done(switcher, monkeypatch):
    # Otherwise the next move would skip the switch and command whichever
    # controller actually holds the joints.
    switcher.reported = {TRAJECTORY: "active", CARTESIAN: "inactive"}

    class Refused:
        ok = False

    monkeypatch.setattr(switcher, "_call", lambda client, request: Refused())
    with pytest.raises(PlanningError):
        switcher.ensure(CARTESIAN)
    assert switcher.active is None


def test_a_silent_manager_is_not_remembered_as_done(switcher, monkeypatch):
    switcher.reported = {TRAJECTORY: "active", CARTESIAN: "inactive"}
    monkeypatch.setattr(switcher, "_call", lambda client, request: None)
    with pytest.raises(PlanningError):
        switcher.ensure(CARTESIAN)
    assert switcher.active is None


def test_a_controller_the_manager_never_loaded_is_reported_as_such(switcher):
    # What a missing cartesian_controllers build looks like, rather than a
    # timeout somewhere further in.
    switcher.reported = {TRAJECTORY: "active"}
    with pytest.raises(PlanningError, match=CARTESIAN):
        switcher.ensure(CARTESIAN)


def test_a_controller_outside_the_known_set_is_refused(switcher):
    switcher.reported = {TRAJECTORY: "active", CARTESIAN: "inactive"}
    with pytest.raises(PlanningError):
        switcher.ensure("some_other_controller")

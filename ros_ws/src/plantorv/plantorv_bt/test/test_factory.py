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

"""The tree semantics, checked without a robot.

Neither plantorv_bt.core nor plantorv_bt.factory imports anything from ROS, so
the control flow that decides what the arm does can be tested at the speed of a
unit test rather than of a simulator. The leaves here are counters.
"""

import os
import textwrap

import pytest

from plantorv_bt.core import Blackboard, Status, TreeNode
from plantorv_bt.factory import BehaviorTreeFactory, TreeBuildError


class Scripted(TreeNode):
    """A leaf that returns whatever the test told it to, and counts ticks."""

    RESULTS = {}
    TICKS = {}

    def _tick(self) -> Status:
        key = self.attributes.get("id", self.name)
        Scripted.TICKS[key] = Scripted.TICKS.get(key, 0) + 1
        outcome = Scripted.RESULTS.get(key, [Status.SUCCESS])
        return outcome[min(Scripted.TICKS[key] - 1, len(outcome) - 1)]


@pytest.fixture(autouse=True)
def _reset():
    Scripted.RESULTS = {}
    Scripted.TICKS = {}


def build(tmp_path, xml, blackboard=None):
    path = os.path.join(tmp_path, "bt.xml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(xml))
    factory = BehaviorTreeFactory()
    factory.register("Task", Scripted)
    return factory.create_from_file(path, blackboard or Blackboard())


def test_sequence_runs_children_in_order(tmp_path):
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <Sequence>
              <Task id="a"/>
              <Task id="b"/>
            </Sequence>
          </BehaviorTree>
        </root>
        """,
    )
    assert tree.tick() is Status.SUCCESS
    assert Scripted.TICKS == {"a": 1, "b": 1}


def test_sequence_resumes_at_the_running_child(tmp_path):
    """The property the whole design rests on: a long motion is not restarted."""
    Scripted.RESULTS = {"b": [Status.RUNNING, Status.RUNNING, Status.SUCCESS]}
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <Sequence>
              <Task id="a"/>
              <Task id="b"/>
            </Sequence>
          </BehaviorTree>
        </root>
        """,
    )
    assert tree.tick() is Status.RUNNING
    assert tree.tick() is Status.RUNNING
    assert tree.tick() is Status.SUCCESS
    # 'a' succeeded once and was not ticked again while 'b' ran.
    assert Scripted.TICKS["a"] == 1
    assert Scripted.TICKS["b"] == 3


def test_reactive_sequence_rechecks_its_conditions(tmp_path):
    Scripted.RESULTS = {"b": [Status.RUNNING, Status.RUNNING, Status.SUCCESS]}
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <ReactiveSequence>
              <Task id="a"/>
              <Task id="b"/>
            </ReactiveSequence>
          </BehaviorTree>
        </root>
        """,
    )
    tree.tick()
    tree.tick()
    assert Scripted.TICKS["a"] == 2


def test_fallback_takes_the_second_branch(tmp_path):
    Scripted.RESULTS = {"a": [Status.FAILURE]}
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <Fallback>
              <Task id="a"/>
              <Task id="b"/>
            </Fallback>
          </BehaviorTree>
        </root>
        """,
    )
    assert tree.tick() is Status.SUCCESS
    assert Scripted.TICKS == {"a": 1, "b": 1}


def test_retry_gives_up_after_the_stated_number_of_attempts(tmp_path):
    Scripted.RESULTS = {"a": [Status.FAILURE]}
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <RetryUntilSuccessful num_attempts="3">
              <Task id="a"/>
            </RetryUntilSuccessful>
          </BehaviorTree>
        </root>
        """,
    )
    assert tree.tick() is Status.RUNNING
    assert tree.tick() is Status.RUNNING
    assert tree.tick() is Status.FAILURE
    assert Scripted.TICKS["a"] == 3


def test_blackboard_ports_are_read_and_written(tmp_path):
    blackboard = Blackboard()
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <Sequence>
              <SetBlackboard output_key="cubes" value="cube_1"/>
              <Task id="a"/>
            </Sequence>
          </BehaviorTree>
        </root>
        """,
        blackboard,
    )
    tree.tick()
    assert blackboard.get("cubes") == "cube_1"


def test_pop_from_list_walks_a_list_then_fails(tmp_path):
    blackboard = Blackboard({"cubes": ["cube_1", "cube_2"]})
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <PopFromList list="{cubes}" output_key="{cube}"/>
          </BehaviorTree>
        </root>
        """,
        blackboard,
    )
    assert tree.tick() is Status.SUCCESS
    assert blackboard.get("cube") == "cube_1"
    assert tree.tick() is Status.SUCCESS
    assert blackboard.get("cube") == "cube_2"
    assert tree.tick() is Status.FAILURE


def test_subtrees_are_expanded(tmp_path):
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <Sequence>
              <SubTree ID="Inner"/>
            </Sequence>
          </BehaviorTree>
          <BehaviorTree ID="Inner">
            <Task id="a"/>
          </BehaviorTree>
        </root>
        """,
    )
    assert tree.tick() is Status.SUCCESS
    assert Scripted.TICKS["a"] == 1


def test_parallel_ticks_every_child_and_waits_for_them(tmp_path):
    Scripted.RESULTS = {"b": [Status.RUNNING, Status.SUCCESS]}
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <Parallel success_count="2" failure_count="1">
              <Task id="a"/>
              <Task id="b"/>
            </Parallel>
          </BehaviorTree>
        </root>
        """,
    )
    assert tree.tick() is Status.RUNNING
    assert Scripted.TICKS == {"a": 1, "b": 1}
    assert tree.tick() is Status.SUCCESS
    # 'a' finished on the first tick and is not ticked again.
    assert Scripted.TICKS == {"a": 1, "b": 2}


def test_keep_running_until_failure_repeats_a_succeeding_child(tmp_path):
    """The loop the sorting tree is built on: repeat until the list runs dry."""
    Scripted.RESULTS = {"a": [Status.SUCCESS, Status.SUCCESS, Status.FAILURE]}
    tree = build(
        tmp_path,
        """\
        <root BTCPP_format="4" main_tree_to_execute="MainTree">
          <BehaviorTree ID="MainTree">
            <KeepRunningUntilFailure>
              <Task id="a"/>
            </KeepRunningUntilFailure>
          </BehaviorTree>
        </root>
        """,
    )
    assert tree.tick() is Status.RUNNING
    assert tree.tick() is Status.RUNNING
    assert tree.tick() is Status.FAILURE


def test_an_unknown_node_is_refused_at_load_time(tmp_path):
    with pytest.raises(TreeBuildError, match="unknown node <Fly>"):
        build(
            tmp_path,
            """\
            <root BTCPP_format="4" main_tree_to_execute="MainTree">
              <BehaviorTree ID="MainTree">
                <Fly/>
              </BehaviorTree>
            </root>
            """,
        )


def test_recursive_subtrees_are_refused(tmp_path):
    with pytest.raises(TreeBuildError, match="recursion"):
        build(
            tmp_path,
            """\
            <root BTCPP_format="4" main_tree_to_execute="MainTree">
              <BehaviorTree ID="MainTree">
                <SubTree ID="MainTree"/>
              </BehaviorTree>
            </root>
            """,
        )


def test_the_shipped_trees_load(tmp_path):
    """The example trees are only useful if they are also valid."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    factory = BehaviorTreeFactory()
    for tag in ("Home", "MoveTo", "Pick", "Place", "DetectObjects", "MatchingTray", "Log"):
        factory.register(tag, Scripted)
    for name in ("pick_and_place.xml", "sort_cubes.xml"):
        factory.create_from_file(os.path.join(here, "trees", name), Blackboard())

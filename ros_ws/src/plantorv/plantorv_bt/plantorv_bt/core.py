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

__maintainers__ = ["Enrico Saccon", "Davide De Martini", "Marco Roveri", "Davide Nardi"]

"""Behaviour tree semantics: statuses, the blackboard, control flow.

A subset of BehaviorTree.CPP 4, in Python, so that the trees written for this
project are ordinary ``bt.xml`` files -- readable in Groot, and portable to the
C++ library if the tree ever needs to run there.

What is implemented is the part that a manipulation tree uses: the two memory
control nodes and their reactive variants, the common decorators, and leaves
that take time. What is not: scripting expressions, pre- and post-conditions,
and ports with types beyond strings and numbers. An unknown tag is an error at
load time rather than something silently ignored, so a tree that references a
node this executor does not have fails when it is read, not halfway through a
run.

The one semantic worth stating plainly is RUNNING. A leaf that starts a robot
motion returns RUNNING every tick until the motion ends. ``Sequence`` remembers
which child that was and resumes there; ``ReactiveSequence`` re-ticks its
earlier children every tick and halts the running one if any of them stops
holding. That difference is the whole reason to use a behaviour tree, so both
are here.
"""

import time
from enum import Enum
from typing import Any, Dict, List, Optional


class Status(Enum):
    """The result of a tick."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

    @property
    def is_finished(self) -> bool:
        return self is not Status.RUNNING


class Blackboard:
    """The tree's shared key-value store.

    Ports written as ``{name}`` in the XML read from and write to this.
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._values: Dict[str, Any] = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._values)


class TreeNode:
    """Base of every node in the tree.

    Parameters
    ----------
    name : str
        The XML tag, or the ``name`` attribute when one is given. Only used in
        logs.
    attributes : dict
        The XML attributes, verbatim. Ports are read out of these.
    blackboard : Blackboard
        Shared with every other node in the tree.
    children : list[TreeNode]
        Empty for leaves.
    context : object
        Whatever the executor needs to hand its leaves -- here, the ROS node.
    """

    def __init__(self, name, attributes, blackboard, children=None, context=None):
        self.name = name
        self.attributes = dict(attributes)
        self.blackboard = blackboard
        self.children: List[TreeNode] = list(children or [])
        self.context = context
        self.status = Status.FAILURE

    # -- ports -----------------------------------------------------------

    def port(self, key: str, default: Any = None, cast=None) -> Any:
        """Read an input port.

        ``target="cube_1"`` is the literal string; ``target="{held}"`` is
        whatever the blackboard has under ``held``.
        """
        raw = self.attributes.get(key)
        if raw is None:
            return default
        value = raw.strip()
        if value.startswith("{") and value.endswith("}"):
            value = self.blackboard.get(value[1:-1].strip(), default)
        if value is None:
            return default
        if cast is None:
            return value
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    def write_port(self, key: str, value: Any) -> None:
        """Write an output port, if the XML pointed it at a blackboard key."""
        raw = self.attributes.get(key, "").strip()
        if raw.startswith("{") and raw.endswith("}"):
            self.blackboard.set(raw[1:-1].strip(), value)

    # -- lifecycle -------------------------------------------------------

    def tick(self) -> Status:
        """Tick the node and remember the result.

        Subclasses override :meth:`_tick`; this wrapper is what keeps
        ``status`` current, which is what ``halt_children`` reads to decide
        whether a child needs halting.
        """
        self.status = self._tick()
        return self.status

    def _tick(self) -> Status:
        raise NotImplementedError

    def halt(self) -> None:
        """Stop this node and everything under it."""
        for child in self.children:
            child.halt()
        self.status = Status.FAILURE

    def halt_children(self, from_index: int = 0) -> None:
        for child in self.children[from_index:]:
            if child.status is Status.RUNNING:
                child.halt()

    def describe(self) -> str:
        target = self.attributes.get("target") or self.attributes.get("object") or ""
        return f"{self.name}({target})" if target else self.name


# -- control nodes -------------------------------------------------------


class Sequence(TreeNode):
    """Children in order, until one fails. Remembers where it got to."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = 0

    def _tick(self) -> Status:
        while self.index < len(self.children):
            status = self.children[self.index].tick()
            if status is Status.RUNNING:
                return Status.RUNNING
            if status is Status.FAILURE:
                self.halt()
                return Status.FAILURE
            self.index += 1
        self.halt()
        return Status.SUCCESS

    def halt(self) -> None:
        self.halt_children()
        self.index = 0


class ReactiveSequence(TreeNode):
    """Children in order, from the first, every tick.

    The point of the re-tick is that the earlier children are conditions: if
    one of them stops holding while a later child is running, the running child
    is halted.
    """

    def _tick(self) -> Status:
        for index, child in enumerate(self.children):
            status = child.tick()
            if status is Status.RUNNING:
                self.halt_children(index + 1)
                return Status.RUNNING
            if status is Status.FAILURE:
                self.halt()
                return Status.FAILURE
        self.halt()
        return Status.SUCCESS


class Fallback(TreeNode):
    """Children in order, until one succeeds. Remembers where it got to."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = 0

    def _tick(self) -> Status:
        while self.index < len(self.children):
            status = self.children[self.index].tick()
            if status is Status.RUNNING:
                return Status.RUNNING
            if status is Status.SUCCESS:
                self.halt()
                return Status.SUCCESS
            self.index += 1
        self.halt()
        return Status.FAILURE

    def halt(self) -> None:
        self.halt_children()
        self.index = 0


class ReactiveFallback(TreeNode):
    """Children in order, from the first, every tick, until one succeeds."""

    def _tick(self) -> Status:
        for index, child in enumerate(self.children):
            status = child.tick()
            if status is Status.RUNNING:
                self.halt_children(index + 1)
                return Status.RUNNING
            if status is Status.SUCCESS:
                self.halt()
                return Status.SUCCESS
        self.halt()
        return Status.FAILURE


class Parallel(TreeNode):
    """All children each tick; succeeds once ``success_count`` of them do.

    A child that has finished is not ticked again until the whole node is
    halted, which is why the results are recorded here rather than read back
    off the children: ``status`` only says what the last tick returned, not
    whether there was one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finished: Dict[int, Status] = {}

    def _tick(self) -> Status:
        wanted = int(self.port("success_count", len(self.children), cast=int))
        tolerated = int(self.port("failure_count", 1, cast=int))

        for index, child in enumerate(self.children):
            if index in self.finished:
                continue
            status = child.tick()
            if status.is_finished:
                self.finished[index] = status

        succeeded = sum(1 for s in self.finished.values() if s is Status.SUCCESS)
        failed = sum(1 for s in self.finished.values() if s is Status.FAILURE)
        if succeeded >= wanted:
            self.halt()
            return Status.SUCCESS
        if failed >= tolerated:
            self.halt()
            return Status.FAILURE
        return Status.RUNNING

    def halt(self) -> None:
        self.finished.clear()
        super().halt()


# -- decorators ----------------------------------------------------------


class Decorator(TreeNode):
    """One child, and something done to its result."""

    @property
    def child(self) -> TreeNode:
        return self.children[0]


class Inverter(Decorator):
    def _tick(self) -> Status:
        status = self.child.tick()
        if status is Status.SUCCESS:
            return Status.FAILURE
        if status is Status.FAILURE:
            return Status.SUCCESS
        return status


class ForceSuccess(Decorator):
    def _tick(self) -> Status:
        status = self.child.tick()
        return status if status is Status.RUNNING else Status.SUCCESS


class ForceFailure(Decorator):
    def _tick(self) -> Status:
        status = self.child.tick()
        return status if status is Status.RUNNING else Status.FAILURE


class RetryUntilSuccessful(Decorator):
    """Retry a failing child. ``num_attempts="-1"`` retries forever.

    This is the decorator that earns its place here: a grasp that fails because
    IK missed on this particular seed usually succeeds on the next try, and
    wrapping the pick in a retry is a one-line way to say so in the tree
    instead of in the planner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts = 0

    def _tick(self) -> Status:
        limit = int(self.port("num_attempts", 1, cast=int))
        status = self.child.tick()
        if status is Status.FAILURE:
            self.attempts += 1
            self.child.halt()
            if limit < 0 or self.attempts < limit:
                return Status.RUNNING
            self.attempts = 0
            return Status.FAILURE
        if status is Status.SUCCESS:
            self.attempts = 0
        return status

    def halt(self) -> None:
        super().halt()
        self.attempts = 0


class Repeat(Decorator):
    """Tick a succeeding child ``num_cycles`` times. ``-1`` means forever."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cycles = 0

    def _tick(self) -> Status:
        limit = int(self.port("num_cycles", 1, cast=int))
        status = self.child.tick()
        if status is Status.SUCCESS:
            self.cycles += 1
            self.child.halt()
            if limit < 0 or self.cycles < limit:
                return Status.RUNNING
            self.cycles = 0
            return Status.SUCCESS
        if status is Status.FAILURE:
            self.cycles = 0
        return status

    def halt(self) -> None:
        super().halt()
        self.cycles = 0


class KeepRunningUntilFailure(Decorator):
    def _tick(self) -> Status:
        status = self.child.tick()
        if status is Status.FAILURE:
            return Status.FAILURE
        if status is Status.SUCCESS:
            self.child.halt()
        return Status.RUNNING


class Timeout(Decorator):
    """Fail a child that runs longer than ``msec``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started: Optional[float] = None

    def _tick(self) -> Status:
        limit = float(self.port("msec", 5000, cast=float)) / 1000.0
        if self.started is None:
            self.started = time.monotonic()
        if time.monotonic() - self.started > limit:
            self.halt()
            return Status.FAILURE
        status = self.child.tick()
        if status.is_finished:
            self.started = None
        return status

    def halt(self) -> None:
        super().halt()
        self.started = None


class SubTreeNode(Decorator):
    """A whole other tree, ticked in place."""

    def _tick(self) -> Status:
        return self.child.tick()


# -- stateless leaves ----------------------------------------------------


class AlwaysSuccess(TreeNode):
    def _tick(self) -> Status:
        return Status.SUCCESS


class AlwaysFailure(TreeNode):
    def _tick(self) -> Status:
        return Status.FAILURE


class SetBlackboard(TreeNode):
    """``<SetBlackboard output_key="tray" value="tray_red"/>``"""

    def _tick(self) -> Status:
        key = self.attributes.get("output_key", "").strip("{} ")
        if not key:
            return Status.FAILURE
        self.blackboard.set(key, self.port("value"))
        return Status.SUCCESS


class Sleep(TreeNode):
    """Run for ``msec``, then succeed. Useful for letting a drop settle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started: Optional[float] = None

    def _tick(self) -> Status:
        limit = float(self.port("msec", 1000, cast=float)) / 1000.0
        if self.started is None:
            self.started = time.monotonic()
        if time.monotonic() - self.started >= limit:
            self.started = None
            return Status.SUCCESS
        return Status.RUNNING

    def halt(self) -> None:
        super().halt()
        self.started = None


class PopFromList(TreeNode):
    """Take the next name off a blackboard list.

    ``<PopFromList list="{cubes}" output_key="{cube}"/>``. Fails when the list
    is empty, which is how a tree says "for each object" with a Fallback.
    """

    def _tick(self) -> Status:
        key = self.attributes.get("list", "").strip("{} ")
        items = self.blackboard.get(key)
        if not items:
            return Status.FAILURE
        remaining = list(items)
        self.write_port("output_key", remaining.pop(0))
        self.blackboard.set(key, remaining)
        return Status.SUCCESS


CONTROL_NODES = {
    "Sequence": Sequence,
    "SequenceWithMemory": Sequence,
    "ReactiveSequence": ReactiveSequence,
    "Fallback": Fallback,
    "ReactiveFallback": ReactiveFallback,
    "Parallel": Parallel,
}

DECORATOR_NODES = {
    "Inverter": Inverter,
    "ForceSuccess": ForceSuccess,
    "ForceFailure": ForceFailure,
    "RetryUntilSuccessful": RetryUntilSuccessful,
    "Retry": RetryUntilSuccessful,
    "Repeat": Repeat,
    "KeepRunningUntilFailure": KeepRunningUntilFailure,
    "Timeout": Timeout,
}

BUILTIN_LEAVES = {
    "AlwaysSuccess": AlwaysSuccess,
    "AlwaysFailure": AlwaysFailure,
    "SetBlackboard": SetBlackboard,
    "Sleep": Sleep,
    "PopFromList": PopFromList,
}

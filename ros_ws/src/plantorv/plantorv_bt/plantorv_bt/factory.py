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

"""Turns a ``bt.xml`` into a tree of nodes.

The dialect is BehaviorTree.CPP 4's::

    <root BTCPP_format="4" main_tree_to_execute="MainTree">
      <BehaviorTree ID="MainTree">
        <Sequence>
          <Home/>
          <Pick object="cube_1"/>
          <Place target="tray_blue"/>
        </Sequence>
      </BehaviorTree>
    </root>

Everything is checked while loading: arity, unknown tags, subtrees that do not
exist and subtrees that refer to themselves. A tree that loads will not fall
over halfway through a run because of a typo in a node name.
"""

import xml.etree.ElementTree as ElementTree
from typing import Dict, Optional, Type

from plantorv_bt.core import (
    BUILTIN_LEAVES,
    CONTROL_NODES,
    DECORATOR_NODES,
    Blackboard,
    SubTreeNode,
    TreeNode,
)

# Groot writes these next to the trees; they describe nodes rather than being
# nodes, so they are skipped.
IGNORED_TAGS = {"TreeNodesModel", "include", "SubTreeNodesModel"}


class TreeBuildError(RuntimeError):
    """The XML does not describe a tree this executor can run."""


class BehaviorTreeFactory:
    """Builds trees, and holds the leaf types they may use."""

    def __init__(self, context=None):
        self.context = context
        self.leaves: Dict[str, Type[TreeNode]] = dict(BUILTIN_LEAVES)

    def register(self, tag: str, node_class: Type[TreeNode]) -> None:
        """Make a leaf type available to trees under the tag ``tag``."""
        self.leaves[tag] = node_class

    @property
    def known_tags(self):
        return sorted(set(CONTROL_NODES) | set(DECORATOR_NODES) | set(self.leaves) | {"SubTree"})

    # -- loading ---------------------------------------------------------

    def create_from_file(self, path: str, blackboard: Optional[Blackboard] = None) -> TreeNode:
        """Read a tree file and build the tree it names as its main one."""
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError as error:
            raise TreeBuildError(f"{path} is not valid XML: {error}")

        if root.tag != "root":
            raise TreeBuildError(f"{path}: the outermost element must be <root>, not <{root.tag}>")

        trees = {
            element.get("ID", f"tree_{index}"): element
            for index, element in enumerate(root.findall("BehaviorTree"))
        }
        if not trees:
            raise TreeBuildError(f"{path} contains no <BehaviorTree>")

        main = root.get("main_tree_to_execute")
        if main is None:
            if len(trees) > 1:
                raise TreeBuildError(
                    f"{path} has {len(trees)} trees; say which one runs with "
                    "main_tree_to_execute on <root>"
                )
            main = next(iter(trees))
        if main not in trees:
            raise TreeBuildError(
                f"{path}: main_tree_to_execute is '{main}', which is not one of "
                f"{sorted(trees)}"
            )

        self._trees = trees
        self._blackboard = blackboard or Blackboard()
        return self._build_tree(main, being_built=())

    # -- building --------------------------------------------------------

    def _build_tree(self, tree_id: str, being_built) -> TreeNode:
        if tree_id in being_built:
            chain = " -> ".join(being_built + (tree_id,))
            raise TreeBuildError(f"subtree recursion: {chain}")

        element = self._trees[tree_id]
        children = [child for child in element if child.tag not in IGNORED_TAGS]
        if len(children) != 1:
            raise TreeBuildError(
                f"BehaviorTree '{tree_id}' must have exactly one root node, found {len(children)}"
            )
        return self._build(children[0], being_built + (tree_id,))

    def _build(self, element, being_built) -> TreeNode:
        tag = element.tag
        attributes = dict(element.attrib)
        name = attributes.get("name", tag)
        children_elements = [child for child in element if child.tag not in IGNORED_TAGS]

        if tag == "SubTree":
            reference = attributes.get("ID")
            if reference is None:
                raise TreeBuildError("<SubTree> needs an ID")
            if reference not in self._trees:
                raise TreeBuildError(
                    f"<SubTree ID=\"{reference}\"> refers to a tree that is not defined; "
                    f"defined trees are {sorted(self._trees)}"
                )
            subtree = self._build_tree(reference, being_built)
            return SubTreeNode(name, attributes, self._blackboard, [subtree], self.context)

        if tag in CONTROL_NODES:
            if not children_elements:
                raise TreeBuildError(f"<{tag}> needs at least one child")
            children = [self._build(child, being_built) for child in children_elements]
            return CONTROL_NODES[tag](name, attributes, self._blackboard, children, self.context)

        if tag in DECORATOR_NODES:
            if len(children_elements) != 1:
                raise TreeBuildError(
                    f"<{tag}> takes exactly one child, found {len(children_elements)}"
                )
            child = self._build(children_elements[0], being_built)
            return DECORATOR_NODES[tag](name, attributes, self._blackboard, [child], self.context)

        if tag in self.leaves:
            if children_elements:
                raise TreeBuildError(f"<{tag}> is a leaf and cannot have children")
            return self.leaves[tag](name, attributes, self._blackboard, [], self.context)

        raise TreeBuildError(f"unknown node <{tag}>; this executor knows {self.known_tags}")

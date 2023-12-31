
from typing import Dict, List, Sequence, Union

import numpy as np

from .node import Node, SplitType


ParamT = Dict[str, str]

class Tree:
    """A tree built by XGBoost."""

    def __init__(self, tree_id: int, nodes: Sequence[Node]) -> None:
        self.tree_id = tree_id
        self.nodes = nodes

    def loss_change(self, node_id: int) -> float:
        """Loss gain of a node."""
        return self.nodes[node_id].loss_chg

    def sum_hessian(self, node_id: int) -> float:
        """Sum Hessian of a node."""
        return self.nodes[node_id].sum_hess

    def base_weight(self, node_id: int) -> float:
        """Base weight of a node."""
        return self.nodes[node_id].base_weight

    def split_index(self, node_id: int) -> int:
        """Split feature index of node."""
        return self.nodes[node_id].split_idx

    def split_condition(self, node_id: int) -> float:
        """Split value of a node."""
        return self.nodes[node_id].split_cond

    def split_categories(self, node_id: int) -> List[int]:
        """Categories in a node."""
        return self.nodes[node_id].categories

    def is_categorical(self, node_id: int) -> bool:
        """Whether a node has categorical split."""
        return self.nodes[node_id].split_type == SplitType.categorical

    def is_numerical(self, node_id: int) -> bool:
        return not self.is_categorical(node_id)

    def parent(self, node_id: int) -> int:
        """Parent ID of a node."""
        return self.nodes[node_id].parent

    def left_child(self, node_id: int) -> int:
        """Left child ID of a node."""
        return self.nodes[node_id].left

    def right_child(self, node_id: int) -> int:
        """Right child ID of a node."""
        return self.nodes[node_id].right

    def is_leaf(self, node_id: int) -> bool:
        """Whether a node is leaf."""
        return self.nodes[node_id].left == -1

    def is_deleted(self, node_id: int) -> bool:
        """Whether a node is deleted."""
        return self.split_index(node_id) == np.iinfo(np.uint32).max

    def __str__(self) -> str:
        stack = [0]
        nodes = []
        while stack:
            node: Dict[str, Union[float, int, List[int]]] = {}
            nid = stack.pop()

            node["node id"] = nid
            node["gain"] = self.loss_change(nid)
            node["cover"] = self.sum_hessian(nid)
            nodes.append(node)

            if not self.is_leaf(nid) and not self.is_deleted(nid):
                left = self.left_child(nid)
                right = self.right_child(nid)
                stack.append(left)
                stack.append(right)
                categories = self.split_categories(nid)

                node["split_idx"] = self.split_index(nid)

                if categories:
                    assert self.is_categorical(nid)
                    node["categories"] = categories
                else:
                    assert self.is_numerical(nid)
                    node["condition"] = self.split_condition(nid)
            if self.is_leaf(nid):
                node["weight"] = self.split_condition(nid)

        string = "\n".join(map(lambda x: "  " + str(x), nodes))
        return string

    def __eq__(self, other):
        if not isinstance(other, Tree):
            return False
        
        # Compare the number of nodes first
        if len(self.nodes) != len(other.nodes):
            return False
        
        # Compare nodes one by one
        # TODO: Make resistant to shuffling
        for node1, node2 in zip(self.nodes, other.nodes):
            if node1 != node2:
                return False

        return True
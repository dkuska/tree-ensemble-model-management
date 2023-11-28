from typing import Any, Dict, List, Union

from .tree import Tree
from .node import Node, SplitType



ParamT = Dict[str, str]


def to_integers(data: Union[bytes, List[int]]) -> List[int]:
    """Convert a sequence of bytes to a list of Python integer"""
    return [v for v in data]


class Model:
    """Gradient boosted tree model."""

    def __init__(self, model: dict) -> None:
        """Construct the Model from a JSON object.

        parameters
        ----------
         model : A dictionary loaded by json representing a XGBoost boosted tree model.
        """
        # Basic properties of a model
        self.learner_model_shape: ParamT = model["learner"]["learner_model_param"]
        self.num_output_group = int(self.learner_model_shape["num_class"])
        self.num_feature = int(self.learner_model_shape["num_feature"])
        self.base_score = float(self.learner_model_shape["base_score"])
        # A field encoding which output group a tree belongs
        self.tree_info = model["learner"]["gradient_booster"]["model"]["tree_info"]

        model_shape: ParamT = model["learner"]["gradient_booster"]["model"][
            "gbtree_model_param"
        ]

        # JSON representation of trees
        j_trees = model["learner"]["gradient_booster"]["model"]["trees"]

        # Load the trees
        self.num_trees = int(model_shape["num_trees"])

        trees: List[Tree] = []
        for i in range(self.num_trees):
            tree: Dict[str, Any] = j_trees[i]
            tree_id = int(tree["id"])
            assert tree_id == i, (tree_id, i)
            # - properties
            left_children: List[int] = tree["left_children"]
            right_children: List[int] = tree["right_children"]
            parents: List[int] = tree["parents"]
            split_conditions: List[float] = tree["split_conditions"]
            split_indices: List[int] = tree["split_indices"]
            # when ubjson is used, this is a byte array with each element as uint8
            default_left = to_integers(tree["default_left"])

            # - categorical features
            # when ubjson is used, this is a byte array with each element as uint8
            split_types = to_integers(tree["split_type"])
            # categories for each node is stored in a CSR style storage with segment as
            # the begin ptr and the `categories' as values.
            cat_segments: List[int] = tree["categories_segments"]
            cat_sizes: List[int] = tree["categories_sizes"]
            # node index for categorical nodes
            cat_nodes: List[int] = tree["categories_nodes"]
            assert len(cat_segments) == len(cat_sizes) == len(cat_nodes)
            cats = tree["categories"]
            assert len(left_children) == len(split_types)

            # The storage for categories is only defined for categorical nodes to
            # prevent unnecessary overhead for numerical splits, we track the
            # categorical node that are processed using a counter.
            cat_cnt = 0
            if cat_nodes:
                last_cat_node = cat_nodes[cat_cnt]
            else:
                last_cat_node = -1
            node_categories: List[List[int]] = []
            for node_id in range(len(left_children)):
                if node_id == last_cat_node:
                    beg = cat_segments[cat_cnt]
                    size = cat_sizes[cat_cnt]
                    end = beg + size
                    node_cats = cats[beg:end]
                    # categories are unique for each node
                    assert len(set(node_cats)) == len(node_cats)
                    cat_cnt += 1
                    if cat_cnt == len(cat_nodes):
                        last_cat_node = -1  # continue to process the rest of the nodes
                    else:
                        last_cat_node = cat_nodes[cat_cnt]
                    assert node_cats
                    node_categories.append(node_cats)
                else:
                    # append an empty node, it's either a numerical node or a leaf.
                    node_categories.append([])

            # - stats
            base_weights: List[float] = tree["base_weights"]
            loss_changes: List[float] = tree["loss_changes"]
            sum_hessian: List[float] = tree["sum_hessian"]

            # Construct a list of nodes that have complete information
            nodes: List[Node] = [
                Node(
                    left_children[node_id],
                    right_children[node_id],
                    parents[node_id],
                    split_indices[node_id],
                    split_conditions[node_id],
                    default_left[node_id] == 1,  # to boolean
                    SplitType(split_types[node_id]),
                    node_categories[node_id],
                    base_weights[node_id],
                    loss_changes[node_id],
                    sum_hessian[node_id],
                )
                for node_id in range(len(left_children))
            ]

            pytree = Tree(tree_id, nodes)
            trees.append(pytree)

        self.trees = trees

    def print_model(self) -> None:
        for i, tree in enumerate(self.trees):
            print("\ntree_id:", i)
            print(tree)
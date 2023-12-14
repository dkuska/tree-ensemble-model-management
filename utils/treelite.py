import json
import os
import pickle

from collections import Counter
import pandas as pd

import treelite
import treelite.sklearn


def get_model_jsons(model_dir):
    models = {}

    # Iterate over Files
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)

        # Load Model from SKLearn
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load Model into Treelite
        treelite_model = treelite.sklearn.import_model(model)
        # Load JSON Representation
        treelite_model_json = json.loads(
            treelite_model.dump_as_json(pretty_print=False)
        )

        models[model_name] = treelite_model_json

    return models


def remove_unneeded_node_keys(node, exact: bool = True):
    """
    Remove keys from nodes that are not important for comparison.
    Usually this refers to the keys that are training related.

    When position of the node is not important it also removes keys showing the position of the node in the tree.
    """
    keys_to_remove = ["data_count", "sum_hess", "gain"]
    if not exact:
        keys_to_remove.extend(["node_id", "left_child", "right_child"])

    reduced_node = {
        key: value for key, value in node.items() if key not in keys_to_remove
    }

    return reduced_node


# Duplicate Ratio


def node_is_leaf(node):
    if "leaf_value" in node:
        return True
    else:
        return False


def get_num_duplicate_nodes(tree1, tree2, exact: bool = True):
    num_duplicate_leaves = 0
    num_duplicate_inner = 0

    tree1_nodes = [remove_unneeded_node_keys(node, exact) for node in tree1["nodes"]]
    tree2_nodes = [remove_unneeded_node_keys(node, exact) for node in tree2["nodes"]]

    for node1 in tree1_nodes:
        if node1 in tree2_nodes:
            if node_is_leaf(node1):
                num_duplicate_leaves += 1
            else:
                num_duplicate_inner += 1
    return num_duplicate_leaves, num_duplicate_inner


def duplicate_trees(trees1, trees2, exact):
    duplicate_data = []
    # Iterate over Trees
    for i, tree1 in enumerate(trees1):
        num_nodes_tree1 = tree1["num_nodes"]
        for j, tree2 in enumerate(trees2):
            num_nodes_tree2 = tree2["num_nodes"]

            num_duplicates_leaves, num_duplicates_inner = get_num_duplicate_nodes(
                tree1, tree2, exact=exact
            )

            duplicate_ratio = (num_duplicates_leaves + num_duplicates_inner) / max(
                [num_nodes_tree1, num_nodes_tree2]
            )

            duplicate_data.append(
                {
                    "num_nodes_tree1": num_nodes_tree1,
                    "num_nodes_tree2": num_nodes_tree2,
                    "num_duplicates_leaves": num_duplicates_leaves,
                    "num_duplicates_inner": num_duplicates_inner,
                    "num_duplicates": num_duplicates_leaves + num_duplicates_inner,
                    "duplicate_ratio": duplicate_ratio,
                }
            )

    return duplicate_data


def compare_models(models: dict, sequentially: bool = False, exact: bool = True):
    duplicate_data = []

    for i, (model_name_a, model_a) in enumerate(models.items()):
        for j, (model_name_b, model_b) in enumerate(models.items()):
            if i == j:
                continue

            if sequentially and j != i + 1:
                continue

            duplicate_dict_list = duplicate_trees(
                model_a["trees"], model_b["trees"], exact=exact
            )

            duplicate_dict_list = [
                {**d, "model1": model_name_a, "model2": model_name_b}
                for d in duplicate_dict_list
            ]
            duplicate_data.extend(duplicate_dict_list)

    return pd.DataFrame(duplicate_data)


# Node Duplicates


def get_node_list(models, exact: bool = True):
    nodes = []

    for i, (model_name, model) in enumerate(models.items()):
        for tree in model["trees"]:
            for node in tree["nodes"]:
                reduced_node = remove_unneeded_node_keys(node, exact=exact)
                nodes.append(reduced_node)

    return nodes


def get_counts_of_unique_nodes(node_list: list):
    # Convert dictionaries to tuples and count occurrences using Counter
    tuple_counts = Counter(tuple(sorted(d.items())) for d in node_list)

    duplicates = {key: count for key, count in tuple_counts.items() if count > 1}

    return duplicates

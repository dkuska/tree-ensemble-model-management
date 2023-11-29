from dataclasses import dataclass
from enum import IntEnum, unique
from typing import List


@unique
class SplitType(IntEnum):
    numerical = 0
    categorical = 1

@dataclass
class Node:
    # properties
    left: int  # Left Child, is -1 for leaves
    right: int  # Right Child
    parent: int  # Parent, is -1 for nodes
    default_left: bool # TODO: Figure out what this does
        
    split_type: SplitType # Split either numerical or categorical
    # For Numeric Splits
    split_idx: int # Index on which split happens, not relevant for leaves
    split_cond: float # Condition on which to split
    # For Categorical Splits
    categories: List[int]

    # Output/Result
    base_weight: float # Weight of Leaf Node
    # Training Relevant 
    loss_chg: float # How much this leaf changes about the loss
    sum_hess: float # Sum of Hessian Matrix

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (
            self.left == other.left and
            self.right == other.right and
            self.parent == other.parent and
            self.split_idx == other.split_idx and
            self.split_cond == other.split_cond and
            self.default_left == other.default_left and
            self.split_type == other.split_type and
            self.categories == other.categories and
            self.base_weight == other.base_weight
        )
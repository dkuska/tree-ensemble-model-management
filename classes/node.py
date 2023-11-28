from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Dict, List

ParamT = Dict[str, str]

@unique
class SplitType(IntEnum):
    numerical = 0
    categorical = 1

@dataclass
class Node:
    # properties
    left: int
    right: int
    parent: int
    split_idx: int
    split_cond: float
    default_left: bool
    split_type: SplitType
    categories: List[int]
    # statistic
    base_weight: float
    loss_chg: float
    sum_hess: float

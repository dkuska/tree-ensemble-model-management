import json

from ..classes import Model, Node, Tree, SplitType, ParamT

def load_model(path: str) -> Model:
    with open(path) as f:
        model_json = json.load(f)
    return Model(model_json)
import xgboost as xgb


def load_model_json(model_path):
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def save_model_json(booster, model_path):
    booster.save_model(model_path)

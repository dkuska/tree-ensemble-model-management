import tempfile
import xgboost as xgb


def save_model_compressed(booster, model_path, compression_func):
    temp_model_path = tempfile.mktemp(suffix=".json")
    booster.save_model(temp_model_path)

    # Read the JSON data from the temporary file
    with open(temp_model_path, "rb") as f:
        model_binary = f.read()

    # Compress byte stream
    compressed = compression_func(model_binary)

    # Write compressed data to a file
    with open(model_path, "wb") as outfile:
        outfile.write(compressed)


def load_model_compressed(model_path, decompression_func):
    # Read and unpack the MessagePack data
    with open(model_path, "rb") as f:
        packed_data = f.read()
    model = decompression_func(packed_data)

    # Temporarily save the binary data to a file
    temp_model_path = tempfile.mktemp(suffix=".json")
    with open(temp_model_path, "wb") as f:
        f.write(model)
    # Load the model using the temporary file
    booster = xgb.Booster()
    booster.load_model(temp_model_path)
    return booster

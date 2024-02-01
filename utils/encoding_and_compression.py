import json
import msgpack
import tempfile
import xgboost as xgb


def save_model_msgpack_bz2(booster, model_path, compression_func):
    # 1st encode as msgpack
    temp_model_path = tempfile.mktemp(suffix=".json")
    booster.save_model(temp_model_path)
    # Read the JSON data from the temporary file
    with open(temp_model_path, "r") as f:
        model_dict = json.load(f)

    # Pack the byte stream using MessagePack
    packed = msgpack.packb(model_dict)

    # 2nd compress packed
    compressed_packed = compression_func(packed)
    with open(model_path, "wb") as outfile:
        outfile.write(compressed_packed)


def load_model_msgpack_bz2(model_path, decompression_func):
    with open(model_path, "rb") as infile:
        compressed_packed = infile.read()

    # 1st decompressed
    packed = decompression_func(compressed_packed)

    # 2nd unpack
    model_dict = msgpack.unpackb(packed)

    # Temporarily save the binary data to a file
    temp_model_path = tempfile.mktemp()
    with open(temp_model_path, "w") as f:
        json.dump(model_dict, f)

    # Load the model using the temporary file
    booster = xgb.Booster()
    booster.load_model(temp_model_path)
    return booster

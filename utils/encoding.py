import json

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import msgpack
import tempfile
import xgboost as xgb


def save_model_avro(booster, model_path_avro, path_avro_schema="xgboost.avsc"):
    # Save the model to a temporary JSON file
    temp_model_path = tempfile.mktemp(suffix=".json")
    booster.save_model(temp_model_path)

    # Read the JSON data from the temporary file
    with open(temp_model_path, "r") as f:
        model_json = json.load(f)

    # Load Avro schema
    avro_schema = avro.schema.parse(open(path_avro_schema, "rb").read())

    # Write the JSON data to an Avro file
    with open(model_path_avro, "wb") as avro_file:
        with DataFileWriter(avro_file, DatumWriter(), avro_schema) as writer:
            writer.append(model_json)


def load_model_avro(model_path_avro):
    # Read from Avro file
    with open(model_path_avro, "rb") as avro_file:
        with DataFileReader(avro_file, DatumReader()) as reader:
            for record in reader:
                model = record

    # Temporarily save the binary data to a file
    temp_model_path = tempfile.mktemp()
    with open(temp_model_path, "w") as f:
        json.dump(model, f)

    # Load the model using the temporary file
    booster = xgb.Booster()
    booster.load_model(temp_model_path)
    return booster


def save_model_msgpack(booster, model_path_msgpack):
    temp_model_path = tempfile.mktemp(suffix=".json")
    booster.save_model(temp_model_path)

    # Read the JSON data from the temporary file
    with open(temp_model_path, "r") as f:
        model_json = json.load(f)

    # Pack the byte stream using MessagePack
    packed = msgpack.packb(model_json)

    # Write the packed data to a file
    with open(model_path_msgpack, "wb") as outfile:
        outfile.write(packed)


def load_model_msgpack(model_path_msgpack):
    # Read and unpack the MessagePack data
    with open(model_path_msgpack, "rb") as f:
        packed_data = f.read()
    model_dict = msgpack.unpackb(packed_data)

    # Temporarily save the binary data to a file
    temp_model_path = tempfile.mktemp()
    with open(temp_model_path, "w") as f:
        json.dump(model_dict, f)

    # Load the model using the temporary file
    booster = xgb.Booster()
    booster.load_model(temp_model_path)
    return booster

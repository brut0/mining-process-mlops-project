""" Endpoint for prediction silica in mining data """

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from loguru import logger
from mlflow.tracking import MlflowClient

import config
from utils.logger import Logger
from utils.s3client import S3Client

Logger(filename=config.LOGGER_FILENAME, level=config.LOG_LEVEL)


run_id = None
model_name = None
model_version = 0
if "RUN_ID" in os.environ:
    run_id = os.environ["RUN_ID"]
else:
    # Get production version of model
    client = MlflowClient(
        f"http://{os.environ['MLFLOW_SERVER_HOST']}:{os.environ['MLFLOW_SERVER_PORT']}"
    )

    logger.info(f"Get versions of model '{config.MODEL_NAME}'")
    latest_versions = client.get_latest_versions(
        name=config.MODEL_NAME, stages=["Production"]
    )
    for version in latest_versions:
        if int(version.version) > model_version:
            run_id = version.run_id
            model_version = int(version.version)
            model_name = version.tags
        logger.info(
            f"run_id:{version.run_id} version:{version.version} model:{version.tags}"
            f" stage:{version.current_stage} status:{version.status}"
        )
logger.info(f"Selected model run_id:{run_id})")

# Load model and scaler
s3_client = S3Client().client
obj = s3_client.get_object(
    Bucket=config.S3_BUCKET,
    Key=f"models/6/{run_id}/artifacts/models_mlflow/model.pkl",
)
model = pickle.load(obj["Body"])
logger.info(f"Model: {model}")

obj = s3_client.get_object(
    Bucket=config.S3_BUCKET,
    Key=f"models/6/{run_id}/artifacts/scaler/scaler.b",
)
scaler = pickle.load(obj["Body"])
logger.info(f"Scaler: {scaler}")


def prepare_features(features):
    logger.info(features)
    df_features = [np.array(pd.Series(features))]
    logger.info(df_features)
    scaled_features = scaler.transform(df_features)
    return scaled_features


def predict(features):
    preds = model.predict(features)
    logger.info(f"Predicted Silica concentrate(%): {preds[0]}")
    return float(preds[0])


app = Flask("silica-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    mining_data = request.get_json()

    features = prepare_features(mining_data)
    prediction = predict(features)

    result = {
        "measurements_count": 1,  # TODO calculate
        "%_silica_concentrate": prediction,
        "model": {
            "name": model_name,
            "run_id": run_id,
            "version": model_version,
        },
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.environ["FLASK_RUN_PORT"])
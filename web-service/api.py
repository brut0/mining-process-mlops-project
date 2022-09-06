""" Endpoint for prediction silica in mining data """

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from loguru import logger

import config
from utils.logger import Logger
from utils.s3client import S3Client

Logger(filename=config.LOGGER_FILENAME, level=config.LOG_LEVEL)

run_id = "3bf92e8680d24fdf9a9e7de7da52314b"

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
    logger.info(f"{preds}")
    return float(preds[0])


app = Flask("silica-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    predictions = predict(features)

    result = {"duration": predictions}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.environ["FLASK_RUN_PORT"])

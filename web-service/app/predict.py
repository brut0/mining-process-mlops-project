""" Prediction endpoint """

import json
import os
import pickle

import config
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Blueprint, Response, request
from loguru import logger
from mlflow.tracking import MlflowClient
from utils.logger import Logger
from utils.s3client import S3Client

load_dotenv()  # take environment variables from .env

Logger(filename=config.LOGGER_FILENAME, level=config.LOG_LEVEL)


def get_runid() -> str:
    run_id = None
    if "RUN_ID" in os.environ:
        run_id = os.environ["RUN_ID"]
    else:
        # Get production version of model
        client = MlflowClient(
            f"http://{os.environ['MLFLOW_SERVER_HOST']}"
            f":{os.environ['MLFLOW_SERVER_PORT']}"
        )

        logger.info(f"Get versions of model '{config.MODEL_NAME}' from MLflow")
        latest_versions = client.get_latest_versions(
            name=config.MODEL_NAME, stages=["Production"]
        )
        model_version = 0
        for version in latest_versions:
            if int(version.version) > model_version:
                run_id = version.run_id
                model_version = int(version.version)
            logger.info(
                f"run_id:{version.run_id}"
                f" version:{version.version}  model:{version.tags} "
                f" stage:{version.current_stage} status:{version.status}"
            )
    return run_id


def get_model_s3(client, run_id):
    obj = client.get_object(
        Bucket=config.S3_BUCKET,
        Key=f"models/6/{run_id}/artifacts/models_mlflow/model.pkl",
    )
    model = pickle.load(obj["Body"])
    return model


def get_scaler_s3(client, run_id):
    obj = client.get_object(
        Bucket=config.S3_BUCKET,
        Key=f"models/6/{run_id}/artifacts/scaler/scaler.b",
    )
    scaler = pickle.load(obj["Body"])
    return scaler


def prepare_features(scaler, features):
    logger.info(features)
    df_features = [np.array(pd.Series(features))]
    logger.info(df_features)
    scaled_features = scaler.transform(df_features)
    return scaled_features


def predict(model, features):
    preds = model.predict(features)
    logger.info(f"Predicted Silica concentrate(%): {preds[0]}")
    return float(preds[0])


bp = Blueprint("ml", __name__)


@bp.route("/predict", methods=["POST"])
def predict_endpoint():
    mining_data = request.get_json()

    run_id = get_runid()
    logger.info(f"Selected model run_id:{run_id})")

    s3_client = S3Client().client
    model = get_model_s3(client=s3_client, run_id=run_id)
    logger.info(f"Model: {model}")

    scaler = get_scaler_s3(client=s3_client, run_id=run_id)
    logger.info(f"Scaler: {scaler}")

    features = prepare_features(scaler=scaler, features=mining_data)
    prediction = predict(model=model, features=features)

    result = {
        "measurements_count": 1,  # TODO calculate
        "%_silica_concentrate": prediction,
        "model": {
            # "name": model_name,
            "run_id": run_id,
            # "version": model_version,
        },
    }

    return Response(json.dumps(result), status=200, mimetype="application/json")

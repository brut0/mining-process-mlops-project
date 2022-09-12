""" Orchestration using Prefect """

import os
import pickle

import config
import mlflow
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from mlflow.tracking import MlflowClient
from model import get_test_data, preprocess_data, validate_month
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.model_selection import cross_val_score
from utils.logger import Logger
from utils.s3client import S3Client

load_dotenv(override=True)  # load environment variables from .env

Logger(filename=config.LOGGER_FILENAME, level=config.LOG_LEVEL)


def get_model(run_id=None):
    s3_client = S3Client().client
    obj = s3_client.get_object(
        Bucket=config.S3_BUCKET,
        Key=f"models/6/{run_id}/artifacts/models_mlflow/model.pkl",
    )
    model = pickle.load(obj['Body'])
    logger.info(f"Model: {model}")

    obj = s3_client.get_object(
        Bucket=config.S3_BUCKET,
        Key=f"models/6/{run_id}/artifacts/scaler/scaler.b",
    )
    scaler = pickle.load(obj['Body'])
    logger.info(f"Scaler: {scaler}")

    return model, scaler


@task
def get_latest_not_staged_model(mlflow_client: MlflowClient = None):
    '''Get latest model that is not staged or deployed
    If version is lower than that staged (deployed) version or
     there is no version than return None
    '''
    latest_versions = mlflow_client.get_latest_versions(name=config.MODEL_NAME)

    logger.info(f"Get versions of model '{config.MODEL_NAME}'")
    run_id = None
    last_not_staged_version = 0
    for version in latest_versions:
        if version.current_stage == 'None':
            if int(version.version) > last_not_staged_version:
                run_id = version.run_id
                last_not_staged_version = int(version.version)
        elif (
            version.current_stage in ('Staging', 'Production')
            and int(version.version) > last_not_staged_version
        ):
            run_id = None
        logger.info(
            f"run_id:{version.run_id} version:{version.version} model:{version.tags}"
            f" stage:{version.current_stage} status:{version.status}"
        )

    return run_id, last_not_staged_version


@task
def retrain_model(run_id=None, data_month=None):
    model, scaler = get_model(run_id=run_id)

    assert data_month > config.TRAIN_DATA_MONTH

    month = validate_month(data_month)
    df, filename = get_test_data(month=month)
    logger.info(f"Retrain data on {filename}")
    X, y = preprocess_data(df)
    X = scaler.transform(X)
    model.fit(X, y)
    score = -np.mean(cross_val_score(model, X, y, scoring=config.SCORING))
    logger.info(f"CV score {config.SCORING}:{score}")


@flow(task_runner=SequentialTaskRunner())
def run():
    mlflow.set_tracking_uri(
        f"http://{os.environ['MLFLOW_SERVER_HOST']}:{os.environ['MLFLOW_SERVER_PORT']}"
    )
    mlflow_client = MlflowClient(
        f"http://{os.environ['MLFLOW_SERVER_HOST']}:{os.environ['MLFLOW_SERVER_PORT']}"
    )

    run_id, _ = get_latest_not_staged_model(mlflow_client=mlflow_client)
    retrain_model(run_id, 8)


run()

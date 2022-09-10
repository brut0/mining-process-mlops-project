""" Functions to operate with data and model """

import glob
import io
import pickle

import config
import numpy as np
import pandas as pd

# import mlflow.pyfunc
from loguru import logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from utils.logger import Logger
from utils.s3client import S3Client

# Do not use '% Iron Concentrate' because of highly correlated(0.8) with target
# and do not use datetime
AGG_FEATURES = [
    'Starch Flow',
    'Amina Flow',
    'Ore Pulp Flow',
    'Ore Pulp pH',
    'Ore Pulp Density',
    'Flotation Column 01 Air Flow',
    'Flotation Column 02 Air Flow',
    'Flotation Column 03 Air Flow',
    'Flotation Column 04 Air Flow',
    'Flotation Column 05 Air Flow',
    'Flotation Column 06 Air Flow',
    'Flotation Column 07 Air Flow',
    'Flotation Column 01 Level',
    'Flotation Column 02 Level',
    'Flotation Column 03 Level',
    'Flotation Column 04 Level',
    'Flotation Column 05 Level',
    'Flotation Column 06 Level',
    'Flotation Column 07 Level',
]
LAB_FEATURES = ['% Iron Feed', '% Silica Feed']
AGG_FEATURES = [clm.replace(' ', '_').lower() for clm in AGG_FEATURES]
LAB_FEATURES = [clm.replace(' ', '_').lower() for clm in LAB_FEATURES]
TARGET = '%_silica_concentrate'

Logger(filename=config.LOGGER_FILENAME, level=config.LOG_LEVEL)


def get_list_data(source='local'):
    if source == 'local':
        files = glob.glob('*.parquet')
    elif source == 's3':
        s3 = S3Client().client
        files = [
            key['Key'] for key in s3.list_objects(Bucket=config.S3_BUCKET)['Contents']
        ]
    return files


def pd_read_s3_parquet(file, bucket, s3_client=None) -> pd.DataFrame:
    '''Read single parquet file from S3'''
    obj = s3_client.get_object(Bucket=bucket, Key=file)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))


def validate_month(month) -> str:
    if not isinstance(month, str):
        month = str(month)
    if int(month) < 10:
        month = '0' + month
    if int(month) > 12:
        pass  # TODO
    return month


def get_train_data(month, year=str(config.DATA_YEAR)) -> pd.DataFrame:
    s3_client = S3Client().client
    all_data = get_list_data(source='s3')
    filenames = [
        file for file in all_data if file <= f"mining_data_{year}-{month}.parquet"
    ]
    df = pd.concat(
        [
            pd_read_s3_parquet(file, bucket=config.S3_BUCKET, s3_client=s3_client)
            for file in filenames
        ]
    )
    return df, filenames


def get_test_data(month, year='2017'):
    s3_client = S3Client().client
    filename = f"mining_data_{year}-{month}.parquet"
    df = pd_read_s3_parquet(filename, bucket=config.S3_BUCKET, s3_client=s3_client)
    return df, filename


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [clm.replace(' ', '_').lower() for clm in df.columns]
    create_feature_dict = {
        **{clm: ['last'] for clm in LAB_FEATURES},
        **{clm: [np.mean, np.max, np.min] for clm in AGG_FEATURES},
    }
    X = (
        df.groupby(pd.Grouper(key='datetime', axis=0, freq='H'))
        .agg(create_feature_dict)
        .reset_index()
        .dropna()
    )
    X.columns = ['_'.join(col) for col in X.columns]
    y = (
        df[['datetime', TARGET]]
        .groupby(pd.Grouper(key='datetime', axis=0, freq='H'))
        .agg('last')
        .reset_index()
        .dropna()
    )

    return X.drop('datetime_', axis=1), y.drop('datetime', axis=1)


def eval_metrics(actual, pred) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    r2 = r2_score(actual, pred)
    max_er = max_error(actual, pred)
    metrics = {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'max_error': max_er}
    return metrics


def register_best_model(mlflow=None, mlflow_client: MlflowClient = None):
    if config.COMPARED_METRIC in ('mae', 'mape', 'mse', 'rmse'):
        metric_sorting = 'ASC'
    elif config.COMPARED_METRIC == 'r2':
        metric_sorting = 'DESC'
    else:
        raise Exception('Not known metric to sort choosing best model in MLflow')

    # Select the model with the better metric
    experiment = mlflow_client.get_experiment_by_name(config.EXPERIMENT_NAME)
    best_run = mlflow_client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{config.COMPARED_METRIC} {metric_sorting}"],
        max_results=1,
    )[0]

    logger.info(
        f"Best model is chosen:"
        f"  {best_run.data.tags['model']} id:{best_run.info.run_id}"
        f"  {config.COMPARED_METRIC}:{best_run.data.metrics[config.COMPARED_METRIC]}"
        f"  mlflow runName: {best_run.data.tags['mlflow.runName']}"
        f"  git commit:{best_run.data.tags['mlflow.source.git.commit']}"
        f"  train files:[{best_run.data.params['train-data-files']}]"
        f"  test file:{best_run.data.params['test-data-file']}"
    )

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name=config.MODEL_NAME,
        tags={'model': best_run.data.params['model']},
    )


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
        elif version.current_stage in ('Staging', 'Production'):
            if int(version.version) > last_not_staged_version:
                run_id = None
        logger.info(
            f"run_id:{version.run_id} version:{version.version} model:{version.tags}"
            f" stage:{version.current_stage} status:{version.status}"
        )

    return run_id, last_not_staged_version


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


def test_and_stage_model(
    mlflow_client: MlflowClient = None, run_id=None, model_version=None
):
    logger.info(f"Test and stage model: run_id={run_id} model_version={model_version}")

    # TODO try to fix
    # model = mlflow.pyfunc.load_model(
    #     model_uri=f"models:/{config.MODEL_NAME}/{model_version}"
    # )
    # filter_string = f"run_id='{run_id}'"
    # results = mlflow_client.search_model_versions(filter_string)
    # logger.info(results)

    model, scaler = get_model(run_id=run_id)

    month = validate_month(config.TEST_DATA_MONTH)
    df_test, filename = get_test_data(month=month)
    X_test, y_test = preprocess_data(df_test)
    X_test = scaler.transform(X_test)

    predicted = model.predict(X_test)
    metrics = eval_metrics(y_test, predicted)
    logger.info(f"Metrics on test data {filename}")
    logger.info(metrics)

    mlflow_client.transition_model_version_stage(
        name=config.MODEL_NAME,
        version=model_version,
        stage="Staging",
        archive_existing_versions=True,
    )
    logger.info(f"Model (run_id={run_id} to 'Staging'")


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

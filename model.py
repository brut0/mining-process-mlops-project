""" Functions to operate with data and model """

import io
import glob

import numpy as np
import pandas as pd
from loguru import logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    r2_score,
    max_error,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

import config
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


def get_test_data(month, year='2017') -> pd.DataFrame:
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


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    r2 = r2_score(actual, pred)
    max_er = max_error(actual, pred)
    metrics = {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'max_error': max_er}
    return metrics


def register_best_model(mlflow=None):
    client = MlflowClient()

    if config.COMPARED_METRIC in ('mae', 'mape', 'mse', 'rmse'):
        metric_sorting = 'ASC'
    elif config.COMPARED_METRIC == 'r2':
        metric_sorting = 'DESC'
    else:
        raise Exception('Not known metric to sort choosing best model in MLflow')

    # select the model with the lowest metric
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{config.COMPARED_METRIC} {metric_sorting}"],
    )[0]

    logger.info(
        f"Best model is chosen: \
        {best_run.data.tags['model']} id: {best_run.info.run_id} \
        {config.COMPARED_METRIC}: {best_run.data.metrics[config.COMPARED_METRIC]} \
        mlflow runName: {best_run.data.tags['mlflow.runName']}\
        git commit: {best_run.data.tags['mlflow.source.git.commit']}\
        train files: {best_run.data.params['train-data-files']} \
        test file: {best_run.data.params['test-data-file']}"
    )

    # register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name="mining-silica-regressor-best",
        tags={'model': best_run.data.params['model']},
    )

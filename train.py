import io
import os
import glob
import pickle

import boto3
import numpy as np
import mlflow
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score,
    max_error,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

import config
from logger import Logger

N_FOLDS = 5
N_JOBS = 8
SEED = 91

SCORING = 'neg_mean_absolute_error'
PREPROCCESS_TYPE = 'minmax_scaler no3rd'

TRAIN_DATA_MONTH = 7
TEST_DATA_MONTH = 8

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

Logger()


models = {
    'Linear Regression': {'model': LinearRegression(), 'params': {}},
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'alpha': (1.0, 10, 25, 50, 75, 100, 200),
            'tol': (1e-09, 1e-08, 1e-06),
        },
    },
    'ElasticNet': {
        'model': ElasticNet(),
        'params': {
            'alpha': (0.005, 0.01, 0.02, 0.05, 0.1),
            'l1_ratio': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0),
            'tol': (1e-09, 1e-08, 1e-06),
        },
    },
    'Desicion Tree': {
        'model': DecisionTreeRegressor(criterion='mae', random_state=SEED),
        'params': {
            'max_depth': (3, 5, 7),
        },
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(
            criterion='mae', n_jobs=N_JOBS, random_state=SEED
        ),
        'params': {
            'max_depth': (3, 4, 5),
            'n_estimators': [200, 500, 1000],
        },
    },
    'XGBoost': {
        'model': XGBRegressor(n_jobs=N_JOBS, random_state=SEED),
        'params': {
            'booster': ['gbtree'],
            'eval_metric': ["mae"],
            'n_estimators': [200, 500, 1000],
            'max_depth': [3, 5, 7],
            'alpha': [0.1, 1, 5, 10, 100],
        },
    },
}


def init_mlflow():
    mlflow.set_tracking_uri(
        f"http://{os.environ['MLFLOW_SERVER_HOST']}:{os.environ['MLFLOW_SERVER_PORT']}"
    )
    logger.info(f"tracking MLflow URI: '{mlflow.get_tracking_uri()}'")
    logger.info(f"{mlflow.list_experiments()}")


def init_s3_client():
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        region_name=os.environ['AWS_REGION'],
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    )
    return s3


def get_list_data(source='local'):
    if source == 'local':
        files = glob.glob('*.parquet')
    elif source == 's3':
        s3 = init_s3_client()
        files = [
            key['Key']
            for key in s3.list_objects(Bucket='kaggle-mining-process')['Contents']
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
    s3_client = init_s3_client()  # TODO
    all_data = get_list_data(source='s3')
    filenames = [
        file for file in all_data if file <= f"mining_data_{year}-{month}.parquet"
    ]
    df = pd.concat(
        [
            pd_read_s3_parquet(file, 'kaggle-mining-process', s3_client=s3_client)
            for file in filenames
        ]
    )
    return df, filenames


def get_test_data(month, year='2017') -> pd.DataFrame:
    s3_client = init_s3_client()  # TODO
    filename = f"mining_data_{year}-{month}.parquet"
    df = pd_read_s3_parquet(filename, 'kaggle-mining-process', s3_client=s3_client)
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


if __name__ == '__main__':
    load_dotenv()  # load environment variables from .env

    init_mlflow()

    month = validate_month(TRAIN_DATA_MONTH)
    df_train, train_filenames = get_train_data(month=month)
    logger.info(f"Train: {df_train.shape}")
    X_train, y_train = preprocess_data(df_train)
    logger.info(f"Train after preprocess: {X_train.shape}{y_train.shape}")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    with open("models/scaler.b", "wb") as f:
        pickle.dump(scaler, f)
    # mlflow.log_artifact("scaler.b", artifact_path="scaler")

    month = validate_month(TEST_DATA_MONTH)
    df_test, test_filename = get_test_data(month=month)
    logger.info(f"Test: {df_test.shape}")
    X_test, y_test = preprocess_data(df_test)
    logger.info(f"Test after preprocess: {X_test.shape}{y_test.shape}")
    X_test = scaler.transform(X_test)

    mlflow.set_experiment(config.EXPERIMENT_NAME)

    for model_name, m in models.items():
        with mlflow.start_run(run_name=f"GridSearchCV_{model_name}"):
            logger.info(m)
            model = m['model']
            params = m['params']
            logger.info(params)

            clf = GridSearchCV(
                model,
                params,
                cv=N_FOLDS,
                scoring=SCORING,
                return_train_score=False,
                n_jobs=N_JOBS,
            )
            clf.fit(X_train, y_train)
            model = clf.best_estimator_
            cv_score = -clf.best_score_
            with open('models/model.bin', 'wb') as f_out:
                pickle.dump(model, f_out)
            test_metrics = eval_metrics(y_test, model.predict(X_test))
            logger.info(clf.best_params_)
            logger.info(f"cross-val {SCORING}: {cv_score}")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value}")
            mlflow.set_tag("model", model_name)
            mlflow.log_param("model", model_name)
            mlflow.log_param("preprocess", PREPROCCESS_TYPE)
            mlflow.log_param('cv_scoring', SCORING)
            mlflow.log_params(model.get_params())
            mlflow.log_param("train-data-month", TRAIN_DATA_MONTH)
            mlflow.log_param("test-data-month", TEST_DATA_MONTH)
            mlflow.log_param("train-data-files", ', '.join(train_filenames))
            mlflow.log_param("test-data-file", test_filename)
            mlflow.log_metric(f"cv_{SCORING}", cv_score)
            mlflow.log_metrics(test_metrics)
            # mlflow.log_artifact(local_path="models/model.bin", artifact_path="models_pickle")
            if m == 'XGBoost':
                mlflow.xgboost.log_model(model, artifact_path="models_mlflow")
            else:
                mlflow.sklearn.log_model(model, artifact_path="models_mlflow")
            mlflow.log_artifact("models/scaler.b", artifact_path="scaler")

''' Train and register best model '''

import os
import pickle
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from loguru import logger
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

import config
from model import (
    eval_metrics,
    get_test_data,
    get_train_data,
    validate_month,
    preprocess_data,
)
from utils.logger import Logger

N_FOLDS = 5
N_JOBS = 8
SEED = 91

SCORING = 'neg_mean_absolute_error'
PREPROCCESS_TYPE = 'minmax_scaler no3rd'

TRAIN_DATA_MONTH = 7
TEST_DATA_MONTH = 8


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
            criterion='absolute_error', n_jobs=N_JOBS, random_state=SEED
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
            'alpha': [0.1, 1, 5, 10, 50, 75, 100, 150, 200],
        },
    },
}


def init_mlflow():
    mlflow.set_tracking_uri(
        f"http://{os.environ['MLFLOW_SERVER_HOST']}:{os.environ['MLFLOW_SERVER_PORT']}"
    )
    logger.info(f"tracking MLflow URI: '{mlflow.get_tracking_uri()}'")
    logger.info(f"{mlflow.list_experiments()}")


if __name__ == '__main__':
    Logger(filename='mlops.log', level=config.LOG_LEVEL)
    load_dotenv(override=True)  # load environment variables from .env

    init_mlflow()
    month = validate_month(TRAIN_DATA_MONTH)
    df_train, train_filenames = get_train_data(month=month)
    logger.info(f"Train: {df_train.shape}")
    X_train, y_train = preprocess_data(df_train)
    logger.info(f"Train after preprocess: {X_train.shape}{y_train.shape}")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    Path(f"{config.MODEL_PATH}{config.MODEL_FILE}").write_bytes(pickle.dumps(scaler))

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
            test_metrics = eval_metrics(y_test, model.predict(X_test))
            logger.info(clf.best_params_)
            logger.info(f"cross-val {SCORING}: {cv_score}")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value}")
            Path(f"{config.MODEL_PATH}{config.MODEL_FILE}").write_bytes(
                pickle.dumps(model)
            )
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
            # mlflow.log_artifact(local_path="models/model.bin",
            #  artifact_path="models_pickle")
            if m == 'XGBoost':
                mlflow.xgboost.log_model(model, artifact_path="models_mlflow")
            else:
                mlflow.sklearn.log_model(model, artifact_path="models_mlflow")
            mlflow.log_artifact("models/scaler.b", artifact_path="scaler")

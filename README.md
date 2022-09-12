# MLOps Zoomcamp Capstone Project
Capstone project for [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).

## Problem definition: Silica concentrate in mining process
The main goal is to use minig data to predict how much silica is in the ore concentrate after flotation. Data comes from one of the most important parts of a mining process: [flotation plant](https://en.wikipedia.org/wiki/Froth_flotation). Concentrate of iron and silica in ore measures right before it is fed into the flotation plant, this data sampled every 1 hour. Other samples measured every 20 seconds, but there is problem with data stamps so this measures could sampled every hour too. Concentrate of iron after flotation couldn't be as feature and should be deleted because measures in lab after flotation.
Exploratory data analysis is in notebooks.
Dataset source: [Quality Prediction in a Mining Process (Kaggle)](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)

## Repository structure
- **_notebooks_**: Jupyter notebooks with EDA and preparation of data for upload to S3
- **_train_**: Automated scripts to train model, register model and orchestration
- **_web-service_**: Deployment of prediction service using Flask as web service
- **_monitoring-service_**: Grafana to monitor evidently service
- **_pyproject.toml_**: Configuration for code quality tools
- **_.pre-commit-config.yaml_**: pre-commit hooks configuration

_Every folder has his own desription in README.md_

## Train, choose and register best model
Full instruction exist in **_train_** directory.
[MLflow](https://www.mlflow.org/) was used for experiment tracking and model registry. After a lot of experiments with various models, feature engineering and hyperparameters **Ridge Regression** as model with metric was choosed with hyperparameters **alpha=50**, **tol=1e-09**. Models compared in MLflow for metric: MAE, RMSE, MAPE and R2.
Developed orchestration script using [Prefect](https://www.prefect.io/) in `prefect_flow.py`.
MLflow and Prefect deployed in Compute Cloud of [Yandex Cloud](https://cloud.yandex.com/)
As artifact storage: S3 Object storage in Yandex Cloud

## Web service
Full instruction exist in **_web-service_** directory.
Deploy the model easily with a couple of commands, the script will make all the checks and only then deploy the service.

Set environment variables in `.env` file.

Run command of Makefile to deploy app:

    make setup
    make deploy

As default get last Production ready model from model registry a run Docker container published in Container Registry of Yandex Cloud

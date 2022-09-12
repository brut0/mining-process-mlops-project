## Experiment tracking and model registry (MLflow)

### MLflow installation

MLflow server configured for [Scenario 5: MLflow Tracking Server enabled with proxied artifact storage access](https://www.mlflow.org/docs/latest/tracking.html#scenario-5-mlflow-tracking-server-enabled-with-proxied-artifact-storage-access) with local PostgreSQL DB
<img src="https://www.mlflow.org/docs/latest/_images/scenario_5.png"  width="800" height="500">


1. Connect to VM in cloud
   _Ubuntu 20.04 in Yandex Cloud was used_
2. Install requirements
    ```bash
    apt install python3-pip
    apt install postgresql postgresql-contrib postgresql-server-dev-all gcc
    sudo -u postgres psql
    pip install psycopg2-binary
    service postgresql restart
    pip install mlflow, pysftp, boto3, psycopg2-binary
    ```
3. Create Object Storage and upload Data files after **_dataset_prepare.ipynb_**
4. Set S3 credentials to let MLflow connect to Yandex Object Storage
   - Set endpoint of S3 in /etc/credentials
    ```bash
    MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
    ```
   - Set ID and KEY of S3 account
    ```bash
    aws configure
    ```
   - Set environment variable
    ```bash
   AWS_REGION = 'ru-central1'
    ```
5. Run MLflow in Cloud
    ```bash
   mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --artifacts-destination s3://<S3_BUCKET_NAME>/models --serve-artifacts -h 0.0.0.0 -p 8000
    ```
6. Access MLflow in browser
    http://<VM_IP>:8000/


### Prefect installation

Prefect could be install on same VM

1. Connect to VM in cloud
   _Ubuntu 20.04 in Yandex Cloud was used_
2. Install requirements
    ```bash
    pip install prefect==2.3.1
    ```
3. Config Prefect
    ```bash
    prefect config unset PREFECT_ORION_UI_API_URL
    prefect config set PREFECT_ORION_UI_API_URL="http://<VM_IP>/api"
    prefect config view PREFECT_ORION_UI_API_URL
4. prefect orion start --host 0.0.0.0
    ```bash
   mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --artifacts-destination s3://<S3_BUCKET_NAME>/models --serve-artifacts -h 0.0.0.0 -p 8000
    ```
5. Access Prefect in browser
    http://<VM_IP>:4200/

### Train model
**_config.py_** contains all necessary configuration of scripts
1. Set environment variables. Create **_.env_** in project directory:
    ```bash
    # S3 Object Storage of Yandex Cloud
    AWS_REGION = 'ru-central1'
    AWS_ACCESS_KEY_ID = 'AAA'
    AWS_SECRET_ACCESS_KEY = 'BBB'

    # MLflow
    MLFLOW_SERVER_HOST = "IP"
    MLFLOW_SERVER_PORT = "PORT"

    # Prefect
    PREFECT_API_URL = 'http://IP:4200/api'
    ```
2. Activate virtual environment and install requirements
    ```bash
   pipenv shell
   pipenv install
    ```
3. Run training model
    ```bash
   python train.py
    ```
### Orchestration
Run prefect flow
    ```bash
   python prefect_flow.py
    ```

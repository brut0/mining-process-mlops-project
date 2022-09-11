

Build docker:
docker build . -t app:latest

Run docker:
docker run -p 5000:80 -e MLFLOW_SERVER_HOST="<HOST>" -e MLFLOW_SERVER_PORT="<PORT>" -e AWS_REGION="ru-central1" -e AWS_ACCESS_KEY_ID="<ID>" -e AWS_SECRET_ACCESS_KEY="<KEY>" app:latest

Push docker container to Yandex Container Registry:
sudo docker tag app cr.yandex/crp15g3ipk1q88a2kdhq/app:latest
sudo docker push cr.yandex/crp15g3ipk1q88a2kdhq/app:latest


Build docker:
docker build . -t app:latest

Run docker:
docker run -p 5000:80 -e MLFLOW_SERVER_HOST="51.250.29.195" -e MLFLOW_SERVER_PORT="8000" -e AWS_REGION="ru-central1" -e AWS_ACCESS_KEY_ID="YCAJEiNCxFDiC3YT6RpEyq2-P" -e AWS_SECRET_ACCESS_KEY="YCMO0xsrTj-crSOYdCJnDDidXodSlz1p3KJM2QXT" app:latest

Push docker container to Yandex Container Registry:
sudo docker tag app cr.yandex/crp15g3ipk1q88a2kdhq/app:latest
sudo docker push cr.yandex/crp15g3ipk1q88a2kdhq/app:latest
#!/usr/bin/env bash

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    export LOCAL_IMAGE_NAME="silica-mining-predictor:latest"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} ..
else
    echo "Image ${LOCAL_IMAGE_NAME} already exist"
fi

docker-compose up -d

sleep 3

pipenv run python test_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down

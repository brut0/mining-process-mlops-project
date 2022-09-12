quality_checks:
	isort --diff .
	black --diff .
	pylint --recursive=y .

unit_test:
	pytest app/

integration_test:
	pytest app/

build_docker:
	./dockerize.sh

publish_docker:
	sudo docker tag app cr.yandex/$CONTAINER_REGISTRY_PATH/app:latest
	sudo docker push cr.yandex/$CONTAINER_REGISTRY_PATH/app:latest

build_publish: quality_checks unit_test integration_test build_docker publish_docker

deploy:
	docker run -it --rm -p 5000:80 -e MLFLOW_SERVER_HOST=$MLFLOW_SERVER_HOST \
	-e MLFLOW_SERVER_PORT=$MLFLOW_SERVER_PORT -e AWS_REGION=$AWS_REGION \
	-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
	app:latest

run:
	flask run

setup:
	bash .env
	pipenv install
	pipenv shell

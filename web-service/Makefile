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
	sudo docker tag app cr.yandex/crp15g3ipk1q88a2kdhq/app:latest
	sudo docker push cr.yandex/crp15g3ipk1q88a2kdhq/app:latest

run:
	flask run

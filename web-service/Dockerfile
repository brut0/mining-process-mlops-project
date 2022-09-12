FROM python:3.10-slim-buster

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]
COPY /app /app/
COPY /docker-entrypoint.sh /app/docker-entrypoint.sh

RUN pipenv install --system --deploy

ENTRYPOINT  /app/docker-entrypoint.sh

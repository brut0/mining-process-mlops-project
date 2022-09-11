""" Endpoint for prediction silica in mining data """

import config
from flask import Flask
from predict import bp
from utils.logger import Logger

Logger(filename=config.LOGGER_FILENAME, level=config.LOG_LEVEL)


def create_app():
    flask_app = Flask("silica-prediction")
    flask_app.register_blueprint(bp)
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run()

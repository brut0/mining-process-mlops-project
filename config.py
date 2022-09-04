""" Global parameters """

S3_BUCKET = 'kaggle-mining-process'
MODEL_PATH = 'models/'
MODEL_FILE = 'model.bin'
SCALER_FILE = 'scaler.b'
MODEL_NAME = 'mining-silica-regressor-best'
DATA_YEAR = 2017
TRAIN_DATA_MONTH = 7
TEST_DATA_MONTH = 8
EXPERIMENT_NAME = "my-experiment-6"
LOG_LEVEL = 'TRACE'  # TODO describe others
SCORING = 'neg_mean_absolute_error'
COMPARED_METRIC = 'r2'
LOGGER_FILENAME = 'mlops.log'

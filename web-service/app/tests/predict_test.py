""" Test 'predict' endpoint """
# pylint: disable=unused-import
# pylint: disable=too-many-arguments

import json
import pickle
from pathlib import Path

import config
import pytest

from . import client


@pytest.fixture
def url():
    return "/predict"


@pytest.fixture
def model():
    test_dir = Path(__file__).parent
    with open(test_dir / "mocks/model.bin", "rb") as f:
        model_file = pickle.load(f)
    return model_file


@pytest.fixture
def scaler():
    test_dir = Path(__file__).parent
    with open(test_dir / "mocks/scaler.b", "rb") as f:
        scaler_file = pickle.load(f)
    return scaler_file


@pytest.fixture
def mining_data():
    test_dir = Path(__file__).parent
    with open(test_dir / "mocks/data.json", "r", encoding="utf-8") as json_data:
        data = json.load(json_data)
    return data


def test_predict_success(client, url, mining_data, model, scaler, mocker):
    run_id = "3bf92e8680d24fdf9a9e7de7da52314b"
    mocker.patch("predict.get_runid", return_value=run_id)
    mocker.patch("predict.get_model_s3", return_value=model)
    mocker.patch("predict.get_scaler_s3", return_value=scaler)
    response = client.post(url, json=mining_data)
    response_data = json.loads(response.data.decode("utf-8"))
    assert response.content_type == "application/json"
    assert response.status_code == 200
    assert isinstance(response_data[config.TARGET], float)
    assert "model" in response_data
    assert response_data["model"]["run_id"] == run_id


def test_predict_bad_data(client, url, mining_data, model, scaler, mocker):
    mocker.patch("predict.get_runid", return_value=None)
    mocker.patch("predict.get_model_s3", return_value=model)
    mocker.patch("predict.get_scaler_s3", return_value=scaler)

    data_1 = mining_data
    data_1["flotation_column_06_level_mean"] = None
    data_2 = mining_data
    data_2["flotation_column_06_level_mean"] = "abs"
    datas = [
        data_1,
        data_2,
        {"sdfsfsfd": 324},
        {},
    ]

    for data in datas:
        response = client.post(url, json=data)
        assert response.content_type == "application/json"
        assert response.status_code == 500


def test_predict_no_connection(client, url, mining_data, mocker):
    mocker.patch("predict.get_runid", return_value=None)
    mocker.patch("predict.get_model_s3", return_value=None)
    mocker.patch("predict.get_scaler_s3", return_value=None)
    response = client.post(url, json=mining_data)
    assert response.content_type == "application/json"
    assert response.status_code == 500


def test_predict_no_model(client, url, mining_data, mocker):
    mocker.patch("predict.get_runid", return_value=None)
    mocker.patch("predict.get_model_s3", return_value=model)
    mocker.patch("predict.get_scaler_s3", return_value=None)
    response = client.post(url, json=mining_data)
    assert response.content_type == "application/json"
    assert response.status_code == 500


def test_predict_no_scaler(client, url, mining_data, mocker):
    mocker.patch("predict.get_runid", return_value=None)
    mocker.patch("predict.get_model_s3", return_value=None)
    mocker.patch("predict.get_scaler_s3", return_value=scaler)
    response = client.post(url, json=mining_data)
    assert response.content_type == "application/json"
    assert response.status_code == 500

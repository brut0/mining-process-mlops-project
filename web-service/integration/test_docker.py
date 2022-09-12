""" Test docker """

import json
from pathlib import Path

import requests
from deepdiff import DeepDiff

test_directory = Path(__file__).parent

with open(test_directory / "house_data.json", "rt", encoding="utf-8") as f_in:
    event = json.load(f_in)

URL = "http://127.0.0.1:8080/predict"
actual_response = requests.post(URL, json=event, timeout=10)
print(actual_response.json())

expected_response = {"price": float(3.2185588298772068)}
print(expected_response)

diff = DeepDiff(actual_response.json(), expected_response, significant_digits=3)
print(f"diff={diff}")

assert "type_changes" not in diff
assert "values_changed" not in diff

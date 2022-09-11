""" Test 'predict' endpoint """
# pylint: disable=unused-import

import pytest

from . import client


@pytest.fixture
def url():
    return "/predict"


@pytest.fixture
def mining_data():
    return {
        "%_iron_feed_last": 59.720000,
        "%_silica_feed_last": 9.220000,
        "starch_flow_mean": 2337.195500,
        "starch_flow_amax": 2686.740000,
        "starch_flow_amin": 2031.240000,
        "amina_flow_mean": 555.275083,
        "amina_flow_amax": 575.542000,
        "amina_flow_amin": 520.983000,
        "ore_pulp_flow_mean": 399.960622,
        "ore_pulp_flow_amax": 410.338000,
        "ore_pulp_flow_amin": 390.668000,
        "ore_pulp_ph_mean": 9.605180,
        "ore_pulp_ph_amax": 9.774600,
        "ore_pulp_ph_amin": 9.452550,
        "ore_pulp_density_mean": 1.754893,
        "ore_pulp_density_amax": 1.777850,
        "ore_pulp_density_amin": 1.716790,
        "flotation_column_01_air_flow_mean": 176.341925,
        "flotation_column_01_air_flow_amax": 176.379742,
        "flotation_column_01_air_flow_amin": 176.304107,
        "flotation_column_02_air_flow_mean": 184.851379,
        "flotation_column_02_air_flow_amax": 185.083301,
        "flotation_column_02_air_flow_amin": 184.619458,
        "flotation_column_03_air_flow_mean": 184.553904,
        "flotation_column_03_air_flow_amax": 184.814990,
        "flotation_column_03_air_flow_amin": 184.292819,
        "flotation_column_04_air_flow_mean": 295.096000,
        "flotation_column_04_air_flow_amax": 295.096000,
        "flotation_column_04_air_flow_amin": 295.096000,
        "flotation_column_05_air_flow_mean": 306.400000,
        "flotation_column_05_air_flow_amax": 306.400000,
        "flotation_column_05_air_flow_amin": 306.400000,
        "flotation_column_06_air_flow_mean": 249.794694,
        "flotation_column_06_air_flow_amax": 257.495000,
        "flotation_column_06_air_flow_amin": 244.445000,
        "flotation_column_07_air_flow_mean": 250.005817,
        "flotation_column_07_air_flow_amax": 254.618000,
        "flotation_column_07_air_flow_amin": 246.973000,
        "flotation_column_01_level_mean": 801.112289,
        "flotation_column_01_level_amax": 822.823000,
        "flotation_column_01_level_amin": 766.605000,
        "flotation_column_02_level_mean": 799.143283,
        "flotation_column_02_level_amax": 818.160000,
        "flotation_column_02_level_amin": 777.026000,
        "flotation_column_03_level_mean": 799.931961,
        "flotation_column_03_level_amax": 821.197000,
        "flotation_column_03_level_amin": 773.888000,
        "flotation_column_04_level_mean": 448.355867,
        "flotation_column_04_level_amax": 505.458000,
        "flotation_column_04_level_amin": 320.142000,
        "flotation_column_05_level_mean": 452.357578,
        "flotation_column_05_level_amax": 554.592000,
        "flotation_column_05_level_amin": 339.236000,
        "flotation_column_06_level_mean": 451.192683,
        "flotation_column_06_level_amax": 504.723000,
        "flotation_column_06_level_amin": 363.322000,
        "flotation_column_07_level_mean": 450.081333,
        "flotation_column_07_level_amax": 538.859000,
        "flotation_column_07_level_amin": 327.225000,
    }


def test_predict_success(client, url, mining_data):
    # mocker.patch("dao.pricing_dao.PricingDAO.update_table_with_ml_calc")
    response = client.post(url, json=mining_data)
    assert response.status_code == 200
    assert response.content_type == "application/json"
    # assert response_data["status"] == "succeeded"

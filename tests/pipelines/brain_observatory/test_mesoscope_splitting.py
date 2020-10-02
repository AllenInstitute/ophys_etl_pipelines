import pytest
from mock import MagicMock
from ophys_etl.pipelines.brain_observatory.scripts import (
    run_mesoscope_splitting)


@pytest.fixture
def mock_volume():
    volume = MagicMock()
    volume.plane_shape = (100, 100)

    return volume


@pytest.fixture
def exp_info():
    experiment_info = {"resolution": 0.5,
                       "offset_x": 20,
                       "offset_y": 50,
                       "rotation": 0.1}

    return experiment_info


def test_conversion_output(mock_volume, exp_info):
    run_mesoscope_splitting.conversion_output(
        mock_volume, "test.out", exp_info)

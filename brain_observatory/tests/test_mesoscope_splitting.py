import pytest
from mock import MagicMock, PropertyMock
from brain_observatory_utils.scripts.run_mesoscope_splitting import (
    conversion_output)


@pytest.fixture
def mock_volume():
    volume = MagicMock()
    volume.plane_shape = (100,100)

    return volume


@pytest.fixture
def exp_info():
    experiment_info = {"resolution": 0.5,
                       "offset_x": 20,
                       "offset_y": 50,
                       "rotation": 0.1}

    return experiment_info


def test_conversion_output(mock_volume, exp_info):
    o, m = conversion_output(mock_volume, "test.out", exp_info)

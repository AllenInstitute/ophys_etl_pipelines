from pathlib import Path
import pytest

import numpy as np

from ophys_etl.modules.mesoscope_splitting import tiff


@pytest.fixture
def example_tiff():
    """
        This TIFF is from an actual experiment, as creating a new one
        with the proper metadata "header" is nontrivial. This specific
        TIFF was chosen because the plane_stride value produced would
        previously result in a bug.
    """
    tiff_path = Path(__file__).parent.joinpath(
            'resources', '914224851_averaged_surface.tiff')
    t = tiff.MesoscopeTiff(tiff_path)

    return t


def test_plane_stride(example_tiff):
    assert 2 == example_tiff.plane_stride


def test_plane_scans(example_tiff):
    correct_value = np.array([-190,    0, -110,    0])
    np.testing.assert_array_equal(
        example_tiff.plane_scans, correct_value)


def test_plane_views(example_tiff):
    assert 2 == len(example_tiff.plane_views)


def test_fast_zs(example_tiff):
    correct_value = np.array([[-190, 0], [-110, 0]])
    np.testing.assert_array_equal(
        example_tiff.fast_zs, correct_value)


def test_volume_scans(example_tiff):
    correct_value = np.array([[-190, -110], [0, 0]])
    np.testing.assert_array_equal(
        example_tiff.volume_scans, correct_value)


def test_volume_stride(example_tiff):
    assert 0 == example_tiff.volume_stride


def test_volume_views(example_tiff):
    assert 0 == len(example_tiff.volume_views)

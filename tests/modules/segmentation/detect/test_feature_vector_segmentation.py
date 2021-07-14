import pytest

import pathlib
import numpy as np
import json

from ophys_etl.types import ExtractROI

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.detect.feature_vector_segmentation import (
        convert_to_lims_roi,
        FeatureVectorSegmenter,
        _is_roi_at_edge)


@pytest.fixture
def seeder_args():
    args = {
            'exclusion_buffer': 1,
            'n_samples': 10,
            'keep_fraction': 0.05,
            'minimum_distance': 3.0,
            'seeder_grid_size': None}
    return args


@pytest.mark.parametrize(
    "origin,mask,expected",
    [
     ((14, 22), np.array([[False, False, False, False, False],
                          [False, True, False, False, False],
                          [False, False, True, False, False],
                          [False, False, False, False, False],
                          [False, False, False, False, False]]),
      ExtractROI(
          id=0,
          x=23,
          y=15,
          width=2,
          height=2,
          mask=[[True, False], [False, True]],
          valid=False)
      )
    ])
def test_roi_converter(origin, mask, expected):
    """
    Test method that converts ROIs to LIMS-like ROIs
    """
    actual = convert_to_lims_roi(origin, mask)
    assert actual == expected


def test_graph_to_img(example_graph):
    """
    smoke test graph_to_img
    """
    img = graph_to_img(example_graph,
                       attribute_name='dummy_attribute')

    # check that image has expected type and shape
    assert type(img) == np.ndarray
    assert img.shape == (40, 40)

    # check that ROI pixels are
    # brighter than non ROI pixels
    roi_mask = np.zeros((40, 40), dtype=bool)
    roi_mask[12:16, 4:7] = True
    roi_mask[25:32, 11:18] = True
    roi_mask[25:27, 15:18] = False

    roi_flux = img[roi_mask].flatten()
    complement = np.logical_not(roi_mask)
    not_roi_flux = img[complement].flatten()

    roi_mu = np.mean(roi_flux)
    roi_std = np.std(roi_flux, ddof=1)
    not_mu = np.mean(not_roi_flux)
    not_std = np.std(not_roi_flux, ddof=1)

    assert roi_mu > not_mu+roi_std+not_std


def test_segmenter(tmpdir, example_graph, example_video, seeder_args):
    """
    Smoke test for segmenter
    """

    segmenter = FeatureVectorSegmenter(graph_input=example_graph,
                                       video_input=example_video,
                                       attribute='dummy_attribute',
                                       filter_fraction=0.2,
                                       n_processors=1,
                                       seeder_args=seeder_args)

    dir_path = pathlib.Path(tmpdir)
    roi_path = dir_path / 'roi.json'
    seed_path = dir_path / 'seed.json'
    plot_path = dir_path / 'plot.png'
    assert not roi_path.exists()
    assert not seed_path.exists()
    assert not plot_path.exists()

    segmenter.run(roi_output=roi_path,
                  seed_output=seed_path,
                  plot_output=plot_path)

    assert roi_path.is_file()
    assert seed_path.is_file()
    assert plot_path.is_file()

    # check that some ROIs got written
    with open(roi_path, 'rb') as in_file:
        roi_data = json.load(in_file)
    assert len(roi_data) > 0

    # test that it can handle not receiving a
    # seed_path or plot_path
    roi_path.unlink()
    seed_path.unlink()
    plot_path.unlink()

    assert not roi_path.exists()
    assert not seed_path.exists()
    assert not plot_path.exists()

    segmenter.run(roi_output=roi_path,
                  seed_output=None,
                  plot_output=None)

    assert roi_path.is_file()
    assert not seed_path.exists()
    assert not plot_path.exists()


def test_segmenter_blank(tmpdir, blank_graph, blank_video, seeder_args):
    """
    Smoke test for segmenter on blank inputs
    """

    segmenter = FeatureVectorSegmenter(graph_input=blank_graph,
                                       video_input=blank_video,
                                       attribute='dummy_attribute',
                                       filter_fraction=0.2,
                                       n_processors=1,
                                       seeder_args=seeder_args)
    dir_path = pathlib.Path(tmpdir)
    roi_path = dir_path / 'roi.json'
    segmenter.run(roi_output=roi_path)


def test_is_roi_at_edge():

    # in the middle of fov
    mask = np.zeros((9, 9), dtype=bool)
    mask[2, 2] = True
    assert not _is_roi_at_edge((10, 10),
                               (20, 20),
                               mask)

    # at upper edge
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 3:5] = True
    assert _is_roi_at_edge((10, 10),
                           (20, 20),
                           mask)

    # at left edge
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:5, 0] = True
    assert _is_roi_at_edge((10, 10),
                           (20, 20),
                           mask)

    # at right edge
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:5, 8] = True
    assert _is_roi_at_edge((10, 10),
                           (20, 20),
                           mask)

    # at bottom edge
    mask = np.zeros((9, 9), dtype=bool)
    mask[8, 3:5] = True
    assert _is_roi_at_edge((10, 10),
                           (20, 20),
                           mask)

    # at upper edge with no room to grow
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 3:5] = True
    assert not _is_roi_at_edge((0, 10),
                               (20, 20),
                               mask)

    # at left edge with no room to grow
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:5, 0] = True
    assert not _is_roi_at_edge((10, 0),
                               (20, 20),
                               mask)

    # at right edge with no room to grow
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:5, 8] = True
    assert not _is_roi_at_edge((10, 10),
                               (20, 19),
                               mask)

    # at bottom edge with no room to grow
    mask = np.zeros((9, 9), dtype=bool)
    mask[8, 3:5] = True
    assert not _is_roi_at_edge((10, 10),
                               (19, 20),
                               mask)

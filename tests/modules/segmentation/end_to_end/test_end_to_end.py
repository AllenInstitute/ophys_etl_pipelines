import pytest
from typing import List, Set, Tuple
import pathlib
import h5py
import numpy as np
import sys

from ophys_etl.modules.segmentation.modules.calculate_edges import (
    CalculateEdges)
from ophys_etl.modules.segmentation.modules.feature_vector_segmentation \
        import FeatureVectorSegmentationRunner
from ophys_etl.modules.segmentation.modules.roi_merging import (
    RoiMergerEngine)
from ophys_etl.modules.segmentation.modules.filter_area import (
    AreaFilterRunner)
from ophys_etl.modules.segmentation.modules.hnc_segmentation_wrapper import (
    HNCSegmentationWrapper)
from ophys_etl.modules.segmentation.processing_log import \
    SegmentationProcessingLog


@pytest.fixture(scope='session')
def synthetic_video_path(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('end_to_end'))

    video_path = tmpdir/'synthetic_video.h5'

    nt = 100
    nx = 32
    ny = 32

    rng = np.random.default_rng(553311)
    data = np.abs(rng.normal(1.0, 0.02, size=(nt, nx, ny)))
    t_array = np.arange(nt)

    for ii in range(10):
        r0 = rng.integers(0, 30)
        c0 = rng.integers(0, 30)
        n_peaks = rng.integers(10, 30)
        t_peak = rng.integers(5, 90, size=n_peaks)
        width = min(rng.integers(3, 5), nx-c0)
        height = min(rng.integers(3, 5), ny-r0)

        for peak in t_peak:
            amp = 5.0+10.0*rng.random()
            sigma = 1.0+2.0*rng.random()
            flux = amp*np.exp(-0.5*((t_array-peak)/sigma)**2)
            for r in range(height):
                for c in range(width):
                    random_factor = rng.normal(1.0, 0.01)
                    data[:, r0+r, c0+c] += flux*random_factor
    data -= data.min()
    data = np.round((2**16-1)*data/data.max()).astype(np.uint16)
    assert data.min() >= 0
    assert data.max() < 2**16

    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    yield video_path


def pixel_set_from_roi_list(roi_list: List[dict]) -> Set[Tuple[int, int]]:
    pixel_set = set()
    for roi in roi_list:
        row0 = roi['y']
        col0 = roi['x']
        for i_row, mask_row in enumerate(roi['mask']):
            for i_col in range(len(mask_row)):
                if mask_row[i_col]:
                    pixel_set.add((row0+i_row, col0+i_col))
    return pixel_set


@pytest.mark.skipif(sys.version_info.minor < 8,
                    reason='tests started failing in 3.7')
@pytest.mark.parametrize('roi_class', ['PearsonFeatureROI', 'PCAFeatureROI'])
def test_edge_fvs_filter_merge(tmpdir, synthetic_video_path, roi_class):
    """
    test FVS segmentation pipeline
    """

    if roi_class == 'PearsonFeatureROI':
        filter_fraction = 0.2
    elif roi_class == 'PCAFeatureROI':
        filter_fraction = 1.0
    else:
        raise RuntimeError("no filter fraction in test")

    tmpdir_path = pathlib.Path(tmpdir)
    graph_path = tmpdir_path/'edge_graph.pkl'
    graph_plot_path = tmpdir_path/'edge_graph.png'

    # calculate edges
    data = {'video_path': str(synthetic_video_path),
            'graph_output': str(graph_path),
            'plot_output': str(graph_plot_path),
            'attribute': 'filtered_hnc_Gaussian',
            'filter_fraction': 0.2,
            'neighborhood_radius': 7}

    edge_runner = CalculateEdges(input_data=data, args=[])
    edge_runner.run()

    assert graph_path.is_file()
    assert graph_plot_path.is_file()

    # find ROIs
    log_path = tmpdir_path/'fvs_qc.h5'
    roi_plot_path = tmpdir_path/'fvs_plot.png'

    data = {'video_input': str(synthetic_video_path),
            'graph_input': str(graph_path),
            'log_path': str(log_path),
            'plot_output': str(roi_plot_path),
            'attribute': 'filtered_hnc_Gaussian',
            'filter_fraction': filter_fraction,
            'n_parallel_workers': 2,
            'roi_class': roi_class,
            'seeder_args': {'keep_fraction': 0.1,
                            'minimum_distance': 20,
                            'n_samples': 6}}

    fvs_runner = FeatureVectorSegmentationRunner(
                     input_data=data, args=[])
    fvs_runner.run()
    assert log_path.is_file()
    assert roi_plot_path.is_file()

    qcfile = SegmentationProcessingLog(log_path, read_only=True)
    raw_rois = qcfile.get_rois_from_group(qcfile.get_last_group())
    n_raw_roi = len(raw_rois)
    assert n_raw_roi > 0

    raw_roi_pixels = pixel_set_from_roi_list(raw_rois)

    with h5py.File(log_path, 'r') as in_file:
        assert 'detect' in in_file
        group = in_file["detect"]
        assert 'seed' in group

    # Select an area filter that won't exclude all ROIs
    areas = np.array([np.array(roi['mask']).sum()
                      for roi in raw_rois])
    min_area, max_area = np.quantile(areas, (0.2, 0.8))

    # filter on area
    data = {'pipeline_stage': 'area filter',
            'log_path': str(log_path),
            'max_area': max_area,
            'min_area': min_area}

    filter_runner = AreaFilterRunner(input_data=data, args=[])
    filter_runner.run()

    # verify pixels were conserved in filtering
    filtered_rois = qcfile.get_rois_from_group(qcfile.get_last_group())
    filtered_pixels = pixel_set_from_roi_list(filtered_rois)
    assert filtered_pixels == raw_roi_pixels

    n_valid_filtered_roi = 0
    for roi in filtered_rois:
        if roi['valid']:
            n_valid_filtered_roi += 1
    assert n_valid_filtered_roi > 0
    assert n_valid_filtered_roi < n_raw_roi
    assert len(filtered_rois) == n_raw_roi

    # merge ROIs
    merged_plot_path = tmpdir_path/'fvs_merged_plot.png'
    data = {'video_input': str(synthetic_video_path),
            'plot_output': str(merged_plot_path),
            'log_path': str(log_path),
            'n_parallel_workers': 2,
            'attribute': 'filtered_hnc_Gaussian'}

    merge_runner = RoiMergerEngine(input_data=data, args=[])
    merge_runner.run()
    assert merged_plot_path.is_file()

    merged_rois = qcfile.get_rois_from_group(qcfile.get_last_group())
    n_merged_roi = len(merged_rois)
    assert n_merged_roi > 0
    assert n_merged_roi <= n_raw_roi

    # check that pixels were conserved
    merged_roi_pixels = pixel_set_from_roi_list(merged_rois)
    assert merged_roi_pixels == raw_roi_pixels


@pytest.mark.skipif(sys.version_info.minor < 8,
                    reason='tests started failing in 3.7')
def test_edge_hnc_filter_merge(tmpdir, synthetic_video_path):
    """
    test HNC segmentation pipeline
    """

    tmpdir_path = pathlib.Path(tmpdir)
    graph_path = tmpdir_path / 'edge_graph.pkl'
    graph_plot_path = tmpdir_path / 'edge_graph.png'

    # calculate edges

    data = {'video_path': str(synthetic_video_path),
            'graph_output': str(graph_path),
            'plot_output': str(graph_plot_path),
            'attribute': 'filtered_hnc_Gaussian',
            'filter_fraction': 0.2,
            'neighborhood_radius': 7}

    edge_runner = CalculateEdges(input_data=data, args=[])
    edge_runner.run()

    assert graph_path.is_file()
    assert graph_plot_path.is_file()

    # find ROIs
    log_path = tmpdir_path/'hnc_qc.h5'
    roi_plot_path = tmpdir_path/'hnc_plot.png'

    data = {'video_input': str(synthetic_video_path),
            'graph_input': str(graph_path),
            'log_path': str(log_path),
            'plot_output': str(roi_plot_path),
            'attribute': 'filtered_hnc_Gaussian',
            'experiment_name': 'test',
            'seeder_args': {'keep_fraction': 0.05},
            'hnc_args': {'postprocessor_min_cell_size': 4,
                         'postprocessor_preferred_cell_size': 15}}

    hnc_runner = HNCSegmentationWrapper(
                     input_data=data, args=[])
    hnc_runner.run()
    assert log_path.is_file()
    assert roi_plot_path.is_file()

    processing_log = SegmentationProcessingLog(log_path, read_only=True)
    raw_rois = processing_log.get_rois_from_group(
            group_name=processing_log.get_last_group())
    n_raw_roi = len(raw_rois)
    assert n_raw_roi > 1

    raw_roi_pixels = pixel_set_from_roi_list(raw_rois)

    # choose filter areas so that we don't lose
    # all ROIs
    areas = np.unique(np.array([np.array(roi['mask']).sum()
                                for roi in raw_rois]))
    min_area = areas[1]
    max_area = areas[-2]

    # filter on area
    data = {'pipeline_stage': 'area filter',
            'log_path': str(log_path),
            'max_area': max_area,
            'min_area': min_area}
    filter_runner = AreaFilterRunner(input_data=data, args=[])
    filter_runner.run()

    filtered_rois = processing_log.get_rois_from_group(
            group_name=processing_log.get_last_group())

    # verify pixels were conserved in filtering
    filtered_pixels = pixel_set_from_roi_list(filtered_rois)
    assert filtered_pixels == raw_roi_pixels

    n_valid_filtered_roi = 0
    for roi in filtered_rois:
        if roi['valid']:
            n_valid_filtered_roi += 1
    assert n_valid_filtered_roi > 0
    assert n_valid_filtered_roi < n_raw_roi
    assert len(filtered_rois) == n_raw_roi

    # merge ROIs
    merged_plot_path = tmpdir_path/'fvs_merged_plot.png'
    data = {'video_input': str(synthetic_video_path),
            'plot_output': str(merged_plot_path),
            'log_path': str(log_path),
            'n_parallel_workers': 2,
            'attribute': 'filtered_hnc_Gaussian'}

    merge_runner = RoiMergerEngine(input_data=data, args=[])
    merge_runner.run()
    assert merged_plot_path.is_file()
    merged_rois = processing_log.get_rois_from_group(
            group_name=processing_log.get_last_group())
    n_merged_roi = len(merged_rois)
    assert n_merged_roi > 0
    assert n_merged_roi <= n_raw_roi

    # check that pixels were conserved
    merged_roi_pixels = pixel_set_from_roi_list(merged_rois)
    assert merged_roi_pixels == raw_roi_pixels

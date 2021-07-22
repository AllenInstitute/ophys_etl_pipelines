import pytest
import pathlib
import h5py
import numpy as np
import json


from ophys_etl.modules.segmentation.modules.calculate_edges import (
    CalculateEdges)

from ophys_etl.modules.segmentation.modules.feature_vector_segmentation import (
    FeatureVectorSegmentationRunner)

from ophys_etl.modules.segmentation.modules.roi_merging import (
    RoiMergerEngine)


@pytest.fixture(scope='session')
def synthetic_video_path(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('end_to_end'))

    video_path = tmpdir/'synthetic_video.h5'

    nt = 100
    nx = 64
    ny = 64

    rng = np.random.default_rng(553311)
    data = rng.normal(10.0, 5.0, size=(nt, nx, ny))
    t_array = np.arange(nt)

    for ii in range(20):
        r0 = rng.integers(0, 50)
        c0 = rng.integers(0, 50)
        n_peaks = rng.integers(3, 6)
        t_peak = rng.integers(5, 100, size=n_peaks)
        width = min(rng.integers(5, 10), 64-c0)
        height = min(rng.integers(5, 10), 64-r0)

        for peak in t_peak:
            amp = 5.0+10.0*rng.random()
            sigma = 1.0+2.0*rng.random()
            flux = amp*np.exp(-0.5*((t_array-peak)/sigma)**2)
            flux /= sigma*np.sqrt(2.0*np.pi)
            for r in range(height):
                for c in range(width):
                    valid = rng.integers(0,2)
                    if valid == 0:
                        continue
                    noise = rng.normal(1.0, 0.3, size=nt)
                    data[:, r0+r, c0+c] += flux+noise
    data -= data.min()
    data = np.round((2**16-1)*data/data.max()).astype(np.uint16)
    assert data.min() >= 0
    assert data.max() < 2**16

    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    yield video_path


def test_edge_fvs_merge_filter(tmpdir, synthetic_video_path):

    tmpdir_path = pathlib.Path(tmpdir)
    graph_path = tmpdir_path/'edge_graph.pkl'
    graph_plot_path = tmpdir_path/'edge_graph.png'

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

    roi_raw_path = tmpdir_path/'fvs_rois.json'
    qc_path = tmpdir_path/'fvs_qc.h5'
    roi_plot_path = tmpdir_path/'fvs_plot.png'

    data = {'video_input': str(synthetic_video_path),
            'graph_input': str(graph_path),
            'roi_output': str(roi_raw_path),
            'qc_output': str(qc_path),
            'plot_output': str(roi_plot_path),
            'attribute': 'filtered_hnc_Gaussian',
            'filter_fraction': 0.2,
            'n_parallel_workers': 2,
            'seeder_args': {'keep_fraction': 0.01,
                            'minimum_distance': 20,
                            'n_samples': 6}}

    fvs_runner = FeatureVectorSegmentationRunner(
                     input_data=data, args=[])
    fvs_runner.run()
    assert roi_raw_path.is_file()
    assert qc_path.is_file()
    assert roi_plot_path.is_file()

    with open(roi_raw_path, 'rb') as in_file:
        raw_rois = json.load(in_file)
    n_raw_roi = len(raw_rois)
    assert n_raw_roi > 0
    with h5py.File(qc_path, 'r') as in_file:
        assert 'detect' in in_file
        assert 'seed' in in_file

    roi_merged_path = tmpdir_path/'fvs_merged_rois.json'
    merged_plot_path = tmpdir_path/'fvs_merged_plot.png'
    data = {'video_input': str(synthetic_video_path),
            'roi_input': str(roi_raw_path),
            'roi_output': str(roi_merged_path),
            'plot_output': str(merged_plot_path),
            'qc_output': str(qc_path),
            'n_parallel_workers': 2,
            'attribute': 'filtered_hnc_Gaussian'}

    merge_runner = RoiMergerEngine(input_data=data, args=[])
    merge_runner.run()
    assert roi_merged_path.is_file()
    assert merged_plot_path.is_file()
    with open(roi_merged_path, 'rb') as in_file:
        merged_rois = json.load(in_file)
    n_merged_roi = len(merged_rois)
    assert n_merged_roi > 0
    assert n_merged_roi < n_raw_roi

import matplotlib

import pytest

import pathlib
import json
import h5py
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane

from ophys_etl.modules.decrosstalk.qc_plotting.pairwise_plot import (
    get_roi_pixels,
    find_overlapping_roi_pairs)

from ophys_etl.modules.decrosstalk.qc_plotting import (
    generate_roi_figure,
    generate_pairwise_figures)

matplotlib.use('Agg')


def test_get_roi_pixels():
    """
    Test method that maps a list of ROIs to a dict of pixels
    """

    roi_list = []
    expected_pixel_set = {}

    roi = OphysROI(x0=4,
                   y0=17,
                   width=5,
                   height=3,
                   mask_matrix=[[True, False, False, True, True],
                                [False, False, False, True, False],
                                [True, True, False, False, False]],
                   roi_id=0,
                   valid_roi=True)

    roi_list.append(roi)
    expected_pixel_set[0] = set([(4, 17), (7, 17), (8, 17), (7, 18),
                                 (4, 19), (5, 19)])

    roi = OphysROI(x0=9,
                   y0=7,
                   width=2,
                   height=5,
                   mask_matrix=[[False, False],
                                [True, False],
                                [False, True],
                                [False, False],
                                [True, True]],
                   roi_id=1,
                   valid_roi=True)

    roi_list.append(roi)
    expected_pixel_set[1] = set([(9, 8), (10, 9), (9, 11), (10, 11)])

    result = get_roi_pixels(roi_list)

    assert len(result) == 2
    assert 0 in result
    assert 1 in result

    assert result[0] == expected_pixel_set[0]
    assert result[1] == expected_pixel_set[1]


def test_find_overlapping_roi_pairs():

    roi_list_0 = []
    roi_list_1 = []

    roi = OphysROI(x0=4, y0=5,
                   width=4, height=3,
                   mask_matrix=[[True, False, False, True],
                                [False, True, True, False],
                                [False, True, False, False]],
                   roi_id=0,
                   valid_roi=True)

    roi_list_0.append(roi)

    # photo-negative of roi_0
    roi = OphysROI(x0=4, y0=5,
                   width=4, height=3,
                   mask_matrix=[[False, True, True, False],
                                [True, False, False, True],
                                [True, False, True, True]],
                   roi_id=1,
                   valid_roi=True)

    roi_list_1.append(roi)

    # intersects one point of roi_0
    roi = OphysROI(x0=6, y0=6,
                   width=2, height=3,
                   mask_matrix=[[True, False],
                                [True, True],
                                [True, True]],
                   roi_id=2,
                   valid_roi=True)

    roi_list_1.append(roi)

    # no intersection with roi_0
    roi = OphysROI(x0=6, y0=6,
                   width=2, height=3,
                   mask_matrix=[[False, False],
                                [True, True],
                                [True, True]],
                   roi_id=3,
                   valid_roi=True)

    roi_list_1.append(roi)

    # one corner overlaps with roi_2 and roi_3
    roi = OphysROI(x0=7, y0=8,
                   width=3, height=4,
                   mask_matrix=[[True, True, False],
                                [True, True, True],
                                [True, True, True],
                                [True, True, True]],
                   roi_id=4,
                   valid_roi=True)

    roi_list_0.append(roi)

    # no overlaps
    roi = OphysROI(x0=7, y0=8,
                   width=3, height=4,
                   mask_matrix=[[False, False, False],
                                [True, True, True],
                                [True, True, True],
                                [True, True, True]],
                   roi_id=5,
                   valid_roi=True)

    roi_list_0.append(roi)

    overlap_list = find_overlapping_roi_pairs(roi_list_0,
                                              roi_list_1)

    assert len(overlap_list) == 3
    assert (0, 2, 1/5, 1/5) in overlap_list
    assert (4, 3, 1/11, 1/4) in overlap_list
    assert (4, 2, 1/11, 1/5) in overlap_list


def test_summary_plot_generation(tmpdir):
    """
    Run a smoke test on qc_plotting.generate_roi_figure
    """
    this_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = this_dir / 'data/qc_plotting'

    input_json_name = data_dir / 'DECROSSTALK_example_input.json'
    with open(input_json_name, 'rb') as in_file:
        src_data = json.load(in_file)

    plane_list = []
    for pair in src_data['coupled_planes']:
        for plane in pair['planes']:

            # redirect maximum projection path
            orig = pathlib.Path(plane['maximum_projection_image_file']).name
            new = data_dir / orig
            plane['maximum_projection_image_file'] = new
            p = DecrosstalkingOphysPlane.from_schema_dict(plane)

            # add QC data file path to plane
            qc_name = data_dir / f'{p.experiment_id}_qc_data.h5'

            p.qc_file_path = qc_name
            plane_list.append(p)

    ophys_planes = []
    for ii in range(0, len(plane_list), 2):
        ophys_planes.append((plane_list[ii], plane_list[ii+1]))

    with open(data_dir / 'DECROSSTALK_example_output.json', 'rb') as in_file:
        output_data = json.load(in_file)

    out_path = tmpdir / 'summary.png'
    assert not out_path.exists()

    generate_roi_figure(1071648245,
                        ophys_planes,
                        output_data,
                        tmpdir / 'summary.png')

    assert out_path.isfile()


@pytest.fixture
def expected_pairwise():
    """
    Pairwise plots that should be generated based on the
    data in resources/
    """
    expected_files = []
    dirname = pathlib.Path('1071738390_1071738393_roi_pairs')
    for fname in ('1080616650_1080616555_comparison.png',
                  '1080616658_1080616553_comparison.png',
                  '1080616659_1080616554_comparison.png'):
        expected_files.append(dirname / fname)

    dirname = pathlib.Path('1071738394_1071738396_roi_pairs')
    for fname in ('1080618091_1080616600_comparison.png',
                  '1080618102_1080616618_comparison.png'):
        expected_files.append(dirname / fname)

    dirname = pathlib.Path('1071738397_1071738399_roi_pairs')
    for fname in ('1080618093_1080623135_comparison.png',
                  '1080618114_1080623164_comparison.png'):
        expected_files.append(dirname / fname)

    dirname = pathlib.Path('1071738400_1071738402_roi_pairs')
    for fname in ('1080616774_1080622865_comparison.png',
                  '1080616776_1080622881_comparison.png'):
        expected_files.append(dirname / fname)
    return expected_files


def test_pairwise_plot_generation(tmpdir, expected_pairwise):
    """
    Run a smoke test on qc_plotting.generate_pairwise_figures
    """
    this_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = this_dir / 'data/qc_plotting'

    input_json_name = data_dir / 'DECROSSTALK_example_input.json'
    with open(input_json_name, 'rb') as in_file:
        src_data = json.load(in_file)

    plane_list = []
    for pair in src_data['coupled_planes']:
        for plane in pair['planes']:

            # redirect maximum projection path
            orig = pathlib.Path(plane['maximum_projection_image_file']).name
            new = data_dir / orig
            plane['maximum_projection_image_file'] = new
            p = DecrosstalkingOphysPlane.from_schema_dict(plane)

            # add QC data file path to plane
            qc_name = data_dir / f'{p.experiment_id}_qc_data.h5'

            p.qc_file_path = qc_name
            plane_list.append(p)

    ophys_planes = []
    for ii in range(0, len(plane_list), 2):
        ophys_planes.append((plane_list[ii], plane_list[ii+1]))

    generate_pairwise_figures(ophys_planes,
                              tmpdir)

    expected_files = set()
    t = pathlib.Path(tmpdir)
    for plot_path in expected_pairwise:
        fname = t / plot_path
        expected_files.add(fname)

    for fname in expected_files:
        if not fname.is_file():
            raise RuntimeError(f"could not find {fname.resolve()}")

    # check that only the expected files are created
    file_list = pathlib.Path(tmpdir).glob('**/*')
    ct = 0
    for fname in file_list:
        if not fname.is_file():
            continue
        ct += 1
        assert fname in expected_files
    assert ct == len(expected_files)


@pytest.mark.parametrize('mangling_operation', ['some', 'all',
                                                'both_some',
                                                'both_all'])
def test_pairwise_plot_generation_nans(tmpdir,
                                       expected_pairwise,
                                       mangling_operation):
    """
    Run a smoke test on qc_plotting.generate_pairwise_figures
    in the case where data being plotted contains NaNs

    Do this by creating copies of the HDF5 files in data/qc_plotting
    with mangled traces

    mangling_operation indicates how we want the ROI pair to
    be mangled:
        'some' -- one ROI has some NaNs in its traces
        'all' -- one ROI has all of its traces set to NaN
        'both_some' -- both ROIs have some NaNs in their traces
        'both_all' -- both ROIs have all of their traces set to NaN
    """

    this_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = this_dir / 'data/qc_plotting'
    plotting_dir = pathlib.Path(tmpdir)/'mangled_plots'

    # directory where we will write the HDF5 files with
    # traces that contain NaNs
    mangled_data_dir = pathlib.Path(tmpdir)/'mangled_data'

    input_json_name = data_dir / 'DECROSSTALK_example_input.json'
    with open(input_json_name, 'rb') as in_file:
        src_data = json.load(in_file)

    # list of planes that need mangling;
    # will be tuples of the form (unmangled_fname, mangled_fname)
    planes_to_mangle = []

    plane_list = []
    for pair in src_data['coupled_planes']:
        for i_plane, plane in enumerate(pair['planes']):

            # redirect maximum projection path
            orig = pathlib.Path(plane['maximum_projection_image_file']).name
            new = data_dir / orig
            plane['maximum_projection_image_file'] = new
            p = DecrosstalkingOphysPlane.from_schema_dict(plane)

            # add QC data file path to plane
            local_fname = f'{p.experiment_id}_qc_data.h5'
            if i_plane == 0 or 'both' in mangling_operation:
                qc_name = mangled_data_dir / local_fname
                orig_qc_name = data_dir / local_fname
                planes_to_mangle.append((orig_qc_name, qc_name))
            else:
                qc_name = data_dir / local_fname

            p.qc_file_path = qc_name
            plane_list.append(p)

    # add NaNs to the traces designated for mangling #############3

    rng = np.random.RandomState(1723124)

    def _copy_data(dataset_name,
                   in_file_handle,
                   out_file_handle,
                   operation):
        """
        Copy the dataset 'dataset_name' from in_file_handle
        to out_file_handle, mangling as specified by `operation`
        """
        data = in_file_handle[dataset_name]
        if not isinstance(data, h5py.Dataset):
            key_list = list(data.keys())
            for key in key_list:
                new_name = f'{dataset_name}/{key}'
                _copy_data(new_name,
                           in_file_handle,
                           out_file_handle,
                           operation)
        else:
            v = data[()]
            if 'signal/trace' in dataset_name:
                if 'some' in operation:
                    dexes = np.arange(len(v), dtype=int)
                    chosen = rng.choice(dexes, len(v)//4, replace=True)
                    chosen = np.unique(chosen)
                    v[chosen] = np.NaN
                elif 'all' in operation:
                    v[:] = np.NaN
                else:
                    raise RuntimeError("cannot interpret "
                                       f"operation: {operation}")
            out_file_handle.create_dataset(dataset_name,
                                           data=v)

    def copy_mangled_h5py(in_fname, out_fname, operation):
        """
        in_fname -- path to the original data file
        out_fname -- path to the new data file
        operation -- 'some' sets some trace values to NaN;
                     'all' sets all trace values to NaN

        Note: operation can also be 'both_some' or 'both_all'
        indicating that both ROIs in the plot have been
        mangled
        """
        assert not out_fname.exists()
        out_fname.parent.mkdir(parents=True, exist_ok=True)
        assert in_fname.is_file()
        with h5py.File(out_fname, 'w') as out_handle:
            with h5py.File(in_fname, 'r') as in_handle:
                key_list = list(in_handle.keys())
                for key in key_list:
                    _copy_data(key, in_handle, out_handle, operation)

    for plane in planes_to_mangle:
        copy_mangled_h5py(plane[0], plane[1], mangling_operation)

    # done mangling #####################

    # proceed with plotting test as usual

    ophys_planes = []
    for ii in range(0, len(plane_list), 2):
        ophys_planes.append((plane_list[ii], plane_list[ii+1]))

    generate_pairwise_figures(ophys_planes,
                              plotting_dir)

    expected_files = set()
    for plot_path in expected_pairwise:
        fname = plotting_dir / plot_path
        expected_files.add(fname)

    for fname in expected_files:
        if not fname.is_file():
            raise RuntimeError(f"could not find {fname.resolve()}")

    # check that only the expected files are created
    file_list = plotting_dir.glob('**/*')
    ct = 0
    for fname in file_list:
        if not fname.is_file():
            continue
        ct += 1
        assert fname in expected_files
    assert ct == len(expected_files)

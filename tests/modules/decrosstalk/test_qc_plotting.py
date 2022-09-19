import matplotlib

import pytest

import pathlib
import json
import h5py
import numpy as np
import PIL

from ophys_etl.types import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane

from ophys_etl.modules.decrosstalk.qc_plotting.pairwise_plot import (
    get_roi_pixels,
    find_overlapping_roi_pairs,
    get_img_thumbnails,
    get_nanmaxed_timestep,
    get_most_active_section)

from ophys_etl.modules.decrosstalk.qc_plotting import (
    generate_roi_figure,
    generate_pairwise_figures)

matplotlib.use('Agg')


@pytest.fixture
def decrosstalk_data_dir():
    this_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = this_dir / 'data/qc_plotting'
    return data_dir


@pytest.fixture
def decrosstalk_input_json_name(
        decrosstalk_data_dir):
    input_json_name = decrosstalk_data_dir / 'DECROSSTALK_example_input.json'
    return input_json_name


@pytest.fixture
def decrosstalk_output_json_name(
        decrosstalk_data_dir):
    output_json_name = decrosstalk_data_dir / 'DECROSSTALK_example_output.json'
    return output_json_name


@pytest.mark.parametrize(
        "trace, maxdex, maxval",
        [(np.array([np.NaN, np.NaN, np.NaN]), -1, None),
         (np.array([1.0, np.NaN, 1.2, 0.9]), 2, 0.2),
         (np.array([1.0, 2.2, 1.3, 0.8]), 1, 1.05)])
def test_get_nanmaxed_timestep(
        trace, maxdex, maxval):
    (actual_dex, actual_val) = get_nanmaxed_timestep(trace)
    assert actual_dex == maxdex
    if maxval is None:
        assert actual_val is None
    else:
        np.testing.assert_allclose(actual_val, maxval)


@pytest.mark.parametrize(
        "trace0, trace1, n_timesteps, expected",
        [(np.array([np.NaN]*20),
          np.array([np.NaN]*20),
          10, (0, 10)),
         (np.array([np.NaN]*20),
          np.array([np.NaN]*20),
          100, (0, 20)),
         (np.array([1.0, np.NaN, 2.0, 5.0, 4.0, 2.0]),
          np.array([0.1, 0.2, np.NaN, 0.1, 0.2, 0.3]),
          4, (1, 5)),
         (np.array([1.0, np.NaN, 2.0, 5.0, 4.0, 2.0]),
          np.array([10.1, 10.2, np.NaN, 10.1, 10.2, 10.3]),
          4, (1, 5)),
         (np.array([1.0, np.NaN, 2.0, 5.0, 4.0, 2.0]),
          np.array([0.1, 0.2, np.NaN, 0.1, 0.2, 0.3]),
          40, (0, 6)),
         (np.array([0.1, 1.2, 0.5, 1.0, np.NaN, 2.0, 5.0, 4.0, 2.0]),
          np.array([0.1, 0.2, np.NaN, 0.1, 0.2, 0.3, np.NaN, 0.2, 0.2]),
          7, (2, 9)),
         (np.array([0.1, 0.2, np.NaN, 0.1, 0.2, 0.3, np.NaN, 0.2, 0.2]),
          np.array([0.1, 1.2, 0.5, 1.0, np.NaN, 2.0, 5.0, 4.0, 2.0]),
          7, (2, 9)),
         (np.array([0.1, 0.2, np.NaN, 0.1, 0.2, 0.3, np.NaN, 0.2, 0.2]),
          np.array([0.1, 7.8, 0.5, 1.0, np.NaN, 2.0, 5.0, 4.0, 2.0]),
          7, (0, 7)),
         (np.array([np.NaN]*20),
          np.array([np.NaN]*20),
          6, (0, 6))])
def test_get_most_active_section(
        trace0,
        trace1,
        n_timesteps,
        expected):
    actual = get_most_active_section(
                trace0=trace0,
                trace1=trace1,
                n_timesteps=n_timesteps)
    assert expected == actual


def test_get_most_active_section_masked():
    """
    Test that get_most_active_section ignores
    timesteps without altering input when n_ignore > 0
    """
    rng = np.random.default_rng(2233)
    trace0 = np.array([0.1]*10)
    trace1 = rng.random(size=10)
    trace1[3] = 100.0
    trace1[6] = 50.0
    trace1_copy = np.copy(trace1)

    result = get_most_active_section(
                trace0=trace0,
                trace1=trace1,
                n_timesteps=5,
                n_ignore=5)

    assert result == (4, 9)
    np.testing.assert_allclose(
            trace0,
            np.array([0.1]*10))
    np.testing.assert_allclose(
            trace1,
            trace1_copy)


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


@pytest.mark.parametrize('x0,y0,x1,y1',
                         [(10, 11, 222, 301),
                          (0, 14, 150, 400),     # extreme xmin
                          (10, 0, 111, 222),     # extreme ymin
                          (0, 0, 400, 100),      # one ROI at origin
                          (509, 111, 13, 18),    # extreme xmax
                          (112, 508, 75, 200),   # extreme ymax
                          (509, 508, 112, 113),  # one ROI at extreme corner
                          (100, 200, 300, 200)
                          ])
def test_get_img_thumbnails(x0, y0, x1, y1, decrosstalk_data_dir):
    """
    Test that, when fed two ROIs that are very distant and do not
    overlap, get_img_thumbnails returns bounds
    (xmin, xmax), (ymin, ymax) bounds that cover both ROIs
    """

    roi0 = OphysROI(x0=x0, y0=y0,
                    width=3, height=4,
                    mask_matrix=[[False, False, False],
                                 [True, True, True],
                                 [True, True, True],
                                 [True, True, True]],
                    roi_id=5,
                    valid_roi=True)

    roi1 = OphysROI(x0=x1, y0=y1,
                    width=3, height=4,
                    mask_matrix=[[False, False, False],
                                 [True, True, True],
                                 [True, True, True],
                                 [True, True, True]],
                    roi_id=5,
                    valid_roi=True)

    img_fname = '1071738402_suite2p_maximum_projection.png'
    raw_img = PIL.Image.open(decrosstalk_data_dir / img_fname, mode='r')
    n_rows = raw_img.size[0]
    n_cols = raw_img.size[1]
    max_img = np.array(raw_img).reshape(n_rows, n_cols)

    (img0,
     img1,
     (xmin, xmax),
     (ymin, ymax)) = get_img_thumbnails(roi0,
                                        roi1,
                                        max_img,
                                        max_img)

    mask = roi0.mask_matrix
    for ix in range(mask.shape[1]):
        xx = ix + roi0.x0
        for iy in range(mask.shape[0]):
            yy = iy + roi0.y0
            if mask[iy, ix]:
                assert yy >= ymin
                assert yy < ymax
                assert xx >= xmin
                assert xx < xmax

    mask = roi1.mask_matrix
    for ix in range(mask.shape[1]):
        xx = ix + roi1.x0
        for iy in range(mask.shape[0]):
            yy = iy + roi1.y0
            if mask[iy, ix]:
                assert yy >= ymin
                assert yy < ymax
                assert xx >= xmin
                assert xx < xmax
    assert xmin >= 0
    assert ymin >= 0
    assert xmax <= 512
    assert ymax <= 512


def test_summary_plot_generation(
        tmpdir,
        decrosstalk_data_dir,
        decrosstalk_input_json_name,
        decrosstalk_output_json_name):
    """
    Run a smoke test on qc_plotting.generate_roi_figure
    """

    with open(decrosstalk_input_json_name, 'rb') as in_file:
        src_data = json.load(in_file)

    plane_list = []
    for pair in src_data['coupled_planes']:
        for plane in pair['planes']:

            # redirect maximum projection path
            orig = pathlib.Path(plane['maximum_projection_image_file']).name
            new = decrosstalk_data_dir / orig
            plane['maximum_projection_image_file'] = new
            p = DecrosstalkingOphysPlane.from_schema_dict(plane)

            # add QC data file path to plane
            qc_name = decrosstalk_data_dir / f'{p.experiment_id}_qc_data.h5'

            p.qc_file_path = qc_name
            plane_list.append(p)

    ophys_planes = []
    for ii in range(0, len(plane_list), 2):
        ophys_planes.append((plane_list[ii], plane_list[ii+1]))

    with open(decrosstalk_output_json_name, 'rb') as in_file:
        output_data = json.load(in_file)

    out_path = tmpdir / 'summary.png'
    assert not out_path.exists()

    generate_roi_figure(1071648245,
                        ophys_planes,
                        output_data,
                        tmpdir / 'summary.png')

    assert out_path.isfile()


@pytest.fixture
def expected_pairwise(
        decrosstalk_input_json_name):
    """
    Pairwise plots that should be generated based on the
    data in resources/
    """

    # create a lookup table mapping ROI_ID to cell_num
    roi_id_to_cell_num = dict()
    with open(decrosstalk_input_json_name, 'rb') as in_file:
        json_data = json.load(in_file)
    for plane_pair in json_data['coupled_planes']:
        for plane in plane_pair['planes']:
            these_rois = []
            for this_roi in plane['rois']:
                these_rois.append(this_roi['id'])
            min_roi = min(these_rois)
            for this_roi in these_rois:
                roi_id_to_cell_num[this_roi] = this_roi-min_roi

    def _extend_expected_files(
            dirname=None,
            roi_pair_set=None,
            roi_id_to_cell_num=None):

        expected_files = []

        for roi_pair in roi_pair_set:
            cell0 = roi_id_to_cell_num[roi_pair[0]]
            cell1 = roi_id_to_cell_num[roi_pair[1]]
            fname = f'cells_{cell0}_{cell1}'
            fname += f'_rois_{roi_pair[0]}_{roi_pair[1]}'
            fname += '_comparison.png'
            expected_files.append(dirname / fname)
        return expected_files

    expected_files = []
    dirname = pathlib.Path('1071738390_1071738393_roi_pairs')
    roi_pair_set = ((1080616650, 1080616555),
                    (1080616658, 1080616553),
                    (1080616659, 1080616554))

    expected_files += _extend_expected_files(
        dirname=dirname,
        roi_pair_set=roi_pair_set,
        roi_id_to_cell_num=roi_id_to_cell_num)

    dirname = pathlib.Path('1071738394_1071738396_roi_pairs')
    roi_pair_set = ((1080618091, 1080616600),
                    (1080618102, 1080616618))

    expected_files += _extend_expected_files(
        dirname=dirname,
        roi_pair_set=roi_pair_set,
        roi_id_to_cell_num=roi_id_to_cell_num)

    dirname = pathlib.Path('1071738397_1071738399_roi_pairs')
    roi_pair_set = ((1080618093, 1080623135),
                    (1080618114, 1080623164))

    expected_files += _extend_expected_files(
        dirname=dirname,
        roi_pair_set=roi_pair_set,
        roi_id_to_cell_num=roi_id_to_cell_num)

    dirname = pathlib.Path('1071738400_1071738402_roi_pairs')
    roi_pair_set = ((1080616774, 1080622865),
                    (1080616776, 1080622881))

    expected_files += _extend_expected_files(
        dirname=dirname,
        roi_pair_set=roi_pair_set,
        roi_id_to_cell_num=roi_id_to_cell_num)

    return expected_files


def test_pairwise_plot_generation(
            tmpdir,
            expected_pairwise,
            decrosstalk_data_dir,
            decrosstalk_input_json_name,
            helper_functions):
    """
    Run a smoke test on qc_plotting.generate_pairwise_figures
    """
    with open(decrosstalk_input_json_name, 'rb') as in_file:
        src_data = json.load(in_file)

    plane_list = []
    for pair in src_data['coupled_planes']:
        for plane in pair['planes']:

            # redirect maximum projection path
            orig = pathlib.Path(plane['maximum_projection_image_file']).name
            new = decrosstalk_data_dir / orig
            plane['maximum_projection_image_file'] = new
            p = DecrosstalkingOphysPlane.from_schema_dict(plane)

            # add QC data file path to plane
            qc_name = decrosstalk_data_dir / f'{p.experiment_id}_qc_data.h5'

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
        assert fname.is_file()

    # check that only the expected files are created
    file_list = pathlib.Path(tmpdir).glob('**/*')
    ct = 0
    for fname in file_list:
        if not fname.is_file():
            continue
        ct += 1
        assert fname in expected_files
    assert ct == len(expected_files)

    helper_functions.clean_up_dir(tmpdir=tmpdir)


@pytest.mark.parametrize('mangling_operation', ['some', 'all',
                                                'both_some',
                                                'both_all',
                                                'both_value'])
def test_pairwise_plot_generation_nans(
        tmpdir,
        decrosstalk_data_dir,
        decrosstalk_input_json_name,
        expected_pairwise,
        mangling_operation,
        helper_functions):
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
        'both_value' -- both ROIs have their traces set to 1.0
                        (this exercises the case in which the
                        min and max values for the 2D histogram
                        are identical)
    """

    plotting_dir = pathlib.Path(tmpdir)/'mangled_plots'

    # directory where we will write the HDF5 files with
    # traces that contain NaNs
    mangled_data_dir = pathlib.Path(tmpdir)/'mangled_data'

    with open(decrosstalk_input_json_name, 'rb') as in_file:
        src_data = json.load(in_file)

    # list of planes that need mangling;
    # will be tuples of the form (unmangled_fname, mangled_fname)
    planes_to_mangle = []

    plane_list = []
    for pair in src_data['coupled_planes']:
        for i_plane, plane in enumerate(pair['planes']):

            # redirect maximum projection path
            orig = pathlib.Path(plane['maximum_projection_image_file']).name
            new = decrosstalk_data_dir / orig
            plane['maximum_projection_image_file'] = new
            p = DecrosstalkingOphysPlane.from_schema_dict(plane)

            # add QC data file path to plane
            local_fname = f'{p.experiment_id}_qc_data.h5'
            if i_plane == 0 or 'both' in mangling_operation:
                qc_name = mangled_data_dir / local_fname
                orig_qc_name = decrosstalk_data_dir / local_fname
                planes_to_mangle.append((orig_qc_name, qc_name))
            else:
                qc_name = decrosstalk_data_dir / local_fname

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
                elif 'value' in operation:
                    v[:] = 1.0
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
                     'both_value' sets all trace values to 1.0

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
        assert fname.is_file()

    # check that only the expected files are created
    file_list = plotting_dir.glob('**/*')
    ct = 0
    for fname in file_list:
        if not fname.is_file():
            continue
        ct += 1
        assert fname in expected_files
    assert ct == len(expected_files)

    helper_functions.clean_up_dir(tmpdir=tmpdir)

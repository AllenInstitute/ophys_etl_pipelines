import os
import tempfile
import numpy as np
import h5py
from ophys_etl.decrosstalk.ophys_plane import OphysPlane

from .utils import teardown_function  # noqa F401
from .utils import get_tmp_dir
from .utils import get_data_dir


def setup_function(function):

    data_dir = get_data_dir()

    parent_tmp = get_tmp_dir()
    input_dir = os.path.join(parent_tmp,
                             'pipeline_test_input')

    function._input_dir = input_dir
    function._temp_files = []

    nt = 10000  # truncate signals in memory

    good_trace_file = os.path.join(data_dir, 'good_traces.h5')
    with h5py.File(good_trace_file, 'r') as in_file:
        all_signals = in_file['signal'][()]
        all_crosstalk = in_file['crosstalk'][()]
        signal_0 = all_signals[0, :nt]
        crosstalk_0 = all_crosstalk[0, :nt]
        signal_1 = all_signals[1, :nt]
        crosstalk_1 = all_crosstalk[1, :nt]

    nx = 80
    ny = 90

    movie_fname_0 = tempfile.mkstemp(dir=input_dir,
                                     prefix='movie_0_',
                                     suffix='.h5')[1]

    movie_fname_1 = tempfile.mkstemp(dir=input_dir,
                                     prefix='movie_0_',
                                     suffix='.h5')[1]

    base_m_0 = os.path.basename(movie_fname_0)
    function._temp_files.append(os.path.join(input_dir,
                                             base_m_0))
    base_m_1 = os.path.basename(movie_fname_1)
    function._temp_files.append(os.path.join(input_dir,
                                             base_m_1))

    roi0 = {}
    roi0['id'] = 0
    roi0['x'] = 15
    roi0['y'] = 33
    roi0['width'] = 7
    roi0['height'] = 5
    roi0['valid_roi'] = True
    roi0['mask_matrix'] = [[True, True, False, True, True, False, False],
                           [False, True, True, True, True, False, False],
                           [False, False, True, True, True, True, True],
                           [False, False, False, True, True, True, True],
                           [True, True, True, True, True, False, False]]

    roi1 = {}
    roi1['id'] = 1
    roi1['x'] = 22
    roi1['y'] = 80
    roi1['width'] = 6
    roi1['height'] = 5
    roi1['valid_roi'] = True
    roi1['mask_matrix'] = [[True, True, False, True, True, False],
                           [True, True, True, True, False, False],
                           [False, False, True, True, True, True],
                           [False, False, True, True, True, True],
                           [True, True, True, True, False, False],
                           [False, False, True, True, True, True],
                           [False, False, False, True, True, True]]

    plane0 = {}
    plane0['ophys_experiment_id'] = 0
    plane0['motion_corrected_stack'] = movie_fname_0
    plane0['motion_border'] = {'x0': 2, 'x1': 2,
                               'y0': 3, 'y1': 3}
    plane0['rois'] = [roi0, roi1]

    plane1 = {}
    plane1['ophys_experiment_id'] = 1
    plane1['motion_corrected_stack'] = movie_fname_1
    plane1['motion_border'] = {'x0': 3, 'x1': 2,
                               'y0': 3, 'y1': 3}
    plane1['rois'] = [roi0, roi1]

    movie_0 = np.zeros((nt, ny, nx), dtype=float)
    movie_1 = np.zeros((nt, ny, nx), dtype=float)

    for iy in range(33, 38):
        for ix in range(15, 22):
            movie_0[:, iy, ix] += signal_0
            movie_1[:, iy, ix] += crosstalk_0

    for iy in range(80, 85):
        for ix in range(22, 28):
            movie_1[:, iy, ix] += signal_1
            movie_0[:, iy, ix] += crosstalk_1

    with h5py.File(movie_fname_0, 'w') as out_file:
        out_file.create_dataset('data', data=movie_0)
    with h5py.File(movie_fname_1, 'w') as out_file:
        out_file.create_dataset('data', data=movie_1)

    output_dir = os.path.join(parent_tmp,
                              'pipeline_test_output')

    session = {}
    session['ophys_session_id'] = 0
    session['qc_output_dir'] = output_dir
    session['coupled_planes'] = []
    pair = {}
    pair['ophys_imaging_plane_group_id'] = 0
    pair['group_order'] = 0
    pair['planes'] = [plane0, plane1]
    session['coupled_planes'].append(pair)

    function._session = session


def test_full_pipeline():

    session = test_full_pipeline._session

    pair = session['coupled_planes'][0]['planes']
    plane0 = OphysPlane.from_schema_dict(pair[0])
    plane1 = OphysPlane.from_schema_dict(pair[1])
    plane0.run_decrosstalk(plane1,
                           cache_dir=session['qc_output_dir'],
                           clobber=True)
    plane1.run_decrosstalk(plane0,
                           cache_dir=session['qc_output_dir'],
                           clobber=True)

    roi_suffixes = ['raw.h5', 'raw_at.h5', 'out.h5', 'out_at.h5',
                    'valid.json', 'out_valid.json', 'crosstalk.json',
                    'valid_ct.json']

    neuropil_suffixes = ['raw.h5', 'out.h5',
                         'valid.json', 'out_valid.json']

    expected_files = []
    for prefix in ('0', '1'):
        for suffix in roi_suffixes:
            fname = 'roi_0_1/%s_%s' % (prefix, suffix)
            expected_files.append(fname)

        for suffix in neuropil_suffixes:
            fname = 'neuropil_0_1/%s_%s' % (prefix, suffix)
            expected_files.append(fname)

    output_dir = session['qc_output_dir']
    for fname in expected_files:
        test_full_pipeline._temp_files.append(os.path.join(output_dir,
                                                           fname))
    test_full_pipeline._temp_files.append(os.path.join(output_dir,
                                                       '0_1_invalid_at.json'))
    test_full_pipeline._temp_files.append(os.path.join(output_dir,
                                                       '1_0_invalid_at.json'))

    for fname in expected_files:
        full_name = os.path.join(output_dir, fname)
        msg = 'could not find %s' % full_name
        assert os.path.isfile(full_name), msg

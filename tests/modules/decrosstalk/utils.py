import os
import copy
import h5py
import PIL.Image as Image
import numpy as np
import tempfile


def get_data_dir():
    """
    Return absolute path to tests/decrosstalk/data
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    assert os.path.isdir(data_dir)
    return data_dir


def create_data(tmpdir):
    """
    Setup for test_full_pipeline.

    Writes movie files to a temporary directory

    Creates a dict mimicking the input data required by the
    decrosstalking pipeline.

    Parameters
    ----------
    tmpdir -- a parent temporary directory wher input_dir
    will be created

    Returns
    -------
    The dict mimicking the input schema for the decrosstalking
    pipeline.
    """

    data_dir = get_data_dir()

    parent_tmp = tmpdir
    input_dir = os.path.join(tmpdir,
                             'pipeline_test_input')

    output_dir = os.path.join(parent_tmp,
                              'pipeline_test_output')

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    max_proj_name = tempfile.mkstemp(dir=input_dir,
                                     prefix='max_projection_',
                                     suffix='.png')[1]

    img_data = np.zeros((nx, ny), dtype=np.uint8)
    img = Image.fromarray(img_data)
    img.save(max_proj_name, format='png')

    movie_fname_0 = tempfile.mkstemp(dir=input_dir,
                                     prefix='movie_0_',
                                     suffix='.h5')[1]

    movie_fname_1 = tempfile.mkstemp(dir=input_dir,
                                     prefix='movie_0_',
                                     suffix='.h5')[1]

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
    roi1['height'] = 7
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
    plane0['maximum_projection_image_file'] = max_proj_name
    plane0['motion_border'] = {'x0': 2, 'x1': 2,
                               'y0': 3, 'y1': 3}
    plane0['rois'] = [roi0, roi1]

    roi_fname = os.path.join(output_dir, 'roi_trace_0.h5')
    np_fname = os.path.join(output_dir, 'neuropil_trace_0.h5')
    plane0['output_roi_trace_file'] = roi_fname
    plane0['output_neuropil_trace_file'] = np_fname

    plane1 = {}
    plane1['ophys_experiment_id'] = 1
    plane1['motion_corrected_stack'] = movie_fname_1
    plane1['maximum_projection_image_file'] = max_proj_name
    plane1['motion_border'] = {'x0': 3, 'x1': 2,
                               'y0': 3, 'y1': 3}
    roi2 = copy.deepcopy(roi0)
    roi2['id'] = 2
    roi3 = copy.deepcopy(roi1)
    roi3['id'] = 3
    plane1['rois'] = [roi2, roi3]

    roi_fname = os.path.join(output_dir, 'roi_trace_1.h5')
    np_fname = os.path.join(output_dir, 'neuropil_trace_1.h5')
    plane1['output_roi_trace_file'] = roi_fname
    plane1['output_neuropil_trace_file'] = np_fname

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

    session = {}
    session['ophys_session_id'] = 0
    session['qc_output_dir'] = output_dir
    session['coupled_planes'] = []
    pair = {}
    pair['ophys_imaging_plane_group_id'] = 0
    pair['group_order'] = 0
    pair['planes'] = [plane0, plane1]
    session['coupled_planes'].append(pair)

    return session

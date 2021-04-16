import os
import h5py
import json
import numpy as np
from ophys_etl.modules.decrosstalk.__main__ import DecrosstalkWrapper

from .utils import create_data


def test_run_decrosstalk(tmpdir):
    """
    Test that ophys_etl/transforms/decrosstalk_wrapper
    runs as expected

    Only verifies that expected outputs are created.
    Does not validate contents of those files.
    """

    session = create_data(tmpdir)
    output_json = os.path.join(tmpdir, 'test_output.json')
    wrapper = DecrosstalkWrapper(input_data=session,
                                 args=['--output_json', output_json])
    wrapper.run()

    expected_files = ['0_qc_data.h5', '1_qc_data.h5']

    # load output_json
    exp_id_to_invalid = {}
    with open(output_json, 'rb') as in_file:
        output_data = json.load(in_file)
        for pair in output_data['coupled_planes']:
            for plane in pair['planes']:
                local = set()
                for k in plane:
                    if 'invalid' in k or 'ghost' in k:
                        for roi_id in plane[k]:
                            local.add(roi_id)
                exp_id = plane['ophys_experiment_id']
                exp_id_to_invalid[exp_id] = local

    for pair in session['coupled_planes']:
        for plane in pair['planes']:
            exp_id = plane['ophys_experiment_id']
            roi_name = plane['output_roi_trace_file']
            np_name = plane['output_neuropil_trace_file']
            assert roi_name != np_name
            assert roi_name not in expected_files
            assert np_name not in expected_files
            expected_files.append(roi_name)
            expected_files.append(np_name)

            # Given that it is possible for the code
            # to gracefully exit saving no ROIs,
            # verify that all of these traces are not
            # full of NaNs. Verify that any traces that
            # are full of NaNs have been flagged as
            # invalid for some reason
            for fname in (roi_name, np_name):
                with h5py.File(fname, 'r') as in_file:
                    roi_names = in_file['roi_names'][()]
                    data = in_file['data'][()]
                    assert not np.isnan(data).all()
                    assert len(roi_names) > 0
                    assert data.shape[0] == len(roi_names)
                    for ii, roi_id in enumerate(roi_names):
                        if np.isnan(data[ii, :]).any():
                            assert np.isnan(data[ii, :]).all()
                            assert roi_id in exp_id_to_invalid[exp_id]
                        else:
                            assert roi_id not in exp_id_to_invalid[exp_id]

    output_dir = session['qc_output_dir']
    for fname in expected_files:
        full_name = os.path.join(output_dir, fname)
        msg = 'could not find %s' % full_name
        assert os.path.isfile(full_name), msg


def test_run_decrosstalk_with_empty_rois(tmpdir):
    """
    Test that ophys_etl/transforms/decrosstalk_wrapper
    runs as expected when ROIs are empty

    Only verifies that expected outputs are created.
    Does not validate contents of those files.
    """

    session = create_data(tmpdir)

    session['coupled_planes'][0]['planes'][0]['rois'] = []
    empty_plane = session['coupled_planes'][0]['planes'][0]
    empty_exp_id = empty_plane['ophys_experiment_id']

    output_json = os.path.join(tmpdir, 'test_output.json')
    wrapper = DecrosstalkWrapper(input_data=session,
                                 args=['--output_json', output_json])
    wrapper.run()

    expected_files = ['0_qc_data.h5', '1_qc_data.h5']

    # load output_json
    exp_id_to_invalid = {}
    with open(output_json, 'rb') as in_file:
        output_data = json.load(in_file)
        for pair in output_data['coupled_planes']:
            for plane in pair['planes']:
                local = set()
                for k in plane:
                    if 'invalid' in k or 'ghost' in k:
                        for roi_id in plane[k]:
                            local.add(roi_id)
                exp_id = plane['ophys_experiment_id']
                exp_id_to_invalid[exp_id] = local

    for pair in session['coupled_planes']:
        for plane in pair['planes']:
            exp_id = plane['ophys_experiment_id']
            roi_name = plane['output_roi_trace_file']
            np_name = plane['output_neuropil_trace_file']
            assert roi_name != np_name
            assert roi_name not in expected_files
            assert np_name not in expected_files
            expected_files.append(roi_name)
            expected_files.append(np_name)

            # Given that it is possible for the code
            # to gracefully exit saving no ROIs,
            # verify that all of these traces are not
            # full of NaNs. Verify that any traces that
            # are full of NaNs have been flagged as
            # invalid for some reason
            for fname in (roi_name, np_name):
                with h5py.File(fname, 'r') as in_file:
                    roi_names = in_file['roi_names'][()]
                    data = in_file['data'][()]
                    if exp_id != empty_exp_id:
                        assert not np.isnan(data).all()
                        assert len(roi_names) > 0
                        assert data.shape[0] == len(roi_names)
                        for ii, roi_id in enumerate(roi_names):
                            if np.isnan(data[ii, :]).any():
                                assert np.isnan(data[ii, :]).all()
                                assert roi_id in exp_id_to_invalid[exp_id]
                            else:
                                assert roi_id not in exp_id_to_invalid[exp_id]
                    else:
                        assert len(roi_names) == 0
                        assert data.shape == (0,)

    output_dir = session['qc_output_dir']
    for fname in expected_files:
        full_name = os.path.join(output_dir, fname)
        msg = 'could not find %s' % full_name
        assert os.path.isfile(full_name), msg

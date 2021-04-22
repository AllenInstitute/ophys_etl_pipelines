import h5py
import numpy as np
import itertools
import pathlib

from ophys_etl.modules.decrosstalk.io_utils import write_qc_data
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types


def create_dataset():
    """
    Create a test dataset exercising all possible permutations of
    validity flags.

    Returns
    -------
    dict
        'roi_flags': a dict mimicing the roi_flags returned by the pipeline

        'raw_traces': ROISetDict of valid traces
        'invalid_raw_traces': same as above for invalid raw traces

        'unmixed_traces': ROISetDict of valid unmixed traces
        'invalid_unmixed_traces': same as above fore invalid unmixed traces

        'raw_events': ROIEventSet of activity in valid raw traces
        'invalid_raw_events': same as above for invalid raw traces

        'unmixed_events': ROIEventSet of activity in valid unmixed traces
        'invalid_unmixed_events': same as above for invalid unmixed traces

        'true_flags': ground truth values of validity flags for all ROIs
    """
    rng = np.random.RandomState(172)
    n_t = 10

    raw_traces = dc_types.ROISetDict()
    invalid_raw_traces = dc_types.ROISetDict()
    unmixed_traces = dc_types.ROISetDict()
    invalid_unmixed_traces = dc_types.ROISetDict()
    raw_events = dc_types.ROIEventSet()
    invalid_raw_events = dc_types.ROIEventSet()
    unmixed_events = dc_types.ROIEventSet()
    invalid_unmixed_events = dc_types.ROIEventSet()
    roi_flags = {}
    roi_flags['decrosstalk_ghost'] = []

    true_flags = []

    iterator = itertools.product([True, False],
                                 [True, False],
                                 [True, False],
                                 [True, False],
                                 [True, False],
                                 [True, False])

    roi_id = -1
    for _f in iterator:
        roi_id += 1
        flags = {'valid_raw_trace': _f[0],
                 'valid_raw_active_trace': _f[1],
                 'valid_unmixed_trace': _f[2],
                 'valid_unmixed_active_trace': _f[3],
                 'converged': _f[4],
                 'ghost': _f[5]}

        if not flags['valid_raw_trace']:
            flags['valid_raw_active_trace'] = False
            flags['valid_unmixed_trace'] = False
            flags['valid_unmixed_active_trace'] = False
        if not flags['valid_raw_active_trace']:
            flags['valid_unmixed_trace'] = False
            flags['valid_unmixed_active_trace'] = False
        if not flags['valid_unmixed_trace']:
            flags['valid_unmixed_active_trace'] = False

        true_flags.append(flags)

        # raw traces
        raw_roi = dc_types.ROIChannels()
        raw_roi['signal'] = rng.random_sample(n_t)
        raw_roi['crosstalk'] = rng.random_sample(n_t)

        raw_np = dc_types.ROIChannels()
        raw_np['signal'] = rng.random_sample(n_t)
        raw_np['crosstalk'] = rng.random_sample(n_t)

        if flags['valid_raw_trace'] and flags['valid_raw_active_trace']:
            raw_traces['roi'][roi_id] = raw_roi
            raw_traces['neuropil'][roi_id] = raw_np
        else:
            invalid_raw_traces['roi'][roi_id] = raw_roi
            invalid_raw_traces['neuropil'][roi_id] = raw_np
            if not flags['valid_raw_trace']:
                continue

        # raw trace events
        ee = dc_types.ROIEventChannels()
        e = dc_types.ROIEvents()
        e['events'] = rng.choice(np.arange(n_t, dtype=int), 3)
        e['trace'] = rng.random_sample(3)
        ee['signal'] = e
        e = dc_types.ROIEvents()
        e['events'] = rng.choice(np.arange(n_t, dtype=int), 3)
        e['trace'] = rng.random_sample(3)
        ee['crosstalk'] = e

        if flags['valid_raw_active_trace']:
            raw_events[roi_id] = ee
        else:
            invalid_raw_events[roi_id] = ee
            continue

        # unmixed traces
        unmixed_roi = dc_types.ROIChannels()
        unmixed_roi['signal'] = rng.random_sample(n_t)
        unmixed_roi['crosstalk'] = rng.random_sample(n_t)
        unmixed_roi['mixing_matrix'] = rng.random_sample((2, 2))
        unmixed_roi['use_avg_mixing_matrix'] = not flags['converged']
        if not flags['converged']:
            unmixed_roi['poorly_converged_signal'] = rng.random_sample(n_t)
            unmixed_roi['poorly_converged_crosstalk'] = rng.random_sample(n_t)
            mm = rng.random_sample((2, 2))
            unmixed_roi['poorly_converged_mixing_matrix'] = mm

        unmixed_np = dc_types.ROIChannels()
        unmixed_np['signal'] = rng.random_sample(n_t)
        unmixed_np['crosstalk'] = rng.random_sample(n_t)
        unmixed_np['mixing_matrix'] = rng.random_sample((2, 2))
        unmixed_np['use_avg_mixing_matrix'] = not flags['converged']
        if not flags['converged']:
            unmixed_np['poorly_converged_signal'] = rng.random_sample(n_t)
            unmixed_np['poorly_converged_crosstalk'] = rng.random_sample(n_t)
            mm = rng.random_sample((2, 2))
            unmixed_np['poorly_converged_mixing_matrix'] = mm

        is_valid = (flags['valid_unmixed_trace'] and
                    flags['valid_unmixed_active_trace'])

        if is_valid:
            unmixed_traces['roi'][roi_id] = unmixed_roi
            unmixed_traces['neuropil'][roi_id] = unmixed_np
        else:
            invalid_unmixed_traces['roi'][roi_id] = unmixed_roi
            invalid_unmixed_traces['neuropil'][roi_id] = unmixed_np
            if not flags['valid_unmixed_trace']:
                continue

        # unmixedtrace events
        ee = dc_types.ROIEventChannels()
        e = dc_types.ROIEvents()
        e['events'] = rng.choice(np.arange(n_t, dtype=int), 3)
        e['trace'] = rng.random_sample(3)
        ee['signal'] = e
        e = dc_types.ROIEvents()
        e['events'] = rng.choice(np.arange(n_t, dtype=int), 3)
        e['trace'] = rng.random_sample(3)
        ee['crosstalk'] = e

        if flags['valid_unmixed_active_trace']:
            unmixed_events[roi_id] = ee
        else:
            invalid_unmixed_events[roi_id] = ee
            continue

        if flags['ghost']:
            roi_flags['decrosstalk_ghost'].append(roi_id)

    output = {}
    output['roi_flags'] = roi_flags
    output['raw_traces'] = raw_traces
    output['invalid_raw_traces'] = invalid_raw_traces
    output['unmixed_traces'] = unmixed_traces
    output['invalid_unmixed_traces'] = invalid_unmixed_traces
    output['raw_events'] = raw_events
    output['invalid_raw_events'] = invalid_raw_events
    output['unmixed_events'] = unmixed_events
    output['invalid_unmixed_events'] = invalid_unmixed_events
    output['true_flags'] = true_flags

    return output


def test_qc_output(tmpdir):
    """
    test that write_qc_data produces the expected HDF5 file
    """

    out_path = pathlib.Path(tmpdir) / 'dummy_qc_data.h5'

    truth = create_dataset()

    write_qc_data(out_path, 1122, truth['roi_flags'],
                  truth['raw_traces'], truth['invalid_raw_traces'],
                  truth['unmixed_traces'], truth['invalid_unmixed_traces'],
                  truth['raw_events'], truth['invalid_raw_events'],
                  truth['unmixed_events'], truth['invalid_unmixed_events'])

    eps = 1.0e-10
    ct_raw_trace = 0
    ct_raw_events = 0
    ct_unmixed_trace = 0
    ct_converged = 0
    ct_not_converged = 0
    ct_unmixed_events = 0
    ct_ghost = 0
    with h5py.File(out_path, 'r') as in_file:
        assert in_file['paired_plane'][()] == 1122
        all_roi_id = set(range(len(truth['true_flags'])))
        written_roi_id = set([int(ii) for ii in in_file['ROI'].keys()])
        assert all_roi_id == written_roi_id

        for roi_id, flags in enumerate(truth['true_flags']):
            for (f0, f1) in [('valid_raw_trace',
                              f'ROI/{roi_id}/valid_raw_trace'),
                             ('valid_raw_active_trace',
                              f'ROI/{roi_id}/valid_raw_active_trace'),
                             ('valid_unmixed_trace',
                              f'ROI/{roi_id}/valid_unmixed_trace'),
                             ('valid_unmixed_active_trace',
                              f'ROI/{roi_id}/valid_unmixed_active_trace')]:

                if flags[f0] ^ in_file[f1][()]:
                    msg = f"For ROI {roi_id}\n"
                    msg += f"{f0}: {flags[f0]}\n"
                    msg += f"{f1}: {in_file[f1][()]}"
                    raise RuntimeError(msg)

            raw_trace_key = f'ROI/{roi_id}/roi/raw/signal/trace'
            if flags['valid_raw_trace'] or raw_trace_key in in_file:
                ct_raw_trace += 1
                if roi_id in truth['raw_traces']['roi']:
                    truth_src = truth['raw_traces']
                else:
                    truth_src = truth['invalid_raw_traces']

                for ff in ('roi', 'neuropil'):
                    for sig in ('signal', 'crosstalk'):
                        trace_key = f'ROI/{roi_id}/{ff}/raw/{sig}/trace'
                        np.testing.assert_allclose(truth_src[ff][roi_id][sig],
                                                   in_file[trace_key][()],
                                                   atol=eps, rtol=eps)

            raw_events_key = f'ROI/{roi_id}/roi/raw/signal/events'
            if flags['valid_raw_active_trace'] or raw_events_key in in_file:
                ct_raw_events += 1
                if roi_id in truth['raw_events']:
                    truth_src = truth['raw_events'][roi_id]
                else:
                    truth_src = truth['invalid_raw_events'][roi_id]
                for sig in ('signal', 'crosstalk'):
                    e_key = f'ROI/{roi_id}/roi/raw/{sig}/events'
                    np.testing.assert_array_equal(truth_src[sig]['events'],
                                                  in_file[e_key][()])

            converged_key = f'ROI/{roi_id}/roi/unmixed/converged'
            if converged_key in in_file:
                assert not (flags['converged'] ^ in_file[converged_key][()])

            unmixed_trace_key = f'ROI/{roi_id}/roi/unmixed/signal/trace'
            if flags['valid_unmixed_trace'] or unmixed_trace_key in in_file:
                ct_unmixed_trace += 1
                if roi_id in truth['unmixed_traces']['roi']:
                    truth_src = truth['unmixed_traces']
                else:
                    truth_src = truth['invalid_unmixed_traces']

                for ff in ('roi', 'neuropil'):
                    for sig in ('signal', 'crosstalk'):
                        trace_key = f'ROI/{roi_id}/{ff}/unmixed/{sig}/trace'

                        np.testing.assert_allclose(truth_src[ff][roi_id][sig],
                                                   in_file[trace_key][()],
                                                   atol=eps, rtol=eps)

                roi_head = f'ROI/{roi_id}/roi/unmixed'
                roi_mm_key = f'{roi_head}/poorly_converged_mixing_matrix'
                roi_sig_key = f'{roi_head}/poorly_converged_signal'
                roi_ct_key = f'{roi_head}/poorly_converged_crosstalk'

                neuropil_head = f'ROI/{roi_id}/neuropil/unmixed'
                np_mm_key = f'{neuropil_head}/poorly_converged_mixing_matrix'
                np_sig_key = f'{neuropil_head}/poorly_converged_signal'
                np_ct_key = f'{neuropil_head}/poorly_converged_crosstalk'

                if flags['converged']:
                    ct_converged += 1
                    for k in (roi_mm_key, roi_sig_key, roi_ct_key,
                              np_mm_key, np_sig_key, np_ct_key):
                        assert k not in in_file
                else:
                    ct_not_converged += 1
                    p_key = 'poorly_converged_mixing_matrix'
                    true_val = truth_src['roi'][roi_id][p_key]
                    np.testing.assert_allclose(true_val,
                                               in_file[roi_mm_key],
                                               atol=eps,
                                               rtol=eps)

                    p_key = 'poorly_converged_signal'
                    true_val = truth_src['roi'][roi_id][p_key]
                    np.testing.assert_allclose(true_val,
                                               in_file[roi_sig_key],
                                               atol=eps,
                                               rtol=eps)

                    p_key = 'poorly_converged_crosstalk'
                    true_val = truth_src['roi'][roi_id][p_key]
                    np.testing.assert_allclose(true_val,
                                               in_file[roi_ct_key],
                                               atol=eps,
                                               rtol=eps)

                    p_key = 'poorly_converged_mixing_matrix'
                    true_val = truth_src['neuropil'][roi_id][p_key]
                    np.testing.assert_allclose(true_val,
                                               in_file[np_mm_key],
                                               atol=eps,
                                               rtol=eps)

                    p_key = 'poorly_converged_signal'
                    true_val = truth_src['neuropil'][roi_id][p_key]
                    np.testing.assert_allclose(true_val,
                                               in_file[np_sig_key],
                                               atol=eps,
                                               rtol=eps)

                    p_key = 'poorly_converged_crosstalk'
                    true_val = truth_src['neuropil'][roi_id][p_key]
                    np.testing.assert_allclose(true_val,
                                               in_file[np_ct_key],
                                               atol=eps,
                                               rtol=eps)

            u_header = f'ROI/{roi_id}/roi/unmixed'
            unmixed_events_key = f'{u_header}/signal/events'
            present = (unmixed_events_key in in_file)
            if flags['valid_unmixed_active_trace'] or present:

                ct_unmixed_events += 1
                if roi_id in truth['unmixed_events']:
                    truth_src = truth['unmixed_events'][roi_id]
                else:
                    truth_src = truth['invalid_unmixed_events'][roi_id]

                for sig in ('signal', 'crosstalk'):
                    events_key = f'{u_header}/{sig}/events'
                    np.testing.assert_array_equal(truth_src[sig]['events'],
                                                  in_file[events_key][()])

            ghost_key = f'ROI/{roi_id}/is_ghost'
            if in_file[ghost_key][()]:
                assert roi_id in truth['roi_flags']['decrosstalk_ghost']
                ct_ghost += 1

    assert ct_raw_trace > 0
    assert ct_raw_events > 0
    assert ct_unmixed_trace > 0
    assert ct_converged > 0
    assert ct_not_converged > 0
    assert ct_unmixed_events > 0
    assert ct_ghost > 0
    assert ct_ghost == len(truth['roi_flags']['decrosstalk_ghost'])

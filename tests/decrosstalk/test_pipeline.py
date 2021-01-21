import os
from ophys_etl.transforms.decrosstalk_wrapper import DecrosstalkWrapper

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

    expected_files.append('0_1_invalid_flags.json')
    expected_files.append('1_0_invalid_flags.json')
    expected_files.append(output_json)

    for pair in session['coupled_planes']:
        for plane in pair['planes']:
            roi_name = plane['output_roi_trace_file']
            np_name = plane['output_neuropil_trace_file']
            assert roi_name != np_name
            assert roi_name not in expected_files
            assert np_name not in expected_files
            expected_files.append(roi_name)
            expected_files.append(np_name)

    output_dir = session['qc_output_dir']
    for fname in expected_files:
        full_name = os.path.join(output_dir, fname)
        msg = 'could not find %s' % full_name
        assert os.path.isfile(full_name), msg

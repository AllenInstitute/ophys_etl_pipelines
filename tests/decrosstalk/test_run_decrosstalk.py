import os
from ophys_etl.decrosstalk.ophys_plane import OphysPlane

from .utils import create_data


def test_full_pipeline(tmpdir):
    """
    Test that OphysPlane.run_decrosstalk() can run
    as expected in production.

    Only verifies that expected outputs are created.
    Does not validate contents of those files.
    """

    session = create_data(tmpdir)

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

    expected_files.append('0_1_invalid_flags.json')
    expected_files.append('1_0_invalid_flags.json')

    output_dir = session['qc_output_dir']
    for fname in expected_files:
        full_name = os.path.join(output_dir, fname)
        msg = 'could not find %s' % full_name
        assert os.path.isfile(full_name), msg

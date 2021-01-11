import os
import numpy as np
import tempfile
import h5py
import ophys_etl.decrosstalk.io_utils as io_utils

def test_write_to_h5():

    data = {}
    data['first_dir'] = 2
    data['second_dir'] = {}
    data['second_dir']['sub1'] = False
    data['second_dir']['sub2'] = np.array([9,4,5,1])
    data['second_dir']['sub3'] = {}
    data['second_dir']['sub3']['a'] = 4.5
    data['second_dir']['sub3']['b'] = np.array([1.2, 4.3])
    data['second_dir']['sub3']['c'] = True

    this_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(this_dir, 'tmp')
    assert os.path.isdir(tmp_dir)
    tmp_fname = tempfile.mkstemp(prefix='write_h5_test_',
                                 suffix='.h5',
                                 dir=tmp_dir)[1]

    # because mkstemp opens a connection to the file
    if os.path.exists(tmp_fname):
        os.unlink(tmp_fname)

    io_utils.write_to_h5(tmp_fname, data)
    try:
        with h5py.File(tmp_fname, mode='r') as in_file:
            assert in_file['first_dir'][()] == 2
            assert in_file['second_dir/sub1'][()] == False
            assert isinstance(in_file['second_dir/sub1'][()], np.bool_)
            np.testing.assert_array_equal(in_file['second_dir/sub2'][()],
                                          np.array([9,4,5,1]))
            assert in_file['second_dir/sub3/a'][()] == 4.5
            assert in_file['second_dir/sub3/c'][()] == True
            assert isinstance(in_file['second_dir/sub3/c'][()], np.bool_)
            np.testing.assert_array_equal(in_file['second_dir/sub3/b'][()],
                                          np.array([1.2,4.3]))
    
    except:
        raise
    finally:
        if os.path.exists(tmp_fname):
            os.unlink(tmp_fname)

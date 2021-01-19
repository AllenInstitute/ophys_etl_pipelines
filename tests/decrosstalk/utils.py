import os


def teardown_function(function):
    """
    Parameters
    ----------
    function -- the function this is being called after
    (teardown_function is a part of pytest's infrastructure)

    Returns
    -------
    Nothing

    Expects function to have a member _temp_files that is a list of
    paths to temporary files created by the test function. This
    method loops over those paths, using os.unlink to remove them.

    Does nothing if function does not have member _temp_files.
    """
    if not hasattr(function, '_temp_files'):
        return
    if isinstance(function._temp_files, list):
        for temp_file in function._temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def get_tmp_dir():
    """
    Return absolute path to tests/decrosstalk/tmp
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(this_dir, 'tmp')
    assert os.path.isdir(tmp_dir)
    return tmp_dir


def get_data_dir():
    """
    Return absolute path to tests/decrosstalk/data
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    assert os.path.isdir(data_dir)
    return data_dir

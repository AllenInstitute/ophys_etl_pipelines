import os


def teardown_function(function):
    if not hasattr(function, '_temp_files'):
        return
    if isinstance(function._temp_files, list):
        for temp_file in function._temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def get_tmp_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(this_dir, 'tmp')
    assert os.path.isdir(tmp_dir)
    return tmp_dir


def get_data_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    assert os.path.isdir(data_dir)
    return data_dir

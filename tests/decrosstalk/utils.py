import os


def teardown_function(function):
    if not hasattr(function, '_temp_files'):
        return
    if isinstance(function._temp_files, list):
        for temp_file in function._temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    elif isinstance(function._temp_files, dict):
        for temp_dir in function._temp_files:
            for fname in function._temp_files[temp_dir]:
                if os.path.exists(os.path.join(temp_dir, fname)):
                    os.unlink(os.path.join(temp_dir, fname))
            for sub_dir in os.listdir(temp_dir):
                _dir = os.path.join(temp_dir, sub_dir)
                if os.path.isdir(_dir):
                    os.removedirs(_dir)
            if os.path.isdir(temp_dir):
                os.removedirs(temp_dir)


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

import os


def get_data_dir():
    """
    Return absolute path to tests/decrosstalk/data
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    assert os.path.isdir(data_dir)
    return data_dir

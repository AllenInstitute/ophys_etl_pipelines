import pytest
import warnings
import json
import pathlib

from ophys_etl.utils.tempfile_util import (
    mkstemp_clean)

from ophys_etl.modules.mesoscope_splitting.__main__ import (
    get_full_field_path)


class DummyLogger(object):

    def warning(self, msg):
        warnings.warn(msg)

    def info(self, msg):
        print(msg)


def test_missing_platform_file():
    """
    Test that warning is logged when platform_json_path is
    missing from args
    """
    with pytest.warns(UserWarning,
                      match="platform_json_path not specified"):
        val = get_full_field_path(runner_args={}, logger=DummyLogger())
    assert val is None


def test_missing_ff_key():
    """
    Test that warning is logged when fullfield_2p_image is missing
    from platform_json
    """
    json_path = mkstemp_clean(suffix='.json')
    with open(json_path, 'w') as out_file:
        out_file.write(json.dumps({'nonsense': 2}))
    these_args = {'platform_json_path': str(json_path)}
    with pytest.warns(UserWarning,
                      match="fullfield_2p_image not present"):
        val = get_full_field_path(
                runner_args=these_args,
                logger=DummyLogger())
    assert val is None
    json_path = pathlib.Path(json_path)
    json_path.unlink()


@pytest.mark.parametrize("upload", (True, False, None))
def test_missing_file(upload):
    """
    Test that warning is logged when fullfield_2p_image is not
    actually a file

    if upload is False, do not list an upload directory

    if upload is None, set upload directory to None
    """
    json_path = mkstemp_clean(suffix='.json')
    with open(json_path, 'w') as out_file:
        out_file.write(
            json.dumps({'fullfield_2p_image': 'silly.txt'}))

    these_args = {'storage_directory': 'nonsense',
                  'data_upload_dir': 'more_nonsense',
                  'platform_json_path': json_path}

    if upload is None:
        these_args['data_upload_dir'] = None
    elif not upload:
        these_args.pop('data_upload_dir')

    with pytest.warns(UserWarning,
                      match="full field image file does not exist"):
        val = get_full_field_path(
                runner_args=these_args,
                logger=DummyLogger())

    assert val is None
    json_path = pathlib.Path(json_path)
    json_path.unlink()


@pytest.mark.parametrize(
        "where_tiff, specify_upload",
        [("storage", True),
         ("storage", False),
         ("storage", None),
         ("upload", True)])
def test_file_exists(
        tmp_path_factory,
        where_tiff,
        specify_upload,
        helper_functions):
    """
    Test that the correct file path is returned when
    fullfield_2p_image exists and is well specified

    where_tiff controls which directory (storage or data_upload)
    contains the fullfield_2p_image

    specify_upload controls whether or not data_upload_dir is actually
    in runner_args
    """
    storage_dir = tmp_path_factory.mktemp('storage_')
    upload_dir = tmp_path_factory.mktemp('upload_')

    if where_tiff == 'storage':
        ff_path = mkstemp_clean(dir=storage_dir, suffix='.tiff')
    elif where_tiff == 'upload':
        ff_path = mkstemp_clean(dir=upload_dir, suffix='.tiff')

    with open(ff_path, 'w') as out_file:
        out_file.write('ta dah!')

    platform_json_path = mkstemp_clean(
            dir=upload_dir,
            suffix='.json')

    with open(platform_json_path, 'w') as out_file:
        out_file.write(
            json.dumps(
                {'fullfield_2p_image': pathlib.Path(ff_path).name}))

    these_args = {'storage_directory': str(storage_dir),
                  'data_upload_dir': str(upload_dir),
                  'platform_json_path': platform_json_path}

    if specify_upload is None:
        these_args['data_upload_dir'] = None
    elif not specify_upload:
        these_args.pop('data_upload_dir')

    val = get_full_field_path(
            runner_args=these_args,
            logger=DummyLogger())

    assert val is not None
    assert str(val.resolve().absolute()) == ff_path

    helper_functions.clean_up_dir(storage_dir)
    helper_functions.clean_up_dir(upload_dir)

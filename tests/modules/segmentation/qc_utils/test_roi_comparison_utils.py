import pytest
import pathlib

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    _validate_paths_v_names,
    create_roi_v_background_grid)


@pytest.mark.parametrize(
        'path_input, name_input, path_expected, name_expected, is_valid',
        [  # mismatch
         (pathlib.Path('a/path.txt'), ['a', 'b'], None, None, False),

         # mismatch
         ([pathlib.Path('a/path.txt'), pathlib.Path('b/path.txt')],
          'a', None, None, False),

         # two singletons
         (pathlib.Path('a/path.txt'),
          'a',
          [pathlib.Path('a/path.txt')],
          ['a'],
          True),

         # two lists
         ([pathlib.Path('a/path.txt'), pathlib.Path('b/path.txt')],
          ['a', 'b'],
          [pathlib.Path('a/path.txt'), pathlib.Path('b/path.txt')],
          ['a', 'b'],
          True),

         # singleton and list
         ([pathlib.Path('a/path.txt')],
          'a',
          [pathlib.Path('a/path.txt')],
          ['a'],
          True),

         # reverse singleton and list
         (pathlib.Path('a/path.txt'),
          ['a'],
          [pathlib.Path('a/path.txt')],
          ['a'],
          True),
        ])
def test_validate_paths_v_names(path_input, name_input,
                                path_expected, name_expected,
                                is_valid):

    if not is_valid:
        with pytest.raises(RuntimeError, match='These must be the same shape'):
            _validate_paths_v_names(path_input,
                                    name_input)

    else:
        (path_output,
         name_output) = _validate_paths_v_names(path_input,
                                                name_input)

        assert path_expected == path_output
        assert name_expected == name_output


def test_create_roi_v_background(tmpdir, background_png,
                                 background_pkl, roi_file):
    """
    This is just going to be a smoke test
    """

    # many backgrounds; many ROIs
    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            attribute_name='dummy_value')

    # one background; many ROIs
    create_roi_v_background_grid(
            background_png,
            'png',
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            attribute_name='dummy_value')

    # one background; one ROI
    create_roi_v_background_grid(
            background_png,
            'png',
            roi_file,
            'a',
            attribute_name='dummy_value')

    # many backgrounds; one ROIs
    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            roi_file,
            'a',
            attribute_name='dummy_value')

    # different combinations of singleton/1-element list inputs
    create_roi_v_background_grid(
            background_png,
            ['png'],
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            attribute_name='dummy_value')

    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            roi_file,
            ['a'],
            attribute_name='dummy_value')

    create_roi_v_background_grid(
            [background_png],
            'png',
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            attribute_name='dummy_value')

    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            [roi_file],
            'a',
            attribute_name='dummy_value')

    # test that error is raised when an unknown background file type
    # is passed in
    with pytest.raises(RuntimeError, match='must be either .png or .pkl'):
        create_roi_v_background_grid(
                [background_png, pathlib.Path('dummy.jpg')],
                ['png', 'junk'],
                [roi_file],
                ['a'],
                attribute_name='dummy_value')

    # test that errors are raised when paths and shapes are of
    # mismatched sizes
    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png'],
                [roi_file, roi_file, roi_file],
                ['a', 'b', 'c'],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                'png',
                [roi_file, roi_file, roi_file],
                ['a', 'b', 'c'],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png],
                ['png', 'pkl'],
                [roi_file, roi_file, roi_file],
                ['a', 'b', 'c'],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png', 'pkl'],
                [roi_file, roi_file, roi_file],
                ['a', 'b'],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png', 'pkl'],
                [roi_file, roi_file],
                ['a', 'b', 'c'],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png', 'pkl'],
                roi_file,
                ['a', 'b', 'c'],
                attribute_name='dummy_value')

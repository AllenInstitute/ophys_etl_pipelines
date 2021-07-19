import pytest
import pathlib
import numpy as np
import json
import PIL.Image
import networkx as nx
from itertools import combinations, product

from ophys_etl.types import ExtractROI

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI)

from ophys_etl.modules.segmentation.merge.roi_utils import (
    ophys_roi_to_extract_roi)

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    roi_list_from_file,
    _validate_paths_v_names,
    create_roi_v_background_grid)


@pytest.fixture(scope='session')
def background_png(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('background_png'))
    rng = np.random.default_rng(887123)
    image_array = rng.integers(0, 255, size=(50, 50)).astype(np.uint8)
    image = PIL.Image.fromarray(image_array)
    file_path = tmpdir/'background.png'
    image.save(file_path)
    yield file_path


@pytest.fixture(scope='session')
def background_pkl(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('background_pkl'))
    graph = nx.Graph()
    rng = np.random.default_rng(543221)
    coords = np.arange(0, 50)
    for xx, yy in combinations(coords, 2):
        minx = max(0, xx-1)
        miny = max(0, yy-1)
        maxx = min(xx+2, 50)
        maxy = min(yy+2, 50)
        xx_other = np.arange(minx, maxx)
        yy_other = np.arange(miny, maxy)
        for x1, y1 in product(xx_other, yy_other):
            graph.add_edge((xx, yy), (x1, y1), dummy_value=rng.random())

    file_path = tmpdir/'background_graph.pkl'
    nx.write_gpickle(graph, file_path)
    yield file_path


@pytest.fixture(scope='session')
def list_of_roi():
    """
    A list of ExtractROIs
    """
    output = []
    rng = np.random.default_rng(11231)
    for ii in range(10):
        x0 = int(rng.integers(0, 30))
        y0 = int(rng.integers(0, 30))
        width = int(rng.integers(4, 10))
        height = int(rng.integers(4, 10))
        mask = rng.integers(0, 2, size=(height,width)).astype(bool)

        # because np.ints are not JSON serializable
        real_mask = []
        for row in mask:
            this_row = []
            for el in row:
                if el:
                    this_row.append(True)
                else:
                    this_row.append(False)
            real_mask.append(this_row)

        roi = ExtractROI(x=x0, width=width,
                         y=y0, height=height,
                         valid_roi=True,
                         mask=real_mask,
                         id=ii)
        output.append(roi)
    return output


@pytest.fixture(scope='session')
def roi_file(tmpdir_factory, list_of_roi):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('roi_reading'))
    file_path = tmpdir/'list_of_rois.json'
    with open(file_path, 'w') as out_file:
        json.dump(list_of_roi, out_file)
    yield file_path


def test_roi_list_from_file(roi_file, list_of_roi):
    raw_actual = roi_list_from_file(roi_file)
    actual = [ophys_roi_to_extract_roi(roi)
              for roi in raw_actual]
    assert actual == list_of_roi



@pytest.mark.parametrize(
        'path_input, name_input, path_expected, name_expected, is_valid',
        [# mismatch
         (pathlib.Path('a/path.txt'), ['a', 'b'], None, None, False),

         # mismatch
         ([pathlib.Path('a/path.txt'), pathlib.Path('b/path.txt')],
          'a', None, None, False),

         # two singletons
         (pathlib.Path('a/path.txt'), 'a', [pathlib.Path('a/path.txt')], ['a'], True),

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



def test_create_roi_v_background(tmpdir, background_png, background_pkl, roi_file):
    """
    This is just going to be a smoke test
    """

    # many backgrounds; many ROIs
    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')

    # one background; many ROIs
    create_roi_v_background_grid(
            background_png,
            'png',
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')

    # one background; one ROI
    create_roi_v_background_grid(
            background_png,
            'png',
            roi_file,
            'a',
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')


    # many backgrounds; one ROIs
    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            roi_file,
            'a',
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')

    # different combinations of singleton/1-element list inputs
    create_roi_v_background_grid(
            background_png,
            ['png'],
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')

    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            roi_file,
            ['a'],
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')

    create_roi_v_background_grid(
            [background_png],
            'png',
            [roi_file, roi_file, roi_file],
            ['a', 'b', 'c'],
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')

    create_roi_v_background_grid(
            [background_png, background_pkl],
            ['png', 'pkl'],
            [roi_file],
            'a',
            [(255, 0, 0), (0, 255, 0)],
            attribute_name='dummy_value')

    # test that error is raised when an unknown background file type
    # is passed in
    with pytest.raises(RuntimeError, match='must be either .png or .pkl'):
        create_roi_v_background_grid(
                [background_png, pathlib.Path('dummy.jpg')],
                ['png', 'junk'],
                [roi_file],
                ['a'],
                [(255, 0, 0), (0, 255, 0)],
                attribute_name='dummy_value')

    # test that errors are raised when paths and shapes are of
    # mismatched sizes
    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png'],
                [roi_file, roi_file, roi_file],
                ['a', 'b', 'c'],
                [(255, 0, 0), (0, 255, 0)],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                'png',
                [roi_file, roi_file, roi_file],
                ['a', 'b', 'c'],
                [(255, 0, 0), (0, 255, 0)],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png],
                ['png', 'pkl'],
                [roi_file, roi_file, roi_file],
                ['a', 'b', 'c'],
                [(255, 0, 0), (0, 255, 0)],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png', 'pkl'],
                [roi_file, roi_file, roi_file],
                ['a', 'b'],
                [(255, 0, 0), (0, 255, 0)],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png', 'pkl'],
                [roi_file, roi_file],
                ['a', 'b', 'c'],
                [(255, 0, 0), (0, 255, 0)],
                attribute_name='dummy_value')

    with pytest.raises(RuntimeError, match='These must be the same shape'):
        create_roi_v_background_grid(
                [background_png, background_pkl],
                ['png', 'pkl'],
                roi_file,
                ['a', 'b', 'c'],
                [(255, 0, 0), (0, 255, 0)],
                attribute_name='dummy_value')

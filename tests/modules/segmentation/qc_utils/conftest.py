import pytest
import pathlib
import numpy as np
import json
import PIL.Image
import networkx as nx
from itertools import combinations, product

from ophys_etl.types import ExtractROI


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
        mask = rng.integers(0, 2, size=(height, width)).astype(bool)

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

        if ii % 2 == 0:
            valid_roi = True
        else:
            valid_roi = False
        roi = ExtractROI(x=x0, width=width,
                         y=y0, height=height,
                         valid_roi=valid_roi,
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

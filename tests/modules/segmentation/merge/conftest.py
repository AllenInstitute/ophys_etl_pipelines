import pytest
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


@pytest.fixture
def example_roi_list():
    rng = np.random.RandomState(6412439)
    roi_list = []
    for ii in range(30):
        x0 = rng.randint(0, 25)
        y0 = rng.randint(0, 25)
        height = rng.randint(3, 7)
        width = rng.randint(3, 7)
        mask = rng.randint(0, 2, (height, width)).astype(bool)
        roi = OphysROI(x0=x0, y0=y0,
                       height=height, width=width,
                       mask_matrix=mask,
                       roi_id=ii,
                       valid_roi=True)
        roi_list.append(roi)

    return roi_list


@pytest.fixture
def example_roi0():
    rng = np.random.RandomState(64322)
    roi = OphysROI(roi_id=4,
                   x0=10,
                   y0=22,
                   width=7,
                   height=11,
                   valid_roi=True,
                   mask_matrix=rng.randint(0, 2,
                                           (11, 7)).astype(bool))

    return roi


@pytest.fixture
def example_video():
    rng = np.random.RandomState(1172312)
    data = rng.random_sample((100, 50, 50))
    return data


@pytest.fixture
def timeseries_and_video_dataset():
    rng = np.random.RandomState(11818)
    video = {}
    timeseries = {}
    for ii in range(10):
        area = rng.randint(10, 20)
        video[ii] = rng.random_sample((100, area))
        timeseries[ii] = {'timeseries': rng.random_sample(100)}
    return {'video': video,
            'timeseries': timeseries}


@pytest.fixture
def roi_and_video_dataset():
    """
    Create a video with a bunch of neighboring, correlated ROIs
    """
    rng = np.random.RandomState(1723133)
    nrows = 100
    ncols = 100
    ntime = 700
    video = rng.randint(0, 100, (ntime, nrows, ncols))

    roi_list = []
    roi_id = 0
    for ii in range(4):
        x0 = rng.randint(14, 61)
        y0 = rng.randint(14, 61)
        height = rng.randint(12, 18)
        width = rng.randint(12, 18)

        mask = np.zeros((height, width)).astype(bool)
        mask[1:-1, 1:-1] = True

        freq = rng.randint(50, 400)
        time_series = np.sin(np.arange(ntime).astype(float)/freq)
        time_series = np.round(time_series).astype(int)
        for ir in range(height):
            for ic in range(width):
                if mask[ir, ic]:
                    video[:, y0+ir, x0+ic] += time_series

        for r0 in range(0, height, height//2):
            for c0 in range(0, width, width//2):
                roi_id += 1
                this_mask = mask[r0:r0+height//2, c0:c0+width//2]
                if this_mask.sum() == 0:
                    continue
                roi = OphysROI(x0=x0+c0, y0=y0+r0,
                               height=this_mask.shape[0],
                               width=this_mask.shape[1],
                               mask_matrix=this_mask,
                               roi_id=roi_id,
                               valid_roi=True)
                roi_list.append(roi)

    return {'video': video, 'roi_list': roi_list}


@pytest.fixture
def timeseries_video_corr_dataset():

    ntime = 50

    rng = np.random.RandomState(712553)
    video_lookup = {}
    timeseries_lookup = {}
    self_corr_lookup = {}

    area = [1, 10, 2, 6, 5]

    for roi_id, area in enumerate(area):
        area = rng.randint(1, 20)
        video_lookup[roi_id] = rng.random_sample((ntime, area))
        timeseries_lookup[roi_id] = {'timeseries': rng.random_sample(ntime)}
        self_corr_lookup[roi_id] = (rng.random_sample(),
                                    rng.random_sample())

    return {'video': video_lookup,
            'timeseries': timeseries_lookup,
            'self_corr': self_corr_lookup}


@pytest.fixture
def small_rois():
    video_lookup = {}
    rng = np.random.RandomState(881223)
    for ii in range(5):
        area = rng.randint(9, 22)
        video_lookup[ii] = rng.random_sample((100, area))
    return video_lookup


@pytest.fixture
def large_rois():
    video_lookup = {}
    rng = np.random.RandomState(881223)
    for ii in range(5, 8, 1):
        area = rng.randint(35, 45)
        video_lookup[ii] = rng.random_sample((100, area))
    return video_lookup

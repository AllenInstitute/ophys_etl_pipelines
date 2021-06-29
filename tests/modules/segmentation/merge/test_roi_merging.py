import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.merge.roi_merging import (
    do_roi_merger)


@pytest.fixture
def whole_dataset():
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


def test_do_roi_merger(whole_dataset):
    """
    smoke test for do_roi_merger
    """
    img_data = np.mean(whole_dataset['video'], axis=0)
    assert img_data.shape == whole_dataset['video'].shape[1:]
    new_roi_list = do_roi_merger(whole_dataset['roi_list'],
                                 whole_dataset['video'],
                                 3,
                                 2.0,
                                 filter_fraction=0.2)

    # test that some mergers were performed
    assert len(new_roi_list) > 0
    assert len(new_roi_list) < len(whole_dataset['roi_list'])

    # check that pixels were conserved
    input_pixels = set()
    for roi in whole_dataset['roi_list']:
        mask = roi.mask_matrix
        for ir in range(roi.height):
            for ic in range(roi.width):
                if mask[ir, ic]:
                    pix = (roi.y0+ir, roi.x0+ic)
                    input_pixels.add(pix)

    output_pixels = set()
    for roi in new_roi_list:
        mask = roi.mask_matrix
        for ir in range(roi.height):
            for ic in range(roi.width):
                if mask[ir, ic]:
                    pix = (roi.y0+ir, roi.x0+ic)
                    output_pixels.add(pix)

    assert output_pixels == input_pixels

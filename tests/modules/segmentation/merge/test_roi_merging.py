import numpy as np
from ophys_etl.modules.segmentation.merge.roi_merging import (
    do_roi_merger)


def test_do_roi_merger(roi_and_video_dataset):
    """
    smoke test for do_roi_merger
    """
    img_data = np.mean(roi_and_video_dataset['video'], axis=0)
    assert img_data.shape == roi_and_video_dataset['video'].shape[1:]
    new_roi_list = do_roi_merger(roi_and_video_dataset['roi_list'],
                                 roi_and_video_dataset['video'],
                                 3,
                                 2.0,
                                 filter_fraction=0.2)

    # test that some mergers were performed
    assert len(new_roi_list) > 0
    assert len(new_roi_list) < len(roi_and_video_dataset['roi_list'])

    # check that pixels were conserved
    input_pixels = set()
    for roi in roi_and_video_dataset['roi_list']:
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

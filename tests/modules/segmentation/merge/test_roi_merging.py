import numpy as np
from ophys_etl.modules.segmentation.merge.roi_merging import (
    do_roi_merger,
    get_new_merger_candidates)


def test_get_new_merger_candidates():

    merger_to_metric = {(11, 9): 4.2,
                        (13, 7): 3.1}

    neighbor_lookup = dict()
    neighbor_lookup[11] = [9, 4, 5, 15]
    neighbor_lookup[7] = [13, 4, 6]

    expected = {(11, 4), (11, 5), (15, 11),
                (7, 4), (7, 6)}

    k_list = list(neighbor_lookup.keys())
    for k in k_list:
        for n in neighbor_lookup[k]:
            if n not in neighbor_lookup:
                neighbor_lookup[n] = []
            neighbor_lookup[n].append(k)

    new_candidates = set(get_new_merger_candidates(neighbor_lookup,
                                                   merger_to_metric))
    assert new_candidates == expected


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

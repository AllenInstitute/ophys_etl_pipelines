import pytest
import numpy as np
import pathlib

from ophys_etl.modules.segmentation.utils.roi_utils import (
    pixel_list_to_extract_roi,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.segmentation.merge.louvain_merging import (
    do_louvain_clustering_on_rois)


@pytest.fixture(scope='session')
def louvain_test_data_fixture():
    rng = np.random.default_rng(771232)

    ntime = 100
    nrows = 20
    ncols = 20
    video = rng.normal(2.0, 0.2, size=(ntime, nrows, ncols))

    rois_as_pixels = []
    rois_as_pixels.append([(11, 14), (12, 14), (13, 14), (10, 14), (9, 14)])
    rois_as_pixels.append([(7, 8), (7, 9), (7, 10), (7, 11), (8, 9), (6, 9)])
    rois_as_pixels.append([(10, 9), (10, 11), (10, 12), (10, 13),
                           (9, 13), (8, 13)])
    rois_as_pixels.append([(5, 5), (5, 6), (4, 4), (6, 5), (6, 6)])
    rois_as_pixels.append([(15, 16), (15, 17), (15, 18),
                           (14, 16), (14, 17), (14, 18)])

    rois_as_pixels.append([(0, 0), (1, 1), (1, 0), (0, 1), (0, 2)])

    tt = np.arange(ntime)
    for roi in rois_as_pixels[:2]:
        trace = np.zeros(ntime, dtype=float)
        trace[20:] += 2.0*np.exp(-0.5*(tt[20:]-20.0)**2)
        trace[60:] += 2.0*np.exp(-0.25*(tt[60:]-60.0)**2)
        for pixel in roi:
            video[:, pixel[0], pixel[1]] = trace

    for roi in rois_as_pixels[2:5]:
        trace = np.zeros(ntime, dtype=float)
        trace[10:] += 2.0*np.exp(-0.5*(tt[10:]-10.0)**2)
        trace[78:] += 2.0*np.exp(-0.25*(tt[78:]-78.0)**2)
        for pixel in roi:
            video[:, pixel[0], pixel[1]] = trace

    for roi in (rois_as_pixels[5],):
        trace = np.zeros(ntime, dtype=float)
        trace[44:] += 2.0*np.exp(-0.5*(tt[44:]-44.0)**2)
        trace[56:] += 2.0*np.exp(-0.25*(tt[56:]-56.0)**2)
        for pixel in roi:
            video[:, pixel[0], pixel[1]] = trace

    roi_list = [extract_roi_to_ophys_roi(
                    pixel_list_to_extract_roi(roi, roi_id))
                for roi_id, roi in enumerate(rois_as_pixels)]

    return {'roi_list': roi_list, 'video': video}


def test_do_louvain_clustering_on_rois(
        tmpdir,
        louvain_test_data_fixture):

    tmpdir_path = pathlib.Path(tmpdir)
    video = louvain_test_data_fixture['video']
    input_roi_list = louvain_test_data_fixture['roi_list']

    input_pixel_set = set()
    input_roi_to_pixel = dict()
    for roi in input_roi_list:
        input_roi_to_pixel[roi.roi_id] = set()
        for pixel in roi.global_pixel_set:
            input_pixel_set.add(pixel)
            input_roi_to_pixel[roi.roi_id].add(pixel)

    (output_roi_list,
     merger_history) = do_louvain_clustering_on_rois(
                            input_roi_list,
                            video,
                            20,
                            0.2,
                            2,
                            tmpdir_path)

    assert len(output_roi_list) < len(input_roi_list)
    assert len(output_roi_list) == 3

    output_pixel_set = set()
    output_roi_to_pixel = dict()
    for roi in output_roi_list:
        output_roi_to_pixel[roi.roi_id] = set()
        for pixel in roi.global_pixel_set:
            output_pixel_set.add(pixel)
            output_roi_to_pixel[roi.roi_id].add(pixel)

    assert output_pixel_set == input_pixel_set

    for merger_pair in merger_history:
        dst = output_roi_to_pixel[merger_pair[0]]
        src = input_roi_to_pixel[merger_pair[1]]
        for pixel in src:
            assert pixel in dst

    # make sure that roi_id 5 did not participate in mergers
    for merger_pair in merger_history:
        if merger_pair[0] == 5:
            assert merger_pair[1] == 5
        if merger_pair[1] == 5:
            assert merger_pair[0] == 5

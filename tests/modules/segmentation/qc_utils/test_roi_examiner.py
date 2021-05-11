import matplotlib.pyplot as plt

import pytest
import h5py
import numpy as np
import pathlib

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.qc_utils.roi_utils import ROIExaminer


@pytest.fixture
def example_movie(tmpdir):
    """
    Create an example movie file; return its path
    """
    dir_path = pathlib.Path(tmpdir)
    file_path = dir_path / "example_movie.h5"
    rng = np.random.RandomState(12341)
    with h5py.File(file_path, "w") as out_file:
        out_file.create_dataset("data",
                                data=rng.randint(0, 10000,
                                                 size=(100, 200, 200)))

    return file_path


@pytest.fixture
def roi_groups():
    """
    List of lists of ROIs to compare
    """
    roi_list_0 = []
    mask = np.zeros((14, 19), dtype=bool)
    mask[5:11, :17] = True
    roi = OphysROI(roi_id=1,
                   x0=11,
                   y0=156,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_0.append(roi)

    mask = np.zeros((22, 8), dtype=bool)
    mask[2:20, 2:6] = True
    roi = OphysROI(roi_id=2,
                   x0=166,
                   y0=84,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_0.append(roi)

    mask = np.zeros((16, 33), dtype=bool)
    mask[8:14, :25] = True
    roi = OphysROI(roi_id=3,
                   x0=100,
                   y0=100,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_0.append(roi)

    roi_list_1 = []

    # duplicate to ensure there is an overlap
    mask = np.zeros((14, 19), dtype=bool)
    mask[5:11, :17] = True
    roi = OphysROI(roi_id=4,
                   x0=11,
                   y0=156,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_1.append(roi)

    mask = np.zeros((7, 9), dtype=bool)
    mask[1:5, 2:6] = True
    roi = OphysROI(roi_id=5,
                   x0=180,
                   y0=99,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_1.append(roi)

    mask = np.zeros((12, 6), dtype=bool)
    mask[3:10, :4] = True
    roi = OphysROI(roi_id=6,
                   x0=6,
                   y0=77,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_1.append(roi)

    roi_list_2 = []

    mask = np.zeros((8, 15), dtype=bool)
    mask[3:, :11] = True
    roi = OphysROI(roi_id=7,
                   x0=47,
                   y0=89,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_2.append(roi)

    mask = np.zeros((19, 5), dtype=bool)
    mask[2:, :3] = True
    roi = OphysROI(roi_id=8,
                   x0=166,
                   y0=13,
                   width=mask.shape[1],
                   height=mask.shape[0],
                   mask_matrix=mask,
                   valid_roi=True)

    roi_list_2.append(roi)

    return [roi_list_0,
            roi_list_1,
            roi_list_2]


def test_roi_examiner(example_movie, roi_groups):
    """
    Run a smoke test on the function in ROIExaminer
    """

    examiner = ROIExaminer(example_movie)
    examiner.load_rois_to_compare([((255, 0, 0), roi_groups[0]),
                                   ((0, 255, 0), roi_groups[1]),
                                   ((0, 0, 255), roi_groups[2])])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    examiner.plot_rois([0, 1], ax, labels=True)
    examiner.plot_rois([0, 1, 2], ax, labels=True)
    examiner.plot_distinct_rois([0, 1], ax, labels=True)
    examiner.plot_overlapping_rois([0, 1], ax, labels=True)

    examiner.plot_distinct_rois([0, 2], ax, labels=True)
    examiner.plot_overlapping_rois([0, 2], ax, labels=True)

    examiner.plot_thumbnail_and_trace(4)

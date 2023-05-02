from typing import List

import numpy as np

from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue


def generate_binary_mask(
    ophys_roi: OphysROI, ophys_roi_mask_values: List[OphysROIMaskValue]
) -> List[List[bool]]:
    """
    Generate binary mask for an ROI

    Parameters
    ----------
    ophys_roi
        The ophys ROI
    ophys_roi_mask_values
        The ophys ROI mask values

    Returns
    -------
    List[List[bool]]
        A list of lists of booleans representing the binary mask
    """
    mask = np.zeros((ophys_roi.height, ophys_roi.width), dtype=bool)
    for ophys_roi_mask_value in ophys_roi_mask_values:
        mask[
            ophys_roi_mask_value.row_index, ophys_roi_mask_value.col_index
        ] = True
    return mask.tolist()

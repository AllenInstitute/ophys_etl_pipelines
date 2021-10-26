import numpy as np
import PIL.Image
import pathlib
import json
from itertools import product
import re

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    extract_roi_to_ophys_roi,
    ophys_roi_to_extract_roi,
    convert_to_lims_roi)

from mask_reticle import mask_reticle, get_reticle

import argparse


def ophys_roi_from_png_path(png_path, roi_id):

    img = np.array(PIL.Image.open(png_path, 'r'))

    mask = (img==255)
    extract_roi = convert_to_lims_roi((0, 0),
                                      mask,
                                      roi_id=roi_id)

    return extract_roi_to_ophys_roi(extract_roi)


def construct_roi(outline_path,
                  mask_path,
                  roi_id=0):
    ophys_roi = ophys_roi_from_png_path(mask_path, roi_id)

    (reticle_rows,
     reticle_cols) = get_reticle()

    outline_img = np.array(PIL.Image.open(outline_path, 'r'))

    outline_img = mask_reticle(outline_img,
                               reticle_rows,
                               reticle_cols)

    outline_mask = (outline_img==0)

    bdry_mask = ophys_roi.boundary_mask
    bdry_valid = np.where(bdry_mask)
    bdry_row = bdry_valid[0]
    bdry_col = bdry_valid[1]

    roi_c_row = np.round(np.mean(bdry_row)).astype(int)
    roi_c_col = np.round(np.mean(bdry_col)).astype(int)

    outline_valid = np.where(outline_mask)
    outline_row = outline_valid[0]
    outline_col = outline_valid[1]
    outline_c_row = np.round(np.mean(outline_row)).astype(int)
    outline_c_col = np.round(np.mean(outline_col)).astype(int)

    row_guess = outline_c_row-roi_c_row
    col_guess = outline_c_col-roi_c_col

    displacements = [0]
    for ii in range(1,max(ophys_roi.width//2, ophys_roi.height//2),1):
        displacements.append(-1*ii)
        displacements.append(ii)

    err_mask = outline_mask.astype(int)

    min_err = None
    best_row = None
    best_col = None

    for _dr, _dc in product(displacements, displacements):
        row = row_guess+_dr
        col = col_guess+_dc
        candidate_row = bdry_row + row
        candidate_col = bdry_col + col

        dummy_mask = np.zeros(err_mask.shape, dtype=int)
        dummy_mask[candidate_row, candidate_col] = 1
        err = np.sum((dummy_mask-err_mask)**2)
        if min_err is None or err<min_err:
            min_err = err
            best_row = row
            best_col = col
        if min_err is not None:
            if min_err == 0:
                break

    output_roi = OphysROI(
                    roi_id=ophys_roi.roi_id,
                    x0=int(best_col),
                    y0=int(best_row),
                    height=ophys_roi.height,
                    width=ophys_roi.width,
                    mask_matrix=ophys_roi.mask_matrix,
                    valid_roi=True)
    return output_roi, min_err



if __name__ == "__main__":


    outline_dir = pathlib.Path('full_outlines')
    assert outline_dir.is_dir()
    mask_dir = pathlib.Path('binary_masks')
    assert mask_dir.is_dir()
    diagnostic_dir = pathlib.Path('spot_checks')
    assert diagnostic_dir.is_dir()

    roi_output = dict()

    roi_id_pattern = re.compile('[0-9]+')

    for outline_path in outline_dir.rglob('*'):

        outline_name = str(outline_path.name)
        if not outline_name.endswith('png'):
            continue
        roi_id_found = roi_id_pattern.findall(outline_name)
        assert len(roi_id_found) == 1
        roi_id = int(roi_id_found[0])
        assert roi_id not in roi_output
        mask_path = mask_dir/f'mask_{roi_id}.png'

        (roi,
         min_err) = construct_roi(outline_path,
                                  mask_path,
                                  roi_id=roi_id)

        roi_output[roi_id] = ophys_roi_to_extract_roi(roi)

        if min_err > 0:
            diagnostic_path = diagnostic_dir / f'diagnostic_{roi_id}.png'

            input_img = np.array(PIL.Image.open(outline_path, 'r'))
            rgb = 255*np.ones((input_img.shape[0], input_img.shape[1], 3),
                           dtype=np.uint8)

            for ic in range(3):
                rgb[:, :, ic] = input_img

            bdry_mask = roi.boundary_mask

            row0 = roi.y0
            col0 = roi.x0
            row1 = row0+roi.height
            col1 = col0+roi.width
            rgb[row0:row1, col0:col1, 0][bdry_mask] = 255
            for ic in (1, 2):
                rgb[row0:row1, col0:col1, ic][bdry_mask] = 0
            output_img = PIL.Image.fromarray(rgb)
            output_img.save(diagnostic_path)

    with open('rois_2020.json', 'w') as out_file:
        out_file.write(json.dumps(roi_output, indent=2))

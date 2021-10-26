import numpy as np
import PIL.Image
import pathlib


import argparse


def get_reticle():

    reticle_img = PIL.Image.open('scratch/template.png', 'r')
    reticle_img = np.array(reticle_img)
    reticle_pixels = np.where(reticle_img == 0)
    reticle_rows = reticle_pixels[0]
    reticle_cols = reticle_pixels[1]
    return reticle_rows, reticle_cols


def mask_reticle(input_img: np.ndarray,
                 reticle_rows: np.ndarray,
                 reticle_cols: np.ndarray):

    dark_pixels = np.where(input_img == 0)
    dark_rows = dark_pixels[0]
    dark_cols = dark_pixels[1]

    match_dr = None
    match_dc = None

    for row, col in zip(dark_rows, dark_cols):
        dr_candidates = row - reticle_rows
        dc_candidates = col - reticle_cols
        for dr, dc in zip(dr_candidates, dc_candidates):
            row_candidates = reticle_rows + dr
            col_candidates = reticle_cols + dc
            if row_candidates.min() < 0:
                continue
            if row_candidates.max() >= input_img.shape[0]:
                continue
            if col_candidates.min() < 0:
                continue
            if col_candidates.max() >= input_img.shape[1]:
                continue

            if (input_img[row_candidates, col_candidates]==0).all():
                match_dr = dr
                match_dc = dc
                break
        if match_dr is not None:
            break

    mask_rows = reticle_rows + match_dr
    mask_cols = reticle_cols + match_dc
    new_img = np.copy(input_img)
    new_img[mask_rows, mask_cols] = 255
    return new_img


def do_masking(input_path, output_path, clobber=False):
    input_path = pathlib.Path(input_path)
    if not input_path.is_file():
        raise RuntimeError(f'input file {str(input_path)} does not exist')
    output_path = pathlib.Path(output_path)
    if output_path.exists() and not clobber:
        raise RuntimeError(f'output file {str(output_path)} exists')

    (reticle_rows,
     reticle_cols) = get_reticle()

    input_img = np.array(PIL.Image.open(input_path, 'r'))

    new_img = mask_reticle(input_img,
                           reticle_rows,
                           reticle_cols)

    new_img = PIL.Image.fromarray(new_img)
    new_img.save(output_path)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    do_masking(args.input, args.output, clobber=args.clobber)

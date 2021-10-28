import matplotlib.figure as mplt_fig
import json
import PIL.Image
import copy
import re
from matplotlib.backends.backend_pdf import PdfPages

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_roi_boundary_to_img)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    extract_roi_to_ophys_roi)

import argparse
import multiprocessing
import pathlib


def plot_rois(extract_list, background_img, color_map, axis):
    for roi in extract_list:
        ophys = extract_roi_to_ophys_roi(roi)
        background_img = add_roi_boundary_to_img(
                    background_img,
                    ophys,
                    color_map[ophys.roi_id],
                    1.0)
    axis.imshow(background_img)
    return axis


def filter_rois(raw_roi_list,
                stat_name=None,
                min_stat=None,
                min_area=None):

    output = []
    for roi in raw_roi_list:
        scores = roi['clasifier_scores']
        if scores['area'] < min_area:
            continue
        stat = max(scores[f'corr_{stat_name}'],
                   scores[f'maximg_{stat_name}'],
                   scores[f'avgimg_{stat_name}'])
        if stat < min_stat:
            continue
        new_roi = copy.deepcopy(roi)
        if 'mask_matrix' in new_roi:
            new_roi['mask'] = new_roi['mask_matrix']
        if 'valid_roi' in new_roi:
            new_roi['valid'] = new_roi['valid_roi']
        if 'roi_id' in new_roi:
            new_roi['id'] = new_roi['roi_id']
        output.append(roi)
    return output


def generate_page(
        raw_roi_list=None,
        stat_name=None,
        min_stat=None,
        min_area=None,
        color_map=None,
        corr_img=None,
        max_img=None,
        experiment_id=None,
        fontsize=15):

    n_rows = 2
    ncols = 3

    raw_extract_list = []
    for roi in raw_roi_list:
        if 'mask_matrix' in roi:
            roi['mask'] = roi['mask_matrix']
        if 'valid_roi' in roi:
            roi['valid'] = roi['valid_roi']
        if 'roi_id' in roi:
            roi['id'] = roi['roi_id']
        raw_extract_list.append(roi)

    valid_roi_list = filter_rois(
                   raw_extract_list,
                   min_area=min_area,
                   stat_name=stat_name,
                   min_stat=min_stat)

    background_list = [corr_img]*n_cols
    background_list += [max_img]*n_cols

    title_list = [f'{experiment_id} correlation projection']
    title_list += [None, 'all ROIs',
                   f'area>{min_area}; {stat_name}>{min_stat:.2f}']
    title_list += [f'{experiment_id} max projection']
    title_list += [None]*(n_cols-1)

    roi_set_list = [None, raw_extract_list, valid_roi_list]*n_rows

    fig = mplt_fig.figure(figsize=(5*n_cols, 5*n_rows))
    axis_list = [fig.add_subplot(n_rows, n_cols, ii+1)
                 for ii in range(n_rows*n_cols)]

    for i_axis in range(axis_list):
        background = np.copy(background_list[i_axis])
        title = title_list[i_axis]
        roi_set = roi_set_list[i_axis]
        axis = axis_list[i_axis]

        if roi_set is not None:
            plot_rois(roi_set, background, color_map, axis)
        else:
            axis.imshow(background)

        if title is not None:
            axis.set_title(title, fontsize=fontsize)

    fig.tight_layout()
    return fig


def path_to_rgb(file_path):
    img = np.array(PIL.Image.open(file_path, 'r'))
    mn = img.min()
    img = img-mn
    mx = img.max()
    img = np.round(255.0*img.astype(float)/mx).astype(np.uint8)
    output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for ic in range(3):
        output[:, :, ic] = img
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_area', type=int, default=50)
    parser.add_argument('--stat_name', type=str, default='dmedian')
    parser.add_argument('--min_stat', type=float, default=0.25)
    parser.add_argument('--output_root', type=str, default=None)

    args = parser.parse_args()
    assert args.output_root is not None
    output_path = f'{output_root}_area_{args.min_area}'
    output_path += f'_{args.stat_name}_{args.min_stat:.2f}'
    output_path += '.pdf'

    info_dir = pathlib.Path('/allen/aibs/informatics')
    assert info_dir.is_dir()
    sfd_dir = info_dir / 'danielsf'
    assert sfd_dir.is_dir()
    s2p_dir = sfd_dir / 'suite2p_210921/v0.10.2/th_3'
    assert s2p_dir.is_dir()
    raw_dir = s2p_dir / 'production'
    assert raw_dir.is_dir()
    classified_dir = s2p_dir / 'classified_secular'
    assert classified_dir.is_dir()

    aamster_dir = info_dir / 'aamster/ticket_325'
    assert aamster_dir.is_dir()
    max_dir = aamster_dir / 'other_projections'
    assert max_dir.is_dir()
    corr_dir = aamster_dir / 'metric_image'
    assert corr_dir.is_dir()

    labeled_fname_list = [n for n in classified_dir.rglob('*json')]
    labeled_fname_list.sort()

    color_map_path = s2p_dir / 'color_map.json'
    with open(color_map_path, 'rb') as in_file:
        global_color_map = json.load(in_file)

    exp_id_pattern = re.compile('0-9]+')

    t0 = time.time()
    ct = 0
    with PdfPages(out_path, 'w') as pdf_handle:
        for labeled_path in labeled_fname_list:
            exp_id = exp_id_pattern.findall(str(labeled_path.name))[0]
            max_img_path = max_dir / f'{exp_id}_max_proj.png'
            if not max_img_path.is_file():
                raise RuntimeError(f'{max_img_path} is not file')
            corr_img_path = corr_dir / f'{exp_id}_correlation_proj.png'
            if not corr_img_path.is_file():
                raise RuntimeError(f'{corr_img_path} is not file')

            max_img = path_to_rgb(max_img_path)
            corr_img = path_to_rgb(corr_img_path)
            color_map = global_color_map[str(exp_id)]
            with open(labeled_path, 'rb') as in_file:
                raw_roi_list = json.load(in_file)
            fig = generate_page(
                    raw_roi_list=raw_roi_list,
                    stat_name=args.stat_name,
                    min_stat=args.min_stat,
                    min_area=args.min_area,
                    color_map=color_map,
                    corr_img=corr_img,
                    max_img=max_img,
                    experiment_id=exp_id)
            pdf_handle.savefig(fig)
            ct += 1
            print(f'{ct} in {time.time()-t0}')

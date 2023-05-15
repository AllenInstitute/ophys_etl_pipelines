import pathlib
import os
import time

import json
from typing import List, Tuple, Dict

import cv2
import h5py
import numpy as np
import pandas as pd

from argschema import ArgSchema, ArgSchemaParser, fields
from deepcell.cli.modules.create_dataset import construct_dataset, \
    VoteTallyingStrategy
from marshmallow import validates_schema, ValidationError
from scipy.signal import find_peaks, peak_prominences

from ophys_etl.types import OphysROI
from ophys_etl.utils.array_utils import get_cutout_indices, get_cutout_padding, \
    normalize_array
from ophys_etl.utils.rois import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)


class _EmptyMaskException(Exception):
    """Raised if mask is empty"""
    pass


class ClassifierArtifactsInputSchema(ArgSchema):
    experiment_id = fields.Str(
        required=True,
        description='Experiment id to generate classifier inputs for'
    )
    # Input data locations.
    video_path = fields.InputFile(
        required=True,
        description="Path to motion corrected(+denoised) video.",
    )
    roi_path = fields.InputFile(
        required=True,
        description="Path to json file containing detected ROIs",
    )
    # Output Artifact location.
    out_dir = fields.OutputDir(
        required=True,
        description="Output directory to put artifacts.",
    )

    # Artifact generation settings.
    low_quantile = fields.Float(
        required=False,
        default=0.2,
        description="Low quantile to saturate/clip to.",
    )
    high_quantile = fields.Float(
        required=False,
        default=0.99,
        description="High quantile to saturate/clip to.",
    )
    cutout_size = fields.Int(
        required=False,
        default=128,
        description="Size of square cutout in pixels.",
    )
    n_frames = fields.Int(
        default=64,
        description='Number of frames to use around a peak'
    )
    temporal_downsampling_factor = fields.Int(
        default=1,
        description='Evenly sample every nth frame so that we '
                    'have `n_frames` total frames. I.e. If it is 4, then'
                    'we start with 4 * `n_frames frames around peak and '
                    'sample every 4th frame'
    )
    is_training = fields.Boolean(
        required=True,
        description='Whether generating inputs for training or inference.'
                    'If training, will limit to only labeled ROIs'
    )
    cell_labeling_app_host = fields.Str(
        required=False,
        allow_none=True,
        default=None,
        description='Cell labeling app host, in order to pull labels'
    )
    fov_shape = fields.Tuple(
        (fields.Int(), fields.Int()),
        default=(512, 512),
        description='field of view shape'
    )
    limit_to_n_highest_peaks = fields.Int(
        default=5,
        description='Number of peaks to use when constructing clip'
    )

    @validates_schema
    def validate_cell_labeling_app_host(self, data):
        if data['is_training'] and data['cell_labeling_app_host'] is None:
            raise ValidationError('Must provide cell_labeling_app_host if '
                                  'is_training')


class ClassifierArtifactsGenerator(ArgSchemaParser):
    """Create cutouts from the average, max, correlation projects for each
    detected ROI. Additionally return the ROI mask. Store outputs as 8bit
    pngs.
    """

    default_schema = ClassifierArtifactsInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args["log_level"])

        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA",
                                              "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")
        t0 = time.time()

        video_path = pathlib.Path(self.args["video_path"])
        exp_id = video_path.name.split("_")[0]

        roi_path = pathlib.Path(self.args["roi_path"])

        with h5py.File(video_path, 'r') as f:
            mov = f['data'][()]

        with open(roi_path, "rb") as in_file:
            extract_roi_list = sanitize_extract_roi_list(
                json.load(in_file))

        if self.args['is_training']:
            selected_rois = self._get_labeled_rois_for_experiment()
        else:
            selected_rois = [roi['id'] for roi in extract_roi_list]
        selected_rois = set(selected_rois)

        self.logger.info("Creating and writing ROI artifacts...")
        brightest_peak_idxs = {}
        exp_peaks = {}

        for roi in extract_roi_list:
            if roi['id'] not in selected_rois:
                continue

            roi = extract_roi_to_ophys_roi(roi=roi)

            peaks = self._find_peaks(
                roi=roi,
                mov=mov,
                exp_id=exp_id)
            exp_peaks[roi.roi_id] = peaks
            clip, brightest_peak_idx = self._construct_clip(
                peaks=peaks,
                roi=roi
            )
            brightest_peak_idxs[roi.roi_id] = brightest_peak_idx

            np.save(
                str(pathlib.Path(self.args['out_dir']) /
                    f'clip_{exp_id}_{roi.roi_id}.npy'),
                clip
            )

        with open(pathlib.Path(self.args['out_dir']) /
                  f'brightest_peak_idx_{exp_id}.json', 'w') as f:
            f.write(json.dumps(brightest_peak_idxs, indent=2))
        with open(pathlib.Path(self.args['out_dir']) /
                  f'peaks_{exp_id}.json', 'w') as f:
            f.write(json.dumps(exp_peaks, indent=2))

        self.logger.info(f"Created ROI artifacts in {time.time()-t0:.0f} "
                         "seconds.")

    def _get_labeled_rois_for_experiment(self) -> List[int]:
        """Get labeled rois for experiment"""
        if self.args['cell_labeling_app_host'] is None:
            raise ValueError('cell_labeling_app_host needed to get '
                             'labeled rois')
        labels = construct_dataset(
            cell_labeling_app_host=self.args['cell_labeling_app_host'],
            vote_tallying_strategy=(VoteTallyingStrategy.MAJORITY
                                    if self.args['is_training'] else None)
        )

        labels = labels.set_index('experiment_id')
        if self.args['experiment_id'] not in labels.index:
            raise ValueError(
                f'No labeled rois for {self.args["experiment_id"]}')
        exp_labels = labels.loc[self.args['experiment_id']]
        roi_ids = exp_labels['roi_id']
        roi_ids = roi_ids.astype(int)
        if isinstance(exp_labels, pd.Series):
            # just 1 roi exists
            roi_ids = [roi_ids]
        else:
            roi_ids = roi_ids.tolist()
        return roi_ids

    def _find_peaks(
            self,
            mov: np.ndarray,
            roi: OphysROI,
            exp_id: str) -> List[Dict]:
        """Write peaks as calculated from trace to disk

        Parameters
        ----------
        mov : np.ndarray
            Ophys movie
        roi : ophys_etl.types.OphysROI
            ROI containing bounding box size and location.
        exp_id : str
            Id of experiment where these ROIs and images come from.
        """
        trace = mov[:,
                    roi.global_pixel_array[:, 0],
                    roi.global_pixel_array[:, 1]].mean(axis=1)
        peaks, _ = find_peaks(trace)
        prominences = peak_prominences(trace, peaks)[0]
        prominence_threshold = np.quantile(prominences, 0.999)
        peaks = peaks[prominences >= prominence_threshold]

        peaks = [{
            'peak': int(peak),
            'trace': float(trace[peak])
        } for peak in peaks]

        return peaks

    def _construct_clip(
            self,
            peaks: List[Dict],
            roi: OphysROI
    ) -> Tuple[np.ndarray, int]:
        """Constructs clip from full movie

        Returns
        -------
        Tuple[np.ndarray, int]
            Clip, where the brightest peak is in clip

        """
        peaks = sorted(peaks, key=lambda x: x['trace'])[::-1][
                :self.args['limit_to_n_highest_peaks']]
        brightest_peak = peaks[0]

        n_frames = self.args['n_frames'] * \
                   self.args['temporal_downsampling_factor']
        nframes_before_after = int(n_frames/2)

        row_indices = get_cutout_indices(
            center_dim=roi.bounding_box_center_y,
            image_dim=self.args['fov_shape'][0],
            cutout_dim=self.args['cutout_size'])
        col_indices = get_cutout_indices(
            center_dim=roi.bounding_box_center_x,
            image_dim=self.args['fov_shape'][0],
            cutout_dim=self.args['cutout_size'])

        with h5py.File(self.args['video_path'], 'r') as f:
            mov_len = f['data'].shape[0]

        peaks = sorted(peaks, key=lambda x: x['peak'])
        frame_idxs = []
        for peak in peaks:
            start_index = max(0, peak['peak'] - nframes_before_after)
            end_index = min(mov_len, peak['peak'] + nframes_before_after)
            if self.args['n_frames'] == 1:
                end_index += 1
            idxs = np.arange(start_index, end_index,
                             self.args['temporal_downsampling_factor'])
            frame_idxs += idxs.tolist()

        # h5py doesn't allow an index to be repeated
        frame_idxs = list(set(frame_idxs))

        frame_idxs = sorted(frame_idxs)

        with h5py.File(self.args['video_path'], 'r') as f:
            frames = f['data'][
                     frame_idxs,
                     row_indices[0]:row_indices[1],
                     col_indices[0]:col_indices[1]
                     ]

        input = self._get_video_clip_for_roi(
            frames=frames,
            fov_shape=self.args['fov_shape'],
            roi=roi
        )

        brightest_peak_idx = \
            frame_idxs.index(brightest_peak['peak'])
        return input, brightest_peak_idx

    def _get_video_clip_for_roi(
            self,
            frames: np.ndarray,
            roi: OphysROI,
            fov_shape: Tuple[int, int],
            normalize_quantiles: Tuple[float, float] = (0.2, 0.99)
    ):
        if len(frames) < self.args['n_frames'] * \
                self.args['limit_to_n_highest_peaks']:
            frames = _temporal_pad_frames(
                desired_seq_len=(
                        self.args['n_frames'] *
                        self.args['limit_to_n_highest_peaks']),
                frames=frames
            )

        if frames.shape[1:] != self.args['cutout_size']:
            frames = _pad_cutout(
                frames=frames,
                roi=roi,
                desired_shape=self.args['cutout_size'],
                fov_shape=self.args['fov_shape'],
                pad_mode='symmetric'
            )

        frames = normalize_array(
            array=frames,
            lower_cutoff=np.quantile(frames, normalize_quantiles[0]),
            upper_cutoff=np.quantile(frames, normalize_quantiles[1])
        )

        frames = _draw_mask_outline_on_frames(
            roi=roi,
            cutout_size=self.args['cutout_size'],
            fov_shape=fov_shape,
            frames=frames
        )
        return frames


def _generate_mask_image(
    fov_shape: Tuple[int, int],
    roi: OphysROI,
    cutout_size: int
) -> np.ndarray:
    """
    Generate mask image from `roi`, cropped to cutout_size X cutout_size

    Parameters
    ----------
    roi
        `OphysROI`

    Returns
    -------
    uint8 np.ndarray with masked region set to 255
    """
    pixel_array = roi.global_pixel_array.transpose()

    mask = np.zeros(fov_shape, dtype=np.uint8)
    mask[pixel_array[0], pixel_array[1]] = 255

    row_indices = get_cutout_indices(
        center_dim=roi.bounding_box_center_y,
        image_dim=fov_shape[0],
        cutout_dim=cutout_size)
    col_indices = get_cutout_indices(
        center_dim=roi.bounding_box_center_x,
        image_dim=fov_shape[0],
        cutout_dim=cutout_size)

    mask = mask[row_indices[0]:row_indices[1], col_indices[0]:col_indices[1]]
    if mask.shape != (cutout_size, cutout_size):
        mask = _pad_cutout(
            frames=mask,
            desired_shape=cutout_size,
            fov_shape=fov_shape,
            pad_mode='constant',
            roi=roi
        )

    if mask.sum() == 0:
        raise _EmptyMaskException(f'Mask for roi {roi.roi_id} is empty')

    return mask


def _temporal_pad_frames(
    desired_seq_len: int,
    frames: np.ndarray,
) -> np.ndarray:
    """
    Pad the frames so that the len equals desired_seq_len

    Parameters
    ----------
    desired_seq_len
        Desired number of frames
    frames
        Frames
    Returns
    -------
    frames, potentially padded
    """
    n_pad = desired_seq_len - len(frames)
    frames = np.pad(frames, mode='edge',
                    pad_width=((0, n_pad), (0, 0), (0, 0)))
    return frames


def _draw_mask_outline_on_frames(
    roi: OphysROI,
    frames: np.ndarray,
    fov_shape: Tuple[int, int],
    cutout_size: int,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    mask = _generate_mask_image(
        fov_shape=fov_shape,
        roi=roi,
        cutout_size=cutout_size
    )
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    if frames.shape[-1] != 3:
        # Make it into 3 channels, to draw a colored contour on it
        frames = np.stack([frames, frames, frames], axis=-1)

    for frame in frames:
        cv2.drawContours(frame, contours, -1,
                         color=color,
                         thickness=1)
    return frames


def _pad_cutout(
    frames: np.ndarray,
    roi: OphysROI,
    desired_shape: int,
    fov_shape: Tuple[int, int],
    pad_mode: str
) -> np.ndarray:
    """If the ROI is too close to the edge of the FOV, then we pad in order
    to have frames of the desired shape"""
    row_pad = get_cutout_padding(
        dim_center=roi.bounding_box_center_y,
        image_dim_size=fov_shape[0],
        cutout_dim=desired_shape)
    col_pad = get_cutout_padding(
        dim_center=roi.bounding_box_center_x,
        image_dim_size=fov_shape[0],
        cutout_dim=desired_shape)

    if len(frames.shape) == 3:
        # Don't pad temporal dimension
        pad_width = ((0, 0), row_pad, col_pad)
    else:
        pad_width = (row_pad, col_pad)
    kwargs = {'constant_values': 0} if pad_mode == 'constant' else {}
    return np.pad(frames,
                  pad_width=pad_width,
                  mode=pad_mode,
                  **kwargs
                  )


def _downsample_frames(
    frames: np.ndarray,
    downsampling_factor: int
) -> np.ndarray:
    """Samples every `downsampling_factor` frame from `frames`"""
    frames = frames[
        np.arange(0,
                  len(frames),
                  downsampling_factor)
    ]
    return frames


if __name__ == "__main__":
    classArtifacts = ClassifierArtifactsGenerator()
    classArtifacts.run()

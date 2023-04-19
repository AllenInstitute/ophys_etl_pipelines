import pathlib
import os
import time

import json
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd

from argschema import ArgSchema, ArgSchemaParser, fields
from deepcell.cli.modules.create_dataset import construct_dataset, \
    VoteTallyingStrategy
from marshmallow import validates_schema, ValidationError

from ophys_etl.types import OphysROI
from ophys_etl.utils.array_utils import normalize_array
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
        default=16,
        description='Total number of frames to generate surrounding peak '
                    'activation for ROI'
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
        for roi in extract_roi_list:
            if roi['id'] not in selected_rois:
                continue

            roi = extract_roi_to_ophys_roi(roi=roi)

            self._write_frames(roi=roi,
                               mov=mov,
                               exp_id=exp_id)

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

    def _write_frames(self,
                      mov: np.ndarray,
                      roi: OphysROI,
                      exp_id: str):
        """Write sequence of frames for ROI to disk

        Parameters
        ----------
        mov : np.ndarray
            Ophys movie
        roi : ophys_etl.types.OphysROI
            ROI containing bounding box size and location.
        exp_id : str
            Id of experiment where these ROIs and images come from.
        """
        desired_shape = (self.args['cutout_size'], self.args['cutout_size'])

        trace = mov[:,
                    roi.global_pixel_array[:, 0],
                    roi.global_pixel_array[:, 1]].mean(axis=1)
        peak_activation_frame = trace.argmax()

        n_frames = \
            self.args['n_frames'] * self.args['temporal_downsampling_factor']
        nframes_before_after = int(n_frames/2)
        frames = mov[
                 max(0, peak_activation_frame - nframes_before_after):
                 peak_activation_frame + nframes_before_after]

        frames = _pad_frames(
            desired_seq_len=n_frames,
            frames=frames
        )

        frames = _crop_frames(
            frames=frames,
            roi=roi,
            desired_shape=desired_shape
        )

        frames = _downsample_frames(
            frames=frames,
            downsampling_factor=self.args['temporal_downsampling_factor']
        )

        frames = normalize_array(
            array=frames,
            lower_cutoff=np.quantile(frames, self.args['low_quantile']),
            upper_cutoff=np.quantile(frames, self.args['high_quantile'])
        )

        frames = _draw_mask_outline_on_frames(
            roi=roi,
            cutout_size=self.args['cutout_size'],
            fov_shape=self.args['fov_shape'],
            frames=frames
        )
        name = f'{exp_id}_{roi.roi_id}.npy'

        if frames.shape[1:] != (*desired_shape, 3):
            msg = f"{exp_id}_{roi.roi_id} has shape {frames.shape}"
            raise RuntimeError(msg)

        out_path = pathlib.Path(self.args['out_dir']) / name

        np.save(str(out_path), frames)


def _generate_mask_image(
    fov_shape: Tuple[int, int],
    roi: OphysROI,
    cutout_size: int,
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

    mask = roi.get_centered_cutout(
            image=mask,
            height=cutout_size,
            width=cutout_size,
            pad_mode='constant'
    )

    if mask.sum() == 0:
        raise _EmptyMaskException(f'Mask for roi {roi.roi_id} is empty')

    return mask


def _pad_frames(
    desired_seq_len: int,
    frames: np.ndarray,
) -> np.ndarray:
    """
    If the peak activation frame happens too close to the beginning
    or end of the movie, then we pad with black in order to have
    self.args['n_frames'] total frames

    Parameters
    ----------
    desired_seq_len
        Desired number of frames surrounding `peak_activation_frame_idx`
    frames
        Frames
    Returns
    -------
    frames, potentially padded
    """
    n_pad = desired_seq_len - len(frames)
    frames = np.concatenate([
        frames,
        np.zeros((n_pad, *frames.shape[1:]), dtype=frames.dtype),
    ])
    return frames


def _draw_mask_outline_on_frames(
    roi: OphysROI,
    frames: np.ndarray,
    fov_shape: Tuple[int, int],
    cutout_size: int
) -> np.ndarray:
    mask = _generate_mask_image(
        fov_shape=fov_shape,
        roi=roi,
        cutout_size=cutout_size
    )
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    # Make it into 3 channels, to draw a colored contour on it
    frames = np.stack([frames, frames, frames], axis=-1)

    for frame in frames:
        cv2.drawContours(frame, contours, -1,
                         color=(0, 255, 0),
                         thickness=1)
    return frames


def _crop_frames(
    frames: np.ndarray,
    roi: OphysROI,
    desired_shape: Tuple[int, int]
) -> np.ndarray:
    frames_cropped = np.zeros_like(frames,
                                   shape=(frames.shape[0], *desired_shape))
    for i, frame in enumerate(frames):
        frames_cropped[i] = roi.get_centered_cutout(
            image=frames[i],
            height=desired_shape[0],
            width=desired_shape[1],
            pad_mode='symmetric'
        )
    return frames_cropped


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

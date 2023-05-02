import pathlib
import os
import time

import json
from typing import List

import h5py
import numpy as np
import pandas as pd

from argschema import ArgSchema, ArgSchemaParser, fields
from deepcell.cli.modules.create_dataset import construct_dataset, \
    VoteTallyingStrategy
from marshmallow import validates_schema, ValidationError
from scipy.signal import find_peaks, peak_prominences

from ophys_etl.types import OphysROI
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

            self._find_peaks(roi=roi,
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

    def _find_peaks(
            self,
            mov: np.ndarray,
            roi: OphysROI,
            exp_id: str):
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

        filename = pathlib.Path(self.args['out_dir']) / \
            f'peaks_{exp_id}_{roi.roi_id}.json'
        with open(filename, 'w') as f:
            f.write(json.dumps(peaks, indent=2))


if __name__ == "__main__":
    classArtifacts = ClassifierArtifactsGenerator()
    classArtifacts.run()

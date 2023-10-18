import pathlib
import os
import time

import json
from typing import Dict, List

import h5py
import numpy as np
import PIL
import pandas as pd

from argschema import ArgSchema, ArgSchemaParser, fields
from deepcell.cli.modules.create_dataset import construct_dataset, \
    VoteTallyingStrategy
from deepcell.cli.schemas.data import ChannelField
from deepcell.datasets.channel import Channel, channel_filename_prefix_map
from marshmallow import validates_schema, ValidationError
from marshmallow.validate import OneOf
from ophys_etl.schemas._roi_schema import ExtractROISchema

from ophys_etl.modules.segmentation.graph_utils.conversion import \
    graph_to_img
from ophys_etl.types import OphysROI, DenseROI
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.utils.motion_border import get_max_correction_from_file, \
    MaxFrameShift, motion_border_from_max_shift
from ophys_etl.utils.video_utils import get_max_and_avg
from ophys_etl.utils.rois import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi, is_inside_motion_border)


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
    rois = fields.Nested(
        ExtractROISchema,
        many=True,
        required=True,
        description="Detected ROIs"
    )
    channels = fields.List(
        ChannelField(),
        required=True,
        description="List of channels to generate thumbnails for"
    )
    graph_path = fields.InputFile(
        required=False,
        allow_none=True,
        default=None,
        description="Path to pickle file containing full movie graph."
                    "Required only if correlation projection is given in "
                    "`channels`",
    )
    motion_correction_shifts_path = fields.InputFile(
        required=True,
        description='Path to file containing motion correction shifts',
        validate=lambda x: pathlib.Path(x).suffix == '.csv'
    )

    # Output Artifact location.
    thumbnails_out_dir = fields.OutputDir(
        required=True,
        description="Output directory to put thumbnails.",
    )
    roi_meta_out_dir = fields.OutputDir(
        required=False,
        description="Output directory to put roi metadata.",
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
    pad_mode = fields.String(
        default='constant',
        description='Pad mode for all channels except the mask when the ROI '
                    'thumbnail would extend past the edge of the frame. '
                    'The mask always gets constant padding.',
        validate=OneOf(('constant', 'symmetric'))
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
    def validate_graph_path(self, data, **kwargs):
        if Channel.CORRELATION_PROJECTION.value in data['channels']:
            if data['graph_path'] is None:
                raise ValidationError('graph_path needs to be provided if '
                                      'passed as a channel')

    @validates_schema
    def validate_cell_labeling_app_host(self, data, **kwargs):
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

        graph_path = pathlib.Path(self.args["graph_path"]) \
            if self.args["graph_path"] is not None else None

        imgs = {}

        with h5py.File(video_path, 'r') as f:
            mov = f['data'][()]
        proj = get_max_and_avg(video=mov)
        self.logger.info("Calculated mean and max images...")

        imgs[Channel.MAX_PROJECTION] = proj['max']
        imgs[Channel.AVG_PROJECTION] = proj['avg']

        if Channel.CORRELATION_PROJECTION.value in self.args['channels']:
            corr_img = graph_to_img(graph_path)
            imgs[Channel.CORRELATION_PROJECTION] = corr_img
            self.logger.info("Calculated correlation image...")

        for channel, img in imgs.items():
            imgs[channel] = self._normalize_image(
                img=img,
                low_cutoff=self.args["low_quantile"],
                high_cutoff=self.args["high_quantile"]
            )

        self.logger.info("Normalized images...")

        extract_roi_list = sanitize_extract_roi_list(
            input_roi_list=self.args['rois'])

        if self.args['is_training']:
            selected_rois = self._get_labeled_rois_for_experiment()
        else:
            selected_rois = [str(roi['id']) for roi in extract_roi_list]
        selected_rois = set(selected_rois)

        maximum_motion_shift = get_max_correction_from_file(
            input_csv=self.args['motion_correction_shifts_path']
        )
        roi_meta = {}

        self.logger.info("Creating and writing ROI artifacts...")
        for roi in extract_roi_list:
            if str(roi['id']) not in selected_rois:
                continue
            roi_meta[roi['id']] = {
                'is_inside_motion_border': (
                    self._calc_is_roi_inside_motion_border(
                        roi=roi,
                        motion_shifts=maximum_motion_shift
                    )
                )
            }

            roi = extract_roi_to_ophys_roi(roi=roi)
            imgs[Channel.MASK] = self._generate_mask_image(
                roi=roi
            )

            imgs[Channel.MAX_ACTIVATION] = \
                self._generate_max_activation_image(
                    mov=mov,
                    roi=roi
                )

            self._write_thumbnails(roi=roi,
                                   imgs=imgs,
                                   exp_id=exp_id)

        roi_meta_out_path = pathlib.Path(self.args['roi_meta_out_dir']) / \
            f'roi_meta_{self.args["experiment_id"]}.json'
        with open(roi_meta_out_path, 'w') as f:
            f.write(json.dumps(roi_meta, indent=2))
        self.logger.info(f'Wrote ROI metadata to {roi_meta_out_path}')

        self.logger.info(f"Created ROI artifacts in {time.time()-t0:.0f} "
                         "seconds.")

    def _calc_is_roi_inside_motion_border(
        self,
        roi: DenseROI,
        motion_shifts: MaxFrameShift
    ) -> bool:
        motion_border = motion_border_from_max_shift(
            max_shift=motion_shifts
        )
        roi = roi.copy()
        roi['max_correction_up'] = motion_border.top
        roi['max_correction_down'] = motion_border.bottom
        roi['max_correction_left'] = motion_border.left_side
        roi['max_correction_right'] = motion_border.right_side

        return is_inside_motion_border(
            roi=roi,
            movie_shape=self.args['fov_shape']
        )

    @staticmethod
    def _normalize_image(
        img: np.ndarray,
        low_cutoff: float,
        high_cutoff: float
    ):
        """Normalize image to between low_cutoff and high_cutoff quantiles
        and then cast to uint8

        Parameters
        ----------
        img
            Image to normalize
        low_cutoff
            Low quantile
        high_cutoff
            High quantile

        Returns
        -------
        np.ndarray normalized between low_cutoff and high_cutoff and cast
        as uint8

        """
        q0, q1 = np.quantile(img, (low_cutoff, high_cutoff))
        return normalize_array(
            array=img,
            lower_cutoff=q0,
            upper_cutoff=q1
        )

    def _generate_mask_image(
        self,
        roi: OphysROI
    ) -> np.ndarray:
        """
        Generate mask image from `roi`

        Parameters
        ----------
        roi
            `OphysROI`

        Returns
        -------
        uint8 np.ndarray with masked region set to 255
        """
        pixel_array = roi.global_pixel_array.transpose()

        mask = np.zeros(self.args['fov_shape'], dtype=np.uint8)
        mask[pixel_array[0], pixel_array[1]] = 255

        return mask

    @staticmethod
    def _generate_max_activation_image(
        mov: np.ndarray,
        roi: OphysROI
    ) -> np.ndarray:
        """
        Generates "max activation" image which is the frame of peak brightness
        for `roi`

        Parameters
        ----------
        mov
            Ophys movie
        roi
            `OphysROI`

        Returns
        -------
        np.ndarray of shape fov_shape
        """
        trace = mov[:,
                    roi.global_pixel_array[:, 0],
                    roi.global_pixel_array[:, 1]].mean(axis=1)
        img = mov[trace.argmax()]
        return img

    def _get_labeled_rois_for_experiment(self) -> List[int]:
        """Get labeled rois for experiment"""
        if self.args['cell_labeling_app_host'] is None:
            raise ValueError('cell_labeling_app_host needed to get '
                             'labeled rois')
        labels = construct_dataset(
            cell_labeling_app_host=self.args['cell_labeling_app_host'],
            vote_tallying_strategy=VoteTallyingStrategy.MAJORITY
        )
        labels['roi_id'] = labels['roi_id'].astype(str)

        labels = labels.set_index('experiment_id')
        if self.args['experiment_id'] not in labels.index:
            raise ValueError(
                f'No labeled rois for {self.args["experiment_id"]}')
        exp_labels = labels.loc[self.args['experiment_id']]
        roi_ids = exp_labels['roi_id']
        if isinstance(exp_labels, pd.Series):
            # just 1 roi exists
            roi_ids = [roi_ids]
        else:
            roi_ids = roi_ids.tolist()
        return roi_ids

    def _write_thumbnails(self,
                          roi: OphysROI,
                          imgs: Dict[Channel, np.ndarray],
                          exp_id: str):
        """Compute image cutout artifacts for an ROI.

        Parameters
        ----------
        roi : ophys_etl.types.OphysROI
            ROI containing bounding box size and location.
        imgs
            Map between `deepcell.datasets.channel.Channel` and img
        exp_id : str
            Id of experiment where these ROIs and images come from.
        """
        desired_shape = (self.args['cutout_size'], self.args['cutout_size'])

        # Get cutouts
        for channel in self.args['channels']:
            channel = getattr(Channel, channel)
            img = imgs[channel]
            thumbnail = roi.get_centered_cutout(
                image=img,
                height=self.args['cutout_size'],
                width=self.args['cutout_size'],
                pad_mode=('constant' if channel == Channel.MASK
                          else self.args['pad_mode'])
            )

            # For max activation, need to normalize the thumbnail
            if channel == Channel.MAX_ACTIVATION:
                thumbnail = self._normalize_image(
                    img=thumbnail,
                    low_cutoff=self.args["low_quantile"],
                    high_cutoff=self.args["high_quantile"]
                )

            # Store the ROI cutouts to disk.
            roi_id = roi.roi_id
            if channel == Channel.MASK:
                if thumbnail.sum() <= 0:
                    msg = f"{exp_id}_{roi_id} has bad mask {thumbnail.shape}"
                    self.logger.warn(msg)

            name = f'{channel_filename_prefix_map[channel]}_{exp_id}_' \
                   f'{roi_id}.png'
            if thumbnail.shape != desired_shape:
                msg = f"{name} has shape {thumbnail.shape}"
                raise RuntimeError(msg)
            thumbnail = PIL.Image.fromarray(thumbnail)
            out_path = pathlib.Path(self.args['thumbnails_out_dir']) / name
            thumbnail.save(out_path)


if __name__ == "__main__":
    classArtifacts = ClassifierArtifactsGenerator()
    classArtifacts.run()

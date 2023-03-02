import pathlib
import os
import time

import json
from typing import Dict

import numpy as np
import PIL

from argschema import ArgSchema, ArgSchemaParser, fields
from deepcell.cli.schemas.data import ChannelField
from deepcell.datasets.channel import Channel, channel_filename_prefix_map
from marshmallow import validates_schema, ValidationError

from ophys_etl.modules.segmentation.graph_utils.conversion import \
    graph_to_img
from ophys_etl.types import ExtractROI, OphysROI
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.utils.video_utils import get_max_and_avg
from ophys_etl.utils.rois import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)


class ClassifierArtifactsInputSchema(ArgSchema):
    # Input data locations.
    video_path = fields.InputFile(
        required=True,
        description="Path to motion corrected(+denoised) video.",
    )
    roi_path = fields.InputFile(
        required=True,
        description="Path to json file containing detected ROIs",
    )
    graph_path = fields.InputFile(
        required=False,
        allow_none=True,
        default=None,
        description="Path to pickle file containing full movie graph."
                    "Required only if correlation projection is given in "
                    "`channels`",
    )
    channels = fields.List(
        ChannelField(),
        required=True,
        description="List of channels to generate thumbnails for",
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
    selected_rois = fields.List(
        fields.Int,
        required=False,
        allow_none=True,
        default=None,
        description="Specific subset of ROIs by ROI id in the experiment FOV "
                    "to produce artifacts for. Only ROIs specified in this "
                    "will have artifacts output.",
    )
    fov_shape = fields.Tuple(
        (fields.Int(), fields.Int()),
        default=(512, 512),
        description='field of view shape'
    )

    @validates_schema
    def validate_graph_path(self, data):
        if Channel.CORRELATION_PROJECTION.value in data['channels']:
            if data['graph_path'] is None:
                raise ValidationError('graph_path needs to be provided if '
                                      'passed as a channel')


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
        graph_path = pathlib.Path(self.args["graph_path"]) \
            if self.args["graph_path"] is not None else None

        imgs = {}

        proj = get_max_and_avg(video_path)
        self.logger.info("Calculated mean and max images...")

        imgs[Channel.MAX_PROJECTION] = proj['max']
        imgs[Channel.AVG_PROJECTION] = proj['avg']

        if Channel.CORRELATION_PROJECTION.value in self.args['channels']:
            corr_img = graph_to_img(graph_path)
            imgs[Channel.CORRELATION_PROJECTION] = corr_img
            self.logger.info("Calculated correlation image...")

        quantiles = (self.args["low_quantile"], self.args["high_quantile"])
        for channel, img in imgs.items():
            q0, q1 = np.quantile(img, quantiles)
            imgs[channel] = normalize_array(
                array=img,
                lower_cutoff=q0,
                upper_cutoff=q1
            )

        self.logger.info("Normalized images...")

        with open(roi_path, "rb") as in_file:
            extract_roi_list = sanitize_extract_roi_list(
                json.load(in_file))

        selected_rois = self.args['selected_rois']
        if selected_rois is None:
            selected_rois = [roi['id'] for roi in extract_roi_list]
        selected_rois = set(selected_rois)

        self.logger.info("Creating and writing ROI artifacts...")
        for roi in extract_roi_list:
            if roi['id'] not in selected_rois:
                continue

            roi = extract_roi_to_ophys_roi(roi=roi)
            imgs[Channel.MASK] = self._generate_mask_image(
                roi=roi
            )

            self._write_thumbnails(roi=roi,
                                   imgs=imgs,
                                   exp_id=exp_id)

        self.logger.info(f"Created ROI artifacts in {time.time()-t0:.0f} "
                         "seconds.")

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
                width=self.args['cutout_size']
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
            out_path = pathlib.Path(self.args['out_dir']) / name
            thumbnail.save(out_path)


if __name__ == "__main__":
    classArtifacts = ClassifierArtifactsGenerator()
    classArtifacts.run()

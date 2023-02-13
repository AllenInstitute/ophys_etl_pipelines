import pathlib
import os
import time

import json
import numpy as np
import PIL

from argschema import ArgSchema, ArgSchemaParser, fields
from ophys_etl.modules.segmentation.graph_utils.conversion import \
    graph_to_img
from ophys_etl.types import ExtractROI
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
        required=True,
        description="Path to pickle file containing full movie graph.",
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
        graph_path = pathlib.Path(self.args["graph_path"])

        proj = get_max_and_avg(video_path)
        self.logger.info("Calculated mean and max images...")
        avg_img = proj["avg"]
        max_img = proj["max"]
        corr_img = graph_to_img(graph_path)
        self.logger.info("Calculated correlation image...")

        quantiles = (self.args["low_quantile"], self.args["high_quantile"])
        q0, q1 = np.quantile(max_img, quantiles)
        max_img = normalize_array(array=max_img,
                                  lower_cutoff=q0,
                                  upper_cutoff=q1)

        q0, q1 = np.quantile(avg_img, quantiles)
        avg_img = normalize_array(array=avg_img,
                                  lower_cutoff=q0,
                                  upper_cutoff=q1)

        q0, q1 = np.quantile(corr_img, quantiles)
        corr_img = normalize_array(array=corr_img,
                                   lower_cutoff=q0,
                                   upper_cutoff=q1)
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
            self._write_thumbnails(extract_roi=roi,
                                   max_img=max_img,
                                   avg_img=avg_img,
                                   corr_img=corr_img,
                                   exp_id=exp_id)

        self.logger.info(f"Created ROI artifacts in {time.time()-t0:.0f} "
                         "seconds.")

    def _write_thumbnails(self,
                          extract_roi: ExtractROI,
                          max_img: np.ndarray,
                          avg_img: np.ndarray,
                          corr_img: np.ndarray,
                          exp_id: int):
        """Compute image cutout artifacts for an ROI.

        Parameters
        ----------
        extract_roi : ophys_etl.types.ExtractROI
            ROI containing bounding box size and location.
        max_img : np.ndarray
            Max projection of movie.
        avg_img : np.ndarray
            Mean projection of movie.
        corr_img : np.ndarray
            Correlation projection of movie.
        exp_id : int
            Id of experiment where these ROIs and images come from.
        """
        desired_shape = (self.args['cutout_size'], self.args['cutout_size'])

        # Get the ROI.
        ophys_roi = extract_roi_to_ophys_roi(extract_roi)
        pixel_array = ophys_roi.global_pixel_array.transpose()

        # Create the mask image and set the masked value.
        mask = np.zeros(max_img.shape, dtype=np.uint8)
        mask[pixel_array[0], pixel_array[1]] = 255

        # Get cutouts
        max_thumbnail = ophys_roi.get_centered_cutout(
            image=max_img,
            height=self.args['cutout_size'],
            width=self.args['cutout_size'])
        avg_thumbnail = ophys_roi.get_centered_cutout(
            image=avg_img,
            height=self.args['cutout_size'],
            width=self.args['cutout_size'])
        corr_thumbnail = ophys_roi.get_centered_cutout(
            image=corr_img,
            height=self.args['cutout_size'],
            width=self.args['cutout_size'])
        mask_thumbnail = ophys_roi.get_centered_cutout(
            image=mask,
            height=self.args['cutout_size'],
            width=self.args['cutout_size'])

        # Store the ROI cutouts to disk.
        roi_id = ophys_roi.roi_id
        if mask_thumbnail.sum() <= 0:
            msg = f"{exp_id}_{roi_id} has bad mask {mask_thumbnail.shape}"
            self.logger.warning(msg)
        for img, name in zip((max_thumbnail, avg_thumbnail,
                              corr_thumbnail, mask_thumbnail),
                             (f"max_{exp_id}_{roi_id}.png",
                              f"avg_{exp_id}_{roi_id}.png",
                              f"correlation_{exp_id}_{roi_id}.png",
                              f"mask_{exp_id}_{roi_id}.png")):

            if img.shape != desired_shape:
                msg = f"{name} has shape {img.shape}"
                raise RuntimeError(msg)
            img = PIL.Image.fromarray(img)
            out_path = pathlib.Path(self.args['out_dir']) / name
            img.save(out_path)


if __name__ == "__main__":
    classArtifacts = ClassifierArtifactsGenerator()
    classArtifacts.run()

import pathlib
import os
import time
from typing import Tuple

import json

import h5py
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
    artifact_path = fields.InputFile(
        required=True,
        description='Path to h5 file containing trace dataset'
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
        max_q0, max_q1 = np.quantile(max_img, quantiles)
        max_img = normalize_array(array=max_img,
                                  lower_cutoff=max_q0,
                                  upper_cutoff=max_q1)

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
                                   exp_id=exp_id,
                                   max_low_quantile=max_q0,
                                   max_high_quantile=max_q1)

        self.logger.info(f"Created ROI artifacts in {time.time()-t0:.0f} "
                         "seconds.")

    def _write_thumbnails(self,
                          extract_roi: ExtractROI,
                          max_img: np.ndarray,
                          avg_img: np.ndarray,
                          corr_img: np.ndarray,
                          exp_id: int,
                          max_low_quantile,
                          max_high_quantile):
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

        # Get the max activation image
        max_activation_img = self._get_max_activation_img(
            roi_id=ophys_roi.roi_id)

        # Create the mask image and set the masked value.
        mask = np.zeros(max_img.shape, dtype=np.uint8)
        mask[pixel_array[0], pixel_array[1]] = 255

        # Compute center of cutout from ROI bounding box.
        center_row = ophys_roi.bounding_box_center_y
        center_col = ophys_roi.bounding_box_center_x

        # Find the indices of the desired cutout in the image.
        row_indices = self._get_cutout_indices(center_row,
                                               max_img.shape[0])
        col_indices = self._get_cutout_indices(center_col,
                                               max_img.shape[1])

        # Create our cutouts.
        max_thumbnail = max_img[row_indices[0]:row_indices[1],
                                col_indices[0]:col_indices[1]]
        avg_thumbnail = avg_img[row_indices[0]:row_indices[1],
                                col_indices[0]:col_indices[1]]
        corr_thumbnail = corr_img[row_indices[0]:row_indices[1],
                                  col_indices[0]:col_indices[1]]
        mask_thumbnail = mask[row_indices[0]:row_indices[1],
                              col_indices[0]:col_indices[1]]
        max_activation_thumbnail = max_activation_img[
                                 row_indices[0]:row_indices[1],
                                 col_indices[0]:col_indices[1]]
        max_activation_thumbnail = normalize_array(
            array=max_activation_thumbnail,
            lower_cutoff=max_low_quantile,
            upper_cutoff=max_high_quantile
        )

        # Find if we need to pad the image.
        row_pad = self._get_padding(center_row, max_img.shape[0])
        col_pad = self._get_padding(center_col, max_img.shape[1])

        # Pad the cutouts if needed.
        padding = (row_pad, col_pad)
        max_thumbnail = np.pad(max_thumbnail,
                               pad_width=padding, mode="constant",
                               constant_values=0)
        avg_thumbnail = np.pad(avg_thumbnail,
                               pad_width=padding, mode="constant",
                               constant_values=0)
        corr_thumbnail = np.pad(corr_thumbnail,
                                pad_width=padding, mode="constant",
                                constant_values=0)
        mask_thumbnail = np.pad(mask_thumbnail,
                                pad_width=padding, mode="constant",
                                constant_values=0)
        max_activation_thumbnail = np.pad(max_activation_thumbnail,
                                          pad_width=padding, mode="constant",
                                          constant_values=0)

        # Store the ROI cutouts to disk.
        roi_id = ophys_roi.roi_id
        if mask_thumbnail.sum() <= 0:
            msg = f"{exp_id}_{roi_id} has bad mask {mask_thumbnail.shape}"
            raise RuntimeError(msg)
        for img, name in zip((max_activation_thumbnail,),
                             (f"max_activation_{exp_id}_{roi_id}.png",)):

            if img.shape != desired_shape:
                msg = f"{name} has shape {img.shape}"
                raise RuntimeError(msg)
            img = PIL.Image.fromarray(img)
            out_path = pathlib.Path(self.args['out_dir']) / name
            img.save(out_path)

    def _get_cutout_indices(
        self,
        center_dim: int,
        image_dim: int,
    ) -> Tuple[int, int]:
        """Find the min/max indices of the cutout within the image size.

        Parameters
        ----------
        center_dim : int
            ROI center coordinate in the dimension of interest.
        image_dim : int
            Image dimension size.

        Returns
        -------
        cutout_indices : Tuple[int, int]
            Indices in the cutout to that cover the ROI in one dimension.
        """
        # Get size of cutout.
        lowside = max(0, center_dim - self.args['cutout_size'] // 2)
        highside = min(image_dim, center_dim + self.args['cutout_size'] // 2)
        return (lowside, highside)

    def _get_padding(self,
                     dim_center: int,
                     image_dim_size: int) -> Tuple[int, int]:
        """If the requested cutout size is beyond any dimension of the image,
        found how much we need to pad by.

        Parameters
        ----------
        dim_center : int
            Index of the center of the ROI bbox in one of the image dimensions
            (row, col)
        image_dim_size : int
            Size of the image in the dimension we are testing for padding.

        Returns
        -------
        padding : Tuple[int, int]
            Amount to pad on at the beginning and/or end of the cutout.
        """
        # If the difference between center and cutout size is less than zero,
        # we need to pad.
        lowside_pad = np.abs(
            min(0, dim_center - self.args['cutout_size'] // 2))
        # If the difference between the center plus the cutout size is
        # bigger than the image size, we need to pad.
        highside_pad = max(
            0, dim_center + self.args['cutout_size'] // 2 - image_dim_size)
        return (lowside_pad, highside_pad)

    def _get_max_activation_img(self, roi_id: int) -> np.ndarray:
        """
        Gets the frame from the video where this roi shows the most activity

        Parameters
        ----------
        roi_id

        Returns
        -------
        frame from the video where this roi shows the most activity

        """
        with h5py.File(self.args['artifact_path'], 'r') as f:
            trace = f['traces'][str(roi_id)][()]
        with h5py.File(self.args['video_path'], 'r') as f:
            img = f['data'][trace.argmax()]
        return img


if __name__ == "__main__":
    classArtifacts = ClassifierArtifactsGenerator()
    classArtifacts.run()

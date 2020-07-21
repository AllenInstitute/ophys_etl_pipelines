import json

import argschema
import marshmallow as mm
import numpy as np
import h5py

from ophys_etl.transforms.roi_transforms import (binarize_roi_mask,
                                                 suite2p_rois_to_coo,
                                                 coo_rois_to_old)
from ophys_etl.transforms.data_loaders import get_max_correction_border


class BinarizeAndCreationException(Exception):
    pass


class BinarizeAndCreateROIsInputSchema(argschema.ArgSchema):

    suite2p_stat_path = argschema.fields.InputFile(
        required=True,
        description="File containing the segmented rois generated"
                    "during a run of Suite2p source extraction")

    motion_corrected_video = argschema.fields.InputFile(
        required=True,
        description="Path to a motion corrected ophys experiment movie (*.h5)"
    )

    motion_correction_values = argschema.fields.InputFile(
        required=True,
        description="Motion correction values in each direction stored in csv "
                    "format"
    )

    output_json = argschema.fields.OutputFile(
        required=True,
        description="Intended destination of output json"
    )

    maximum_motion_shift = argschema.fields.Float(
        default=30.0,
        required=False,
        allow_none=False,
        description="The maximum allowable motion shift for a frame before "
                    "it is considered an anomaly and thrown out of processing"
    )

    abs_threshold = argschema.fields.Float(
        default=None,
        required=False,
        allow_none=True,
        description="The absolute threshold to binarize ROI masks against. "
                    "If not provided will use quantile to generate threshold."
    )

    binary_quantile = argschema.fields.Float(
        default=None,
        required=False,
        allow_none=True,
        description="The quantile against which an ROI is binarized. If not "
                    "provided will use default function value of 0.1."
    )

    @mm.post_load()
    def check_args(self, data, **kwargs):
        if (data['binary_quantile'] <= 0) or (data['abs_threshold'] <= 0):
            raise BinarizeAndCreationException("Binary quantile and abs "
                                               "threshold must be positive "
                                               "values")


class BinarizerAndROICreator(argschema.ArgSchemaParser):

    def binarize_and_create(self):

        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # load in the rois from the stat file and movie path for shape
        self.logger.info("Loading suite2p ROIs and video size.")
        suite2p_stats = np.load(self.args['suite2p_stat_path'])
        with h5py.File(self.args['motion_corrected_video'], 'r') as open_vid:
            movie_shape = open_vid['data'].shape

        # binarize the masks
        self.logger.info("Binarizing the ROIs created by Suite2p.")
        coo_rois = suite2p_rois_to_coo(suite2p_stats, movie_shape)

        binarized_coo_rois = []
        for coo_roi in coo_rois:
            if self.args['binary_quantile'] is not None:
                binary_mask = binarize_roi_mask(coo_roi,
                                                self.args['abs_threshold'],
                                                self.args['binary_quantile'])
            else:
                binary_mask = binarize_roi_mask(coo_roi,
                                                self.args['abs_threshold'])
            binarized_coo_rois.append(binary_mask)
        self.logger.info("Binarized ROIs from Suite2p, total binarized: "
                         f"{len(binarized_coo_rois)}")

        # load the motion correction values
        self.logger.info("Loading motion correction border values from "
                         f" {self.args['motion_correction_values']}")
        motion_border = get_max_correction_border(
            self.args['motion_correction_values'],
            self.args['maximum_motion_shift'])

        # create the rois
        self.logger.info("Transforming ROIs to old segmentation style.")
        old_segmentation_rois = coo_rois_to_old(binarized_coo_rois,
                                                motion_border,
                                                movie_shape)

        # save the rois as a json file to output directory
        self.logger.info("Writing old style ROIs to json file at "
                         f"{self.args['output_json']}")
        with open(self.args["output_json"], 'w') as open_out:
            json.dump(old_segmentation_rois, open_out)


if __name__ == '__main__':
    roi_creator_and_binarizer = BinarizerAndROICreator()
    roi_creator_and_binarizer.binarize_and_create()

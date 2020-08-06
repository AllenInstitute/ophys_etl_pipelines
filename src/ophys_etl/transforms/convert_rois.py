from pathlib import Path

from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.fields import (List, InputFile, Str, OutputFile, Float,
                              Nested, Int, Bool)
from marshmallow.validate import Range
import numpy as np
import pandas as pd
import h5py


from ophys_etl.transforms.roi_transforms import (binarize_roi_mask,
                                                 suite2p_rois_to_coo,
                                                 coo_rois_to_lims_compatible)
from ophys_etl.transforms.data_loaders import get_max_correction_values


class BinarizeAndCreationException(Exception):
    pass


class BinarizeAndCreateROIsInputSchema(ArgSchema):

    suite2p_stat_path = Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to s2p output stat file containing ROIs generated "
                     "during source extraction")
    )

    motion_corrected_video = Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to motion corrected video file *.h5")
    )

    motion_correction_values = InputFile(
        required=True,
        description=("Path to motion correction values for each frame "
                     "stored in .csv format")
    )

    output_json = OutputFile(
        required=True,
        description=("Path to a file to write output data.")
    )

    maximum_motion_shift = Float(
        missing=30.0,
        required=False,
        allow_none=False,
        description=("The maximum allowable motion shift for a frame in pixels"
                     " before it is considered an anomaly and thrown out of "
                     "processing")
    )

    abs_threshold = Float(
        missing=None,
        required=False,
        allow_none=True,
        description=("The absolute threshold to binarize ROI masks against. "
                     "If not provided will use quantile to generate "
                     "threshold.")
    )

    binary_quantile = Float(
        missing=0.1,
        validate=Range(min=0, max=1),
        description=("The quantile against which an ROI is binarized. If not "
                     "provided will use default function value of 0.1.")
    )


class LIMSCompatibleROIFormat(DefaultSchema):
    id = Int(required=True,
             description=("Unique ID of the ROI, get's overwritten writting "
                          "to LIMS"))
    x = Int(required=True,
            description="X location of top left corner of ROI in pixels")
    y = Int(required=True,
            description="Y location of top left corner of ROI in pixels")
    width = Int(required=True,
                description="Width of the ROI in pixels")
    height = Int(required=True,
                 description="Height of the ROI in pixels")
    valid_roi = Bool(required=True,
                     description=("Boolean indicating if the ROI is a valid "
                                  "cell or not"))
    mask_matrix = List(List(Bool), required=True,
                       description=("Bool nested list describing which pixels "
                                    "in the ROI area are part of the cell"))
    max_correction_up = Float(required=True,
                              description=("Max correction in pixels in the "
                                           "up direction"))
    max_correction_down = Float(required=True,
                                description=("Max correction in pixels in the "
                                             "down direction"))
    max_correction_left = Float(required=True,
                                description=("Max correction in pixels in the "
                                             "left direction"))
    max_correction_right = Float(required=True,
                                 description="Max correction in the pixels in "
                                             "the right direction")
    mask_image_plane = Int(required=True,
                           description=("The old segmentation pipeline stored "
                                        "overlapping ROIs on separate image "
                                        "planes. For compatibility purposes, "
                                        "this field must be kept, but will "
                                        "always be set to zero for the new "
                                        "updated pipeline"))
    exclusion_labels = List(Str, required=True,
                            description=("LIMS ExclusionLabel names used to "
                                         "track why a given ROI is not "
                                         "considered a valid_roi. (examples: "
                                         "motion_border, "
                                         "classified_as_not_cell)"))


class BinarizeAndCreateROIsOutputSchema(DefaultSchema):
    LIMS_compatible_rois = Nested(LIMSCompatibleROIFormat,
                                  many=True)


class BinarizerAndROICreator(ArgSchemaParser):
    default_schema = BinarizeAndCreateROIsInputSchema
    default_output_schema = BinarizeAndCreateROIsOutputSchema

    def binarize_and_create(self):
        """
        This function takes ROIs (regions of interest) outputted from
        suite2p in a stat.npy file and converts them to a LIMS compatible
        data format for storage and further processing. This process
        binarizes the masks and then changes the formatting before writing
        to a json output file.
        """

        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # load in the rois from the stat file and movie path for shape
        self.logger.info("Loading suite2p ROIs and video size.")
        suite2p_stats = np.load(self.args['suite2p_stat_path'],
                                allow_pickle=True)
        with h5py.File(self.args['motion_corrected_video'], 'r') as open_vid:
            movie_shape = open_vid['data'][0].shape

        # binarize the masks
        self.logger.info("Binarizing the ROIs created by Suite2p.")
        coo_rois = suite2p_rois_to_coo(suite2p_stats, movie_shape)

        binarized_coo_rois = []
        for coo_roi in coo_rois:
            binary_mask = binarize_roi_mask(coo_roi,
                                            self.args['abs_threshold'],
                                            self.args['binary_quantile'])
            binarized_coo_rois.append(binary_mask)
        self.logger.info("Binarized ROIs from Suite2p, total binarized: "
                         f"{len(binarized_coo_rois)}")

        # load the motion correction values
        self.logger.info("Loading motion correction border values from "
                         f" {self.args['motion_correction_values']}")
        motion_correction_df = pd.read_csv(
            self.args['motion_correction_values'])
        motion_border = get_max_correction_values(
            motion_correction_df['x'],
            motion_correction_df['y'],
            self.args['maximum_motion_shift'])

        # create the rois
        self.logger.info("Transforming ROIs to LIMS compatible style.")
        LIMS_compatible_rois = coo_rois_to_lims_compatible(binarized_coo_rois,
                                                           motion_border,
                                                           movie_shape)

        # save the rois as a json file to output directory
        self.logger.info("Writing LIMs compatible ROIs to json file at "
                         f"{self.args['output_json']}")

        out_dict = {
            'LIMS_compatible_rois': LIMS_compatible_rois
        }

        self.output(out_dict,
                    output_path=self.args['output_json'])


if __name__ == '__main__':  # pragma: no cover
    roi_creator_and_binarizer = BinarizerAndROICreator()
    roi_creator_and_binarizer.binarize_and_create()

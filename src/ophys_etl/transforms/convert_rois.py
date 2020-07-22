from pathlib import Path

from argschema import ArgSchema, ArgSchemaParser
from argschema.fields import (List, InputFile, Str, OutputFile, Float,
                              Nested, Int, Bool)
import marshmallow as mm
import numpy as np
import h5py

from ophys_etl.transforms.roi_transforms import (binarize_roi_mask,
                                                 suite2p_rois_to_coo,
                                                 coo_rois_to_old)
from ophys_etl.transforms.data_loaders import get_max_correction_border


class BinarizeAndCreationException(Exception):
    pass


class BinarizeAndCreateROIsInputSchema(ArgSchema):

    suite2p_stat_path = Str(
        required=True,
        description=("Path to s2p output stat file containing ROIs generated "
                     "during source extraction")
    )

    motion_corrected_video = Str(
        required=True,
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
        default=30.0,
        required=False,
        allow_none=False,
        description=("The maximum allowable motion shift for a frame in pixels"
                     " before it is considered an anomaly and thrown out of "
                     "processing")
    )

    abs_threshold = Float(
        default=None,
        required=False,
        allow_none=True,
        description=("The absolute threshold to binarize ROI masks against. "
                     "If not provided will use quantile to generate "
                     "threshold.")
    )

    binary_quantile = Float(
        default=None,
        required=False,
        allow_none=True,
        description=("The quantile against which an ROI is binarized. If not "
                     "provided will use default function value of 0.1.")
    )

    @mm.post_load()
    def check_args(self, data, **kwargs):
        if data['binary_quantile'] is not None:
            if data['binary_quantile'] <= 0:
                raise BinarizeAndCreationException("Binary quantile must be"
                                                   "positive value")
        if data['abs_threshold'] is not None:
            if data['abs_threshold'] <= 0:
                raise BinarizeAndCreationException("Abs threshold must be "
                                                   "positive value")
        return data

    @mm.post_load()
    def check_file_existance(self, data, **kwargs):
        """
        have to check manually for existance of .h5 and .npy files because
        InputFile fails on these, at least on windows because of a use of
        generic open() to check for existance.
        """
        if not Path(data['suite2p_stat_path']).exists():
            raise BinarizeAndCreationException("suite2p_stat_path supplied "
                                               "is not a valid path.")
        if not Path(data['motion_corrected_video']).exists():
            raise BinarizeAndCreationException("motion_corrected_video "
                                               "supplied is not a valid path.")
        return data


class OldSegmentationROISchema(ArgSchema):
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
    mask_image_plane = Int(required=True,
                           description=("What tiff file the ROI lie "
                                        "within. this is being deprecated so "
                                        "will always be assigned 0"))
    exclusion_labels = List(Int, required=True,
                            description=("Codes for reasoning of exclusion of "
                                         "an ROI"))


class BinarizeAndCreateROIsOutputSchema(ArgSchema):
    old_rois = Nested(OldSegmentationROISchema, many=True)


class BinarizerAndROICreator(ArgSchemaParser):
    default_schema = BinarizeAndCreateROIsInputSchema

    def binarize_and_create(self):

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
            Path(self.args['motion_correction_values']),
            self.args['maximum_motion_shift'])

        # create the rois
        self.logger.info("Transforming ROIs to old segmentation style.")
        old_segmentation_rois = coo_rois_to_old(binarized_coo_rois,
                                                motion_border,
                                                movie_shape)

        # save the rois as a json file to output directory
        self.logger.info("Writing old style ROIs to json file at "
                         f"{self.args['output_json']}")

        self.output(old_segmentation_rois,
                    output_path=self.args['output_json'])


if __name__ == '__main__':  # pragma: no cover
    roi_creator_and_binarizer = BinarizerAndROICreator()
    roi_creator_and_binarizer.binarize_and_create()

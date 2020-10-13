import json
from pathlib import Path

import h5py
import numpy as np
from argschema import ArgSchema, ArgSchemaParser
from argschema.fields import Float, InputFile, Int, OutputFile, Str, Bool
from marshmallow import ValidationError
from marshmallow.validate import Range

from ophys_etl.extractors.motion_correction import \
        get_max_correction_from_file
from ophys_etl.schemas import DenseROISchema
from ophys_etl.transforms.roi_transforms import (binarize_roi_mask,
                                                 coo_rois_to_lims_compatible,
                                                 suite2p_rois_to_coo,
                                                 morphological_transform)
from ophys_etl.filters import filter_by_aspect_ratio


class PostProcessException(Exception):
    pass


class PostProcessROIsInputSchema(ArgSchema):
    suite2p_stat_path = Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to s2p output stat file containing ROIs generated "
                     "during source extraction"))
    motion_corrected_video = Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to motion corrected video file *.h5"))
    motion_correction_values = InputFile(
        required=True,
        description=("Path to motion correction values for each frame "
                     "stored in .csv format. This .csv file is expected to"
                     "have a header row of either:\n"
                     "['framenumber','x','y','correlation','kalman_x',"
                     "'kalman_y']\n['framenumber','x','y','correlation',"
                     "'input_x','input_y','kalman_x',"
                     "'kalman_y','algorithm','type']"))
    output_json = OutputFile(
        required=True,
        description=("Path to a file to write output data."))
    maximum_motion_shift = Float(
        missing=30.0,
        required=False,
        allow_none=False,
        description=("The maximum allowable motion shift for a frame in pixels"
                     " before it is considered an anomaly and thrown out of "
                     "processing"))
    abs_threshold = Float(
        missing=None,
        required=False,
        allow_none=True,
        description=("The absolute threshold to binarize ROI masks against. "
                     "If not provided will use quantile to generate "
                     "threshold."))
    binary_quantile = Float(
        missing=0.1,
        validate=Range(min=0, max=1),
        description=("The quantile against which an ROI is binarized. If not "
                     "provided will use default function value of 0.1."))
    npixel_threshold = Int(
        default=40,
        required=False,
        description=("ROIs with fewer pixels than this will be labeled as "
                     "invalid and small size."))
    aspect_ratio_threshold = Float(
        default=0.2,
        required=False,
        description=("ROIs whose aspect ratio is <= this value are "
                     "not recorded. This captures a large majority of "
                     "Suite2P-created artifacts from motion border"))
    morphological_ops = Bool(
        default=True,
        required=False,
        description=("whether to perform morphological operations after "
                     "binarization. ROIs that are washed away to empty "
                     "after this operation are eliminated from the record. "
                     "This can apply to ROIs that were previously labeled "
                     "as small size, for example."))


class PostProcessROIs(ArgSchemaParser):
    default_schema = PostProcessROIsInputSchema

    def run(self):
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
        self.logger.info("Filtering and Binarizing the ROIs created by "
                         "Suite2p.")
        coo_rois = suite2p_rois_to_coo(suite2p_stats, movie_shape)

        # filter raw rois by aspect ratio
        filtered_coo_rois = filter_by_aspect_ratio(
                coo_rois,
                self.args['aspect_ratio_threshold'])
        self.logger.info("Filtered out "
                         f"{len(coo_rois) - len(filtered_coo_rois)} "
                         "ROIs with aspect ratio <= "
                         f"{self.args['aspect_ratio_threshold']}")

        binarized_coo_rois = []
        for filtered_coo_roi in filtered_coo_rois:
            binary_mask = binarize_roi_mask(filtered_coo_roi,
                                            self.args['abs_threshold'],
                                            self.args['binary_quantile'])
            binarized_coo_rois.append(binary_mask)
        self.logger.info("Binarized ROIs from Suite2p, total binarized: "
                         f"{len(binarized_coo_rois)}")

        # load the motion correction values
        self.logger.info("Loading motion correction border values from "
                         f" {self.args['motion_correction_values']}")
        motion_border = get_max_correction_from_file(
            self.args['motion_correction_values'],
            self.args['maximum_motion_shift'])

        # create the rois
        self.logger.info("Transforming ROIs to LIMS compatible style.")
        compatible_rois = coo_rois_to_lims_compatible(
                binarized_coo_rois, motion_border, movie_shape,
                self.args['npixel_threshold'])

        if self.args['morphological_ops']:
            compatible_rois = [morphological_transform(roi, shape=movie_shape)
                               for roi in compatible_rois]
            # eliminate None
            compatible_rois = [roi for roi in compatible_rois if roi]

        # validate ROIs
        errors = DenseROISchema(many=True).validate(compatible_rois)
        if any(errors):
            raise ValidationError(f"Schema validation errors: {errors}")

        # save the rois as a json file to output directory
        self.logger.info("Writing LIMs compatible ROIs to json file at "
                         f"{self.args['output_json']}")

        with open(self.args['output_json'], 'w') as f:
            json.dump(compatible_rois, f, indent=2)


if __name__ == '__main__':  # pragma: no cover
    roi_post = PostProcessROIs()
    roi_post.run()

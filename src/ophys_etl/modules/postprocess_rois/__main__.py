import json
import h5py
import numpy as np
from argschema import ArgSchemaParser
from marshmallow import ValidationError

from ophys_etl.extractors.motion_correction import \
        get_max_correction_from_file
from ophys_etl.schemas import DenseROISchema
from ophys_etl.transforms.roi_transforms import (binarize_roi_mask,
                                                 coo_rois_to_lims_compatible,
                                                 suite2p_rois_to_coo,
                                                 morphological_transform)
from ophys_etl.filters import filter_by_aspect_ratio
from ophys_etl.modules.postprocess_rois.schemas import \
        PostProcessROIsInputSchema


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
            n_rois = len(compatible_rois)
            # eliminate None
            compatible_rois = [roi for roi in compatible_rois if roi]
            n_rois_morphed = len(compatible_rois)
            self.logger.info("morphological transform reduced number of "
                             f"ROIs from {n_rois} to {n_rois_morphed}")

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

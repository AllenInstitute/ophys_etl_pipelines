import argschema
import h5py
import json
from hnccorr import Movie

from ophys_etl.modules.segmentation.modules.schemas import \
        HNCSegmentationWrapperInputSchema
from ophys_etl.modules.segmentation import hnc_segmentation_utils as hsu


class HNCSegmentationWrapper(argschema.ArgSchemaParser):
    default_schema = HNCSegmentationWrapperInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        hnc_segmenter = hsu.hnc_construct(self.args["hnc_args"],
                                          self.args["video_input"])

        with h5py.File(self.args["video_input"], "r") as f:
            data = f["data"][()]
        movie = Movie(name=self.args["experiment_name"], data=data)
        self.logger.info("movie data read in from "
                         f"{self.args['video_input']}")

        hnc_segmenter.segment(movie)

        self.logger.setLevel(self.args["log_level"])
        self.logger.info("segmentation complete")

        segmentations = hnc_segmenter.segmentations_to_list()
        rois = [hsu.hnc_roi_to_extract_roi(s, i + 1)
                for i, s in enumerate(segmentations)]

        with open(self.args["roi_output"], "w") as f:
            json.dump(rois, f, indent=2)
        self.logger.info("segmented ROIs written to "
                         f"{self.args['roi_output']}")


if __name__ == "__main__":
    hseg = HNCSegmentationWrapper()
    hseg.run()

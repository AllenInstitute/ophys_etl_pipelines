import argschema
import h5py
import json
import numpy as np
from hnccorr import Movie

from ophys_etl.modules.segmentation.modules.schemas import \
        HNCSegmentationWrapperInputSchema
from ophys_etl.modules.segmentation.hnc_segmentation_utils import hnc_construct
from ophys_etl.modules.segmentation.qc_utils.roi_utils import \
        hnc_roi_to_extract_roi


class HNCSegmentationWrapper(argschema.ArgSchemaParser):
    default_schema = HNCSegmentationWrapperInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        hnc_segmenter = hnc_construct(self.args["hnc_args"],
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
        rois = [hnc_roi_to_extract_roi(s, i + 1)
                for i, s in enumerate(segmentations)]

        seed_coords = [i["coords"] for i in hnc_segmenter.seeder._seeds]
        seed_values = [i["value"] for i in hnc_segmenter.seeder._seeds]
        seed_excluded = [i["excluded"] for i in hnc_segmenter.seeder._seeds]
        with h5py.File(self.args["seed_output"], "w") as f:
            seeds = f.create_group("seeds")
            seeds.create_dataset("coordinates", data=np.array(seed_coords))
            seeds.create_dataset("values", data=seed_values)
            seeds.create_dataset("excluded", data=seed_excluded)
            f.create_dataset("seed_image", data=hnc_segmenter.seeder._seed_img)
        self.logger.info("seeds written to "
                         f"{self.args['seed_output']}")

        with open(self.args["roi_output"], "w") as f:
            json.dump(rois, f, indent=2)
        self.logger.info("segmented ROIs written to "
                         f"{self.args['roi_output']}")


if __name__ == "__main__":
    hseg = HNCSegmentationWrapper()
    hseg.run()

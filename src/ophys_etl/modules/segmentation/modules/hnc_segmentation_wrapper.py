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
from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder


class HNCSegmentationWrapper(argschema.ArgSchemaParser):
    default_schema = HNCSegmentationWrapperInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        seeder = ImageMetricSeeder(**self.args['seeder_args'])
        hnc_segmenter = hnc_construct(seeder,
                                      self.args["hnc_args"],
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

        fig = Figure(figsize=(8, 8))
        axes = fig.add_subplot(111)
        with h5py.File(self.args["seed_output"], "w") as f:
                with h5py.File(seed_output, "r") as f:
                    add_seeds_to_axes(fig, axes, seed_h5_group=f["seeding"])
                fig.savefig(seed_plot_output)
                logger.info(f'wrote {seed_plot_output}')
            seeds = f.create_group("seeding")
        self.logger.info("seeds written to "
                         f"{self.args['seed_output']}")

        with open(self.args["roi_output"], "w") as f:
            json.dump(rois, f, indent=2)
        self.logger.info("segmented ROIs written to "
                         f"{self.args['roi_output']}")


if __name__ == "__main__":
    hseg = HNCSegmentationWrapper()
    hseg.run()

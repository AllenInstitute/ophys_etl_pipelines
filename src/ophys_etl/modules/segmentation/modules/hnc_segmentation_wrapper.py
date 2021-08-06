import argschema
import h5py
import networkx as nx
from hnccorr import Movie

from ophys_etl.modules.segmentation.modules.schemas import \
        HNCSegmentationWrapperInputSchema
from ophys_etl.modules.segmentation.detect.hnc_segmentation_utils import (
        hnc_construct, ExplicitHNCcorrCandidate, HNCcorrSegmentationObjects)
from ophys_etl.modules.segmentation.qc_utils.roi_utils import \
        hnc_roi_to_extract_roi
from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder
from ophys_etl.modules.segmentation.graph_utils.conversion import graph_to_img
from ophys_etl.modules.segmentation.processing_log import \
        SegmentationProcessingLog


class HNCSegmentationWrapper(argschema.ArgSchemaParser):
    default_schema = HNCSegmentationWrapperInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        # define a seeder
        seeder = ImageMetricSeeder(**self.args['seeder_args'])
        graph_img = graph_to_img(
                nx.read_gpickle(self.args['graph_input']),
                attribute_name=self.args["attribute"])
        seeder.select_seeds(graph_img, sigma=None)
        # define all the HNCcorr repo objects
        hnc_objects: HNCcorrSegmentationObjects = hnc_construct(
                self.args["hnc_args"])
        # read in the movie and turn it into the HNCcorr Movie object
        with h5py.File(self.args["video_input"], "r") as f:
            data = f["data"][()]
        movie = Movie(name=self.args["experiment_name"], data=data)
        self.logger.info("movie data read in from "
                         f"{self.args['video_input']}")

        n_candidates = 0
        segmentations = []
        rois = []
        # the segmentation loop
        for seed in seeder:
            candidate = ExplicitHNCcorrCandidate(seed, hnc_objects)
            n_candidates += 1
            self.logger.info(f"Cells identified: {len(segmentations)}, "
                             f"Next candidate: {n_candidates}")
            best_segmentation = candidate.segment(movie)
            if best_segmentation is not None:
                segmentations.append(best_segmentation)
                seeder.exclude_pixels(best_segmentation.selection)
                rois.append(hnc_roi_to_extract_roi(
                    {"coordinates": list(best_segmentation.selection)},
                    len(rois) + 1))

        self.logger.setLevel(self.args["log_level"])
        self.logger.info("segmentation complete")

        # log detection to hdf5 processing log
        processing_log = SegmentationProcessingLog(path=self.args["log_path"],
                                                   read_only=False)
        processing_log.log_detection(
                attribute=self.args["attribute"].encode("utf-8"),
                rois=rois,
                group_name="detect",
                seeder=seeder,
                seeder_group_name="seed")
        self.logger.info(
            f'logged detection step to {str(self.args["log_path"])}')

        processing_log = SegmentationProcessingLog(path=self.args["log_path"],
                                                   read_only=True)
        # create plots of this detection step
        if self.args["seed_plot_output"] is not None:
            fig = processing_log.create_seeder_figure(
                    group_keys=["detect", "seed"])
            fig.savefig(self.args["seed_plot_output"], dpi=300)
            self.logger.info(f'wrote {self.args["seed_plot_output"]}')

        if self.args["plot_output"] is not None:
            figure = processing_log.create_roi_metric_figure(
                    rois_group="detect",
                    attribute_group="detect",
                    metric_image_group=["detect", "seed"])
            figure.savefig(self.args["plot_output"], dpi=300)
            self.logger.info(f'wrote {self.args["plot_output"]}')


if __name__ == "__main__":
    hseg = HNCSegmentationWrapper()
    hseg.run()

import argschema
import h5py
import json
import networkx as nx
from hnccorr import Movie
from matplotlib.figure import Figure

from ophys_etl.modules.segmentation.modules.schemas import \
        HNCSegmentationWrapperInputSchema
from ophys_etl.modules.segmentation.detect.hnc_segmentation_utils import (
        hnc_construct, ExplicitHNCcorrCandidate, HNCcorrSegmentationObjects)
from ophys_etl.modules.segmentation.qc_utils.roi_utils import \
        hnc_roi_to_extract_roi
from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder
from ophys_etl.modules.segmentation.qc.seed import add_seeds_to_axes
from ophys_etl.modules.segmentation.graph_utils.conversion import graph_to_img
from ophys_etl.modules.segmentation.qc_utils.roi_utils import create_roi_plot


class HNCSegmentationWrapper(argschema.ArgSchemaParser):
    default_schema = HNCSegmentationWrapperInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        # define a seeder
        seeder = ImageMetricSeeder(**self.args['seeder_args'])
        graph_img = graph_to_img(
                nx.read_gpickle(self.args['graph_input']),
                attribute_name=self.args["attribute_name"])
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

        candidates = []
        segmentations = []
        rois = []
        # the segmentation loop
        for seed in seeder:
            candidate = ExplicitHNCcorrCandidate(seed, hnc_objects)
            candidates.append(candidate)
            self.logger.info(f"Cells identified: {len(segmentations)}, "
                             f"Next candidate: {len(candidates)}")
            best_segmentation = candidate.segment(movie)
            if best_segmentation is not None:
                segmentations.append(best_segmentation)
                seeder.exclude_pixels(best_segmentation.selection)
                rois.append(hnc_roi_to_extract_roi(
                    {"coordinates": list(best_segmentation.selection)},
                    len(rois) + 1))

        self.logger.setLevel(self.args["log_level"])
        self.logger.info("segmentation complete")

        if self.args['seed_output'] is not None:
            with h5py.File(self.args['seed_output'], "w") as f:
                seeder.log_to_h5_group(f)
            self.logger.info(f"wrote {self.args['seed_output']}")

            if self.args['seed_plot_output'] is not None:
                fig = Figure(figsize=(8, 8))
                axes = fig.add_subplot(111)
                with h5py.File(self.args['seed_output'], "r") as f:
                    add_seeds_to_axes(fig, axes, seed_h5_group=f["seeding"])
                fig.savefig(self.args['seed_plot_output'])
                self.logger.info(f"wrote {self.args['seed_plot_output']}")

        self.logger.info(f"writing {self.args['roi_output']}")
        with open(self.args['roi_output'], 'w') as out_file:
            out_file.write(json.dumps(rois, indent=2))

        if self.args['plot_output'] is not None:
            create_roi_plot(self.args['plot_output'], graph_img, rois)
            self.logger.info(f"wrote {self.args['plot_output']}")


if __name__ == "__main__":
    hseg = HNCSegmentationWrapper()
    hseg.run()

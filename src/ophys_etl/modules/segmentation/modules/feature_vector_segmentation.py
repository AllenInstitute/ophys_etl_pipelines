import matplotlib
matplotlib.use('Agg')

import argschema
import pathlib

from ophys_etl.modules.segmentation.modules.schemas import \
    FeatureVectorSegmentationInputSchema

from ophys_etl.modules.segmentation.graph_utils.feature_vector_segmentation import (
    FeatureVectorSegmenter)


class FeatureVectorSegmentationRunner(argschema.ArgSchemaParser):

    default_schema = FeatureVectorSegmentationInputSchema

    def run(self):
        graph_input = pathlib.Path(self.args['graph_input'])
        video_input = pathlib.Path(self.args['video_input'])
        n_processors = self.args['n_parallel_workers']
        attr = self.args['attribute']
        segmenter = FeatureVectorSegmenter(graph_input,
                                           video_input,
                                           attribute=attr,
                                           n_processors=n_processors)

        if self.args['plot_output'] is not None:
            plot_output = pathlib.Path(self.args['plot_output'])
        else:
            plot_output=None
        segmenter.run(roi_output=self.args['roi_output'],
                      seed_output=pathlib.Path(self.args['seed_output']),
                      plot_output=plot_path)


if __name__ == "__main__":
    seg = FeatureVectorSegmentationRunner()
    seg.run()

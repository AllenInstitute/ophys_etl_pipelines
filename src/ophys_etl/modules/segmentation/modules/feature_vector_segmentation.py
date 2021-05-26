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
        img_path = pathlib.Path(self.args['graph_input'])
        video_path = pathlib.Path(self.args['video_input'])
        n_processors = self.args['n_parallel_workers']
        attr = self.args['attribute']
        segmenter = FeatureVectorSegmenter(img_path,
                                           video_path,
                                           attribute=attr,
                                           n_processors=n_processors)

        if self.args['plot_output'] is not None:
            plot_path = pathlib.Path(self.args['plot_output'])
        else:
            plot_path=None
        segmenter.run(roi_path=self.args['roi_output'],
                      seed_path=pathlib.Path(self.args['seed_output']),
                      plot_path=plot_path)


if __name__ == "__main__":
    seg = FeatureVectorSegmentationRunner()
    seg.run()

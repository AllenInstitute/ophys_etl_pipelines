import matplotlib
matplotlib.use('Agg')

import argschema
import pathlib

from ophys_etl.modules.segmentation.modules.schemas import \
    HNCSegmentationInputSchema

from ophys_etl.modules.segmentation.graph_utils.hnc_segmentation import (
    HNCSegmenter)


class HNCSegmentationRunner(argschema.ArgSchemaParser):

    default_schema = HNCSegmentationInputSchema

    def run(self):
        img_path = pathlib.Path(self.args['img_path'])
        video_path = pathlib.Path(self.args['video_path'])
        n_processors = self.args['n_parallel_workers']
        attr = self.args['attribute']
        segmenter = HNCSegmenter(img_path,
                                 video_path,
                                 attribute=attr,
                                 n_processors=n_processors)

        if self.args['plot_path'] is not None:
            plot_path = pathlib.Path(self.args['plot_path'])
        else:
            plot_path=None
        segmenter.run(roi_path=self.args['roi_output'],
                      seed_path=pathlib.Path(self.args['seed_path']),
                      plot_path=plot_path)


if __name__ == "__main__":
    seg = HNCSegmentationRunner()
    seg.run()

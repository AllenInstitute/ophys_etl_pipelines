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

        segmenter.run(roi_path=self.args['roi_output'],
                      seed_path_dir=pathlib.Path(self.args['seed_dir']))

if __name__ == "__main__":
    seg = HNCSegmentationRunner()
    seg.run()

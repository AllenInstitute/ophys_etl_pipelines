import argschema
import h5py
import json
import numpy as np
from hnccorr import HNCcorr, HNCcorrConfig, Movie
from typing import TypedDict, List, Tuple

from ophys_etl.modules.segmentation.modules.schemas import \
        HNCSegmentationWrapperInputSchema
from ophys_etl.types import ExtractROI


class HNC_ROI(TypedDict):
    coordinates: List[Tuple[int, int]]


def hnc_roi_to_extract_roi(hnc_roi: HNC_ROI, id: int) -> ExtractROI:
    coords = np.array(hnc_roi["coordinates"])
    y0, x0 = coords.min(axis=0)
    height, width = coords.ptp(axis=0) + 1
    mask = np.zeros(shape=(height, width), dtype=bool)
    for y, x in coords:
        mask[y - y0, x - x0] = True
    roi = ExtractROI(
            id=id,
            x=int(x0),
            y=int(y0),
            width=int(width),
            height=int(height),
            mask=[i.tolist() for i in mask])
    return roi


class HNCSegmentationWrapper(argschema.ArgSchemaParser):
    default_schema = HNCSegmentationWrapperInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        hconfig = HNCcorrConfig(**self.args["hnc_args"])

        with h5py.File(self.args["video_input"], "r") as f:
            data = f["data"][()]
        movie = Movie(name=self.args["experiment_name"], data=data)
        self.logger.info("movie data read in from "
                         f"{self.args['video_input']}")

        hnc_segmenter = HNCcorr.from_config(config=hconfig)
        hnc_segmenter.segment(movie)
        self.logger.info("segmentation complete")

        segmentations = hnc_segmenter.segmentations_to_list()
        rois = [hnc_roi_to_extract_roi(s, i + 1)
                for i, s in enumerate(segmentations)]

        with open(self.args["roi_output"], "w") as f:
            json.dump(rois, f, indent=2)
        self.logger.info("segmented ROIs written to "
                         f"{self.args['roi_output']}")


if __name__ == "__main__":
    hseg = HNCSegmentationWrapper()
    hseg.run()

import argschema
import pathlib

import h5py
import json

from ophys_etl.modules.segmentation.modules.schemas import \
    RoiMergerSchema

import ophys_etl.modules.segmentation.postprocess_utils.roi_merging as merging

import logging
import time


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class RoiMergerEngine(argschema.ArgSchemaParser):

    default_schema = RoiMergerSchema


    def run(self):

        t0 = time.time()
        with open(self.args['roi_input'], 'rb') as in_file:
            raw_roi_list = json.load(in_file)

        roi_list = []
        for roi in raw_roi_list:
            roi_list.append(merging.extract_roi_to_ophys_roi(roi))
        del raw_roi_list

        with h5py.File(self.args['video_input'], 'r') as in_file:
            video_data = in_file['data'][()]

        keep_going = True
        while keep_going:
            n_roi_0 = len(roi_list)

            (keep_going,
             roi_list) = merging.attempt_merger_pixel_correlation(
                                                video_data,
                                                roi_list,
                                                self.args['filter_fraction'],
                                                self.args['n_parallel_workers'])
            n_roi_1 = len(roi_list)
            duration = time.time()-t0
            self.logger.info(f'Merged {n_roi_0} ROIs to {n_roi_1} '
                             f'after {duration:.2f} seconds')

        output_list = []
        for roi in roi_list:
            new_roi = merging.ophys_roi_to_extract_roi(roi)
            output_list.append(new_roi)

        with open(self.args['roi_output'], 'w') as out_file:
            out_file.write(json.dumps(output_list, indent=2))


if __name__ == "__main__":
    merger_engine = RoiMergerEngine()
    merger_engine.run()

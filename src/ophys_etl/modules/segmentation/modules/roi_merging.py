from typing import Union, List
import argschema
import pathlib
import h5py
import json

from ophys_etl.modules.segmentation.modules.schemas import \
    RoiMergerSchema

import ophys_etl.modules.segmentation.merge.roi_merging as merging
import ophys_etl.modules.segmentation.merge.roi_utils as roi_utils

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

import logging
import time


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def write_out_rois(roi_list: List[OphysROI],
                   out_name: Union[str, pathlib.Path]) -> None:
    """
    Write a list of ROIs into the LIMS-friendly JSONized format

    Parameters
    ----------
    roi_list: Union[List[OphysROI], List[OphysROI]]

    out_name: Union[str, pathlib.Path]
        Path to file to be written

    Returns
    -------
    None
    """
    output_list = []
    for roi in roi_list:
        new_roi = roi_utils.ophys_roi_to_extract_roi(roi)
        output_list.append(new_roi)

    with open(out_name, 'w') as out_file:
        out_file.write(json.dumps(output_list, indent=2))

    return None


class RoiMergerEngine(argschema.ArgSchemaParser):

    default_schema = RoiMergerSchema

    def run(self):

        with h5py.File(self.args['video_input'], 'r') as in_file:
            video_data = in_file['data'][()]

        t0 = time.time()
        with open(self.args['roi_input'], 'rb') as in_file:
            raw_roi_list = json.load(in_file)

        roi_list = []
        roi_id_set = set()
        for roi in raw_roi_list:
            roi['valid'] = True
            ophys_roi = roi_utils.extract_roi_to_ophys_roi(roi)
            if ophys_roi.roi_id in roi_id_set:
                raise RuntimeError(f'roi id {ophys_roi.roi_id} duplicated '
                                   'in initial input')
            roi_list.append(ophys_roi)
            roi_id_set.add(ophys_roi.roi_id)
        del raw_roi_list

        roi_list = merging.do_roi_merger(
                                roi_list,
                                video_data,
                                self.args['n_parallel_workers'],
                                self.args['corr_acceptance'])

        write_out_rois(roi_list, self.args['roi_output'])

        duration = time.time()-t0
        self.logger.info(f'Finished in {duration:.2f} seconds')


if __name__ == "__main__":
    merger_engine = RoiMergerEngine()
    merger_engine.run()

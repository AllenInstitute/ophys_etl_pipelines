from typing import Union, List
import argschema
import pathlib
import h5py
import json
import matplotlib
import datetime
import copy
import numpy as np

from ophys_etl.modules.segmentation.modules.schemas import \
    RoiMergerSchema

import ophys_etl.modules.segmentation.merge.roi_merging as merging
import ophys_etl.modules.segmentation.utils.roi_utils as roi_utils
from ophys_etl.modules.segmentation.qc.detect import roi_metric_qc_plot
from ophys_etl.modules.segmentation.qc.merge import roi_merge_plot

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
        roi_list = roi_utils.roi_list_from_file(
                        pathlib.Path(self.args['roi_input']))
        roi_id_set = set([roi.roi_id for roi in roi_list])
        if len(roi_id_set) != len(roi_list):
            raise RuntimeError("There were ROI ID values duplicated in "
                               f"{self.args['roi_input']}")

        original_roi_list = copy.deepcopy(roi_list)

        roi_list, merger_ids = merging.do_roi_merger(
                roi_list,
                video_data,
                self.args['n_parallel_workers'],
                self.args['corr_acceptance'],
                filter_fraction=self.args['filter_fraction'],
                anomalous_size=self.args['anomalous_size'])

        # log merging to hdf5 QC output
        with h5py.File(self.args['qc_output'], "a") as h5file:
            # TODO: merging should output something to QC
            if "merge" in list(h5file.keys()):
                del h5file["merge"]
            group = h5file.create_group("merge")
            group.create_dataset(
                    "group_creation_time",
                    data=str(datetime.datetime.now()).encode("utf-8"))
            group.create_dataset("merger_ids",
                                 data=np.array(merger_ids))
        self.logger.info(f'added group "merge" to {self.args["qc_output"]}')

        write_out_rois(roi_list, self.args['roi_output'])
        self.logger.info(f'wrote {self.args["roi_output"]}')

        if self.args['plot_output'] is not None:
            merged_roi_list = [roi_utils.ophys_roi_to_extract_roi(roi)
                               for roi in roi_list]
            unmerged_roi_list = [roi_utils.ophys_roi_to_extract_roi(roi)
                                 for roi in original_roi_list]
            # TODO: store ROIs in QC output hdf5
            # NOTE: this QC plot is exactly like the one output from detection
            # it should be named something different in the workflow to
            # keep the QC evaluation of detection and merging separate.
            figure = matplotlib.figure.Figure(figsize=(10, 10))
            with h5py.File(self.args['qc_output'], "r") as f:
                merger_ids = f["merge"]["merger_ids"][()]
                if 'detect' not in f:
                    # in production, we will pass the same QC file
                    # between the detect and merge steps, but,
                    # maybe someone will not
                    logger.warn("Unable to create merging plot; "
                                "'detect' group does not exist in "
                                f"{self.args['qc_output']}")
                    metric_image = np.zeros(video_data.shape[1:],
                                            dtype="uint8")
                    attribute = "None"
                else:
                    metric_image = f["detect"]["metric_image"][()]
                    attribute = f["detect"]["attribute"][()].decode("utf-8")

            roi_metric_qc_plot(
                    figure=figure,
                    metric_image=metric_image,
                    attribute=attribute,
                    roi_list=merged_roi_list)
            figure.tight_layout()
            figure.savefig(self.args['plot_output'], dpi=300)
            logger.info(f'wrote {self.args["plot_output"]}')

            # plot the merge
            figure = matplotlib.figure.Figure(figsize=(20, 20))
            roi_merge_plot(figure=figure,
                           metric_image=metric_image,
                           attribute=attribute,
                           original_roi_list=unmerged_roi_list,
                           merged_roi_list=merged_roi_list,
                           merger_ids=merger_ids)
            figure.tight_layout()
            figure.savefig(self.args['merge_plot_output'], dpi=300)
            logger.info(f'wrote {self.args["merge_plot_output"]}')

        duration = time.time()-t0
        self.logger.info(f'Finished in {duration:.2f} seconds')


if __name__ == "__main__":
    merger_engine = RoiMergerEngine()
    merger_engine.run()

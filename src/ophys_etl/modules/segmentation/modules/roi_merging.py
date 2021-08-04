import argschema
import h5py
import numpy as np

from ophys_etl.modules.segmentation.modules.schemas import \
    RoiMergerSchema
import ophys_etl.modules.segmentation.merge.roi_merging as merging
import ophys_etl.modules.segmentation.utils.roi_utils as roi_utils
from ophys_etl.modules.segmentation.processing_log import \
    SegmentationProcessingLog

import logging
import time


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class RoiMergerEngine(argschema.ArgSchemaParser):

    default_schema = RoiMergerSchema

    def run(self):

        with h5py.File(self.args['video_input'], 'r') as in_file:
            video_data = in_file['data'][()]

        t0 = time.time()
        processing_log = SegmentationProcessingLog(self.args["log_path"])
        original_roi_list = processing_log.get_rois_from_group(
                group_name=self.args["rois_group"])

        roi_list = roi_utils.ophys_roi_list_from_deserialized(
                original_roi_list)

        roi_id_set = set([roi.roi_id for roi in roi_list])
        if len(roi_id_set) != len(roi_list):
            raise RuntimeError("There were ROI ID values duplicated in "
                               f"{self.args['roi_input']}")

        roi_list, merger_ids = merging.do_roi_merger(
                roi_list,
                video_data,
                self.args['n_parallel_workers'],
                self.args['corr_acceptance'],
                filter_fraction=self.args['filter_fraction'],
                anomalous_size=self.args['anomalous_size'])

        merged_roi_list = [roi_utils.ophys_roi_to_extract_roi(i)
                           for i in roi_list]

        # log merging to hdf5 QC output
        processing_log.log_merge(rois=merged_roi_list,
                                 merger_ids=np.array(merger_ids),
                                 group_name="merge")
        self.logger.info(f'added group {processing_log.get_last_group()} '
                         f'to {processing_log.path}')

        if self.args['plot_output'] is not None:
            # NOTE: this QC plot is exactly like the one output from detection
            # it should be named something different in the workflow to
            # keep the QC evaluation of detection and merging separate.
            figure = processing_log.create_roi_metric_figure(
                    rois_group="merge",
                    attribute_group="detect",
                    metric_image_group=["detect", "seed"])
            figure.savefig(self.args['plot_output'], dpi=300)
            logger.info(f'wrote {self.args["plot_output"]}')

            # plot the merge
            figure = processing_log.create_roi_merge_figure(
                    original_rois_group="detect",
                    merged_rois_group="merge",
                    attribute_group="detect",
                    metric_image_group=["detect", "seed"])
            figure.savefig(self.args['merge_plot_output'], dpi=300)
            logger.info(f'wrote {self.args["merge_plot_output"]}')

        duration = time.time()-t0
        self.logger.info(f'Finished in {duration:.2f} seconds')


if __name__ == "__main__":
    merger_engine = RoiMergerEngine()
    merger_engine.run()

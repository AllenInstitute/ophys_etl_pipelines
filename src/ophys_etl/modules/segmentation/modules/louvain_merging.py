import argschema

import pathlib
import h5py
import numpy as np
import time

from ophys_etl.modules.segmentation.modules.schemas import (
    LouvainRoiMergerSchema)

from ophys_etl.modules.segmentation.processing_log import (
    SegmentationProcessingLog)
from ophys_etl.modules.segmentation.merge.louvain_utils import (
    find_roi_clusters)
from ophys_etl.modules.segmentation.merge.louvain_merging import (
    do_louvain_clustering_on_rois)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    extract_roi_to_ophys_roi,
    ophys_roi_to_extract_roi)

import logging


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class LouvainMergerEngine(argschema.ArgSchemaParser):

    default_schema = LouvainRoiMergerSchema

    def run(self):
        merger_start_time = time.time()
        processing_log = SegmentationProcessingLog(
                            self.args['log_path'],
                            read_only=True)

        raw_roi_list = processing_log.get_rois_from_group(
                            group_name=self.args['rois_group'])

        input_roi_list = [extract_roi_to_ophys_roi(roi) for roi in
                          raw_roi_list]

        t0 = time.time()
        roi_clusters = find_roi_clusters(input_roi_list)
        duration = time.time()-t0
        self.logger.info(f'finding clusters took {duration:.2f} seconds')

        with h5py.File(self.args['video_input'], 'r') as in_file:
            full_video = in_file['data'][()]

        merged_roi_list = []
        full_history = []
        n_clusters = len(roi_clusters)
        for i_cluster, cluster in enumerate(roi_clusters):
            if len(cluster) == 1:
                merged_roi_list.append(ophys_roi_to_extract_roi(cluster[0]))
                continue
            area = 0
            for roi in cluster:
                area += roi.area
            t0 = time.time()
            self.logger.info(f'starting cluster {i_cluster} of {n_clusters} '
                             f'with area {area:.2e} -- {len(cluster)} ROIs')
            (new_cluster,
             this_history) = do_louvain_clustering_on_rois(
                                 cluster,
                                 full_video,
                                 self.args['kernel_size'],
                                 self.args['filter_fraction'],
                                 self.args['n_parallel_workers'],
                                 pathlib.Path(self.args['scratch_dir']),
                                 only_neighbors=True)

            for pair in this_history:
                if pair[0] != pair[1]:
                    full_history.append(pair)

            duration = time.time()-t0
            self.logger.info(f'done in {duration:.2e} seconds -- '
                             f'kept {len(new_cluster)} ROIs')
            for roi in new_cluster:
                merged_roi_list.append(ophys_roi_to_extract_roi(roi))

        processing_log = SegmentationProcessingLog(
                              self.args['log_path'],
                              read_only=False)

        processing_log.log_merge(rois=merged_roi_list,
                                 roi_source_group=self.args['rois_group'],
                                 merger_ids=np.array(full_history),
                                 group_name="merge (louvain)")

        if self.args['plot_output'] is not None:
            # NOTE: this QC plot is exactly like the one output from detection
            # it should be named something different in the workflow to
            # keep the QC evaluation of detection and merging separate.
            figure = processing_log.create_roi_metric_figure(
                        rois_group="merge (louvain)",
                        attribute_group="detect",
                        metric_image_group=["detect", "seed"])
            figure.savefig(self.args['plot_output'], dpi=300)
            logger.info(f'wrote {self.args["plot_output"]}')

        if self.args['merge_plot_output'] is not None:
            # plot the merge
            figure = processing_log.create_roi_merge_figure(
                        original_rois_group="detect",
                        merged_rois_group="merge (louvain)",
                        attribute_group="detect",
                        metric_image_group=["detect", "seed"])
            figure.savefig(self.args['merge_plot_output'], dpi=300)
            logger.info(f'wrote {self.args["merge_plot_output"]}')

        duration = time.time()-merger_start_time
        self.logger.info(f'Finished in {duration:.2f} seconds')


if __name__ == "__main__":
    merger_engine = LouvainMergerEngine()
    merger_engine.run()

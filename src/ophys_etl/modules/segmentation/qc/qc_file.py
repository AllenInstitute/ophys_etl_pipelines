import h5py
import logging
import datetime
import matplotlib
import numpy as np
from pathlib import Path
from typing import List

from ophys_etl.modules.segmentation.utils import roi_utils
from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder
from ophys_etl.modules.segmentation.qc.seed import add_seeds_to_axes
from ophys_etl.modules.segmentation.qc.detect import roi_metric_qc_plot
from ophys_etl.modules.segmentation.qc.merge import roi_merge_plot

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def timestamp_group(group: h5py.Group) -> None:
    """adds a 'group_creation_time' utf-8 encoded timestamp to group

    Parameters
    ----------
    group: h5py.Group
        the group to timestamp

    """
    group.create_dataset(
            "group_creation_time",
            data=str(datetime.datetime.now()).encode("utf-8"))


class SegmentationQCFile:
    """I/O methods for segmentation quality-control (log) file
    """

    def __init__(self, path: Path):
        self.path = path
        self.steps_dataset_name = "processing_steps"

    def append_processing_step(self,
                               group_name: str,
                               overwrite: bool = False) -> str:
        with h5py.File(self.path, "a") as f:
            steps = []
            if self.steps_dataset_name in f:
                if overwrite:
                    logger.warn(
                            f"overwriting dataset {self.steps_dataset_name}")
                else:
                    steps = [i.decode("utf-8")
                             for i in f[self.steps_dataset_name][()]]
                del f[self.steps_dataset_name]
            # get unique group name, if necessary. e.g. multiple
            # steps called 'filter'
            increment = 0
            for step in steps:
                if step.startswith(group_name):
                    increment += 1
            if increment != 0:
                logger.warning(f"{group_name} already exists in log")
                group_name = f"{group_name}_{increment}"
                logger.warning(f"created new group {group_name}")
            steps.append(group_name)
            f.create_dataset(self.steps_dataset_name,
                             data=[i.encode("utf-8") for i in steps])
        return group_name

    def get_last_group(self) -> str:
        with h5py.File(self.path, "r") as f:
            steps = [i.decode("utf-8")
                     for i in f[self.steps_dataset_name][()]]
        return steps[-1]

    def log_detection(self,
                      attribute: str,
                      rois: List[ExtractROI],
                      group_name: str = "detect") -> None:
        """log the detection phase of segmentation to a file

        Parameters
        ----------
        attribute: str
            the name of the edge attribute extracted from the
            input graph
        rois: List[ExtractROI]
            the list of ROIs resulting from detection
        group_name: str
            the name of the hdf5 group for logging this step
            (default = 'detect')

        """
        group_name = self.append_processing_step(group_name, overwrite=True)
        with h5py.File(self.path, "a") as h5file:
            if group_name in h5file:
                # only one detect step per QC file
                del h5file[group_name]
                logging.warn(f"overwriting existing {group_name} group in "
                             f"{self.path}")
            group = h5file.create_group(group_name)
            group.create_dataset("attribute", data=attribute)
            group.create_dataset("rois",
                                 data=roi_utils.serialize_extract_roi_list(
                                     rois))
            timestamp_group(group)

    def log_merge(self,
                  rois: List[ExtractROI],
                  merger_ids: np.ndarray,
                  group_name: str = "merge") -> None:
        """log the merge phase of segmentation to a file

        Parameters
        ----------
        rois: List[ExtractROI]
            the list of ROIs resulting from detection
        group_name: str
            the name of the hdf5 group for logging this step
            (default = 'detect')
        merger_ids: np.ndarray
            an n_merge x 2 shape array, each row indicating (dst, src)
            ROI IDs from a merge

        """
        group_name = self.append_processing_step(group_name, overwrite=False)
        with h5py.File(self.path, "a") as h5file:
            group = h5file.create_group(group_name)
            group.create_dataset("rois",
                                 data=roi_utils.serialize_extract_roi_list(
                                     rois))
            group.create_dataset("merger_ids", data=merger_ids)
            timestamp_group(group)

    def log_seeder(self,
                   seeder: ImageMetricSeeder,
                   parent_group: str = "detect",
                   group_name: str = "seed") -> None:
        """log seeder information to a new subgroup, usually a subgroup
        of 'detect' group

        Parameters
        ----------
        seeder: ImageMetricSeeder
            a seeder instance, typically one used during a detection step
        parent_group: h5py.Group
            the hdf5 group within which to create the seeder log
        group_name: str
            name of the new subgroup

        """
        (provided_seeds,
         excluded_seeds,
         exclusion_reason,
         seed_image) = seeder.get_logged_values()

        with h5py.File(self.path, "a") as f:
            parent_group = f[parent_group]
            if group_name in parent_group:
                raise ValueError(f"group {parent_group} in {self.path} "
                                 f"already has a group named {group_name}")
            group = parent_group.create_group(group_name)
            group.create_dataset("provided_seeds", data=provided_seeds)
            group.create_dataset("excluded_seeds", data=excluded_seeds)
            group.create_dataset("exclusion_reason", data=exclusion_reason)
            group.create_dataset("seed_image", data=seed_image)
            timestamp_group(group)

    def create_seeder_figure(self,
                             group_keys: List[str] = ["detect", "seed"]
                             ) -> matplotlib.figure.Figure:
        """return a figure showing seeds

        Parameters
        ----------
        group_keys: List[str]
            a list of keys indicating the hierarchical path to the seed
            log group. I.e. ['a','b'] will look in the h5py.File['a']['b']
            group

        Returns
        -------
        fig: matplotlib.figure.Figure
            a figure showing seeds

        """
        fig = matplotlib.figure.Figure(figsize=(8, 8))
        axes = fig.add_subplot(111)
        with h5py.File(self.path, "r") as group:
            for k in group_keys:
                group = group[k]
            add_seeds_to_axes(fig, axes, seed_h5_group=group)
        return fig

    def create_roi_metric_figure(
            self,
            rois_group: str = "detect",
            attribute_group: str = "detect",
            metric_image_group: List[str] = ["detect", "seed"]
            ) -> matplotlib.figure.Figure:
        """return a figure showing ROIs

        Parameters
        ----------
        rois_group: str
            the h5py group within the file containing dataset 'rois', i.e.
            'detect', 'merge', etc.
        attribute_group: str
            the h5py group within the file containging dataset 'attribute',
            typically 'detect'
        metric_image_group: List[str]
            a list of keys indicating the hierarchical path to the seed
            log group. I.e. ['a','b'] will look in the h5py.File['a']['b']
            group

        Returns
        -------
        fig: matplotlib.figure.Figure
            a figure showing ROIs and stats

        """
        rois = self.get_rois_from_group(group_name=rois_group)
        with h5py.File(self.path, "r") as group:
            attribute = \
                group[attribute_group]["attribute"][()].decode("utf-8")
            for k in metric_image_group:
                group = group[k]
            metric_image = group["seed_image"][()]
        figure = matplotlib.figure.Figure(figsize=(15, 15))
        roi_metric_qc_plot(figure=figure,
                           metric_image=metric_image,
                           attribute=attribute,
                           roi_list=rois)
        figure.tight_layout()
        return figure

    def create_roi_merge_figure(
            self,
            original_rois_group: str = "detect",
            merged_rois_group: str = "merge",
            attribute_group: str = "detect",
            metric_image_group: List[str] = ["detect", "seed"]
            ) -> matplotlib.figure.Figure:
        original_rois = self.get_rois_from_group(
                group_name=original_rois_group)
        merged_rois = self.get_rois_from_group(
                group_name=merged_rois_group)
        with h5py.File(self.path, "r") as group:
            merger_ids = group[merged_rois_group]["merger_ids"][()]
            attribute = \
                group[attribute_group]["attribute"][()].decode("utf-8")
            for k in metric_image_group:
                group = group[k]
            metric_image = group["seed_image"][()]

        figure = matplotlib.figure.Figure(figsize=(20, 20))
        roi_merge_plot(figure=figure,
                       metric_image=metric_image,
                       attribute=attribute,
                       original_roi_list=original_rois,
                       merged_roi_list=merged_rois,
                       merger_ids=merger_ids)
        figure.tight_layout()
        return figure

    def get_rois_from_group(self,
                            group_name: str,
                            dataset_name: str = "rois") -> List[ExtractROI]:
        """read and deserialize ROIs from a group

        Parameters
        ----------
        group_name: str
            the name of the group, i.e. 'detect', 'merge', etc.
        dataset_name: str
            the name of the dataset to read and deserialize,
            typically 'rois'

        Returns
        -------
        rois: List[ExtractROI]
            the deserialized list of ROIs

        """
        with h5py.File(self.path, "r") as f:
            rois = roi_utils.deserialize_extract_roi_list(
                    f[group_name][dataset_name][()])
        return rois

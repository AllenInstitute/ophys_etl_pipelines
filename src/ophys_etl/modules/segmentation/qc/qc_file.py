import h5py
import logging
import datetime
import matplotlib
from pathlib import Path
from typing import List

from ophys_etl.modules.segmentation.utils import roi_utils
from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder
from ophys_etl.modules.segmentation.qc.seed import add_seeds_to_axes
from ophys_etl.modules.segmentation.qc.detect import roi_metric_qc_plot

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

    def log_detection(self,
                      attribute: str,
                      rois: List[ExtractROI],
                      seeder: ImageMetricSeeder,
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
        seeder: ImageMetricSeeder
            the seeder used by the detection step

        """
        with h5py.File(self.path, "a") as h5file:
            if "detect" in list(h5file.keys()):
                # only one detect step per QC file
                del h5file["detect"]
                logging.warn("overwriting existing 'detect' group in "
                             f"{self.path}")
            group = h5file.create_group("detect")
            group.create_dataset("attribute", data=attribute)
            group.create_dataset("rois",
                                 data=roi_utils.serialize_extract_roi_list(
                                     rois))
            self.log_seeder(seeder, parent_group=group)
            timestamp_group(group)

    def log_seeder(self,
                   seeder: ImageMetricSeeder,
                   parent_group: h5py.Group,
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

        if group_name in parent_group:
            raise ValueError(f"group {parent_group} in {self.path} "
                             f"already has a group named {group_name}")

        (provided_seeds,
         excluded_seeds,
         exclusion_reason,
         seed_image) = seeder.get_logged_values()

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

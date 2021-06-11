import json
import h5py
import numpy as np
import tempfile
from typing import TypedDict, List, Tuple
from pathlib import Path
import networkx as nx
import hnccorr.base as hncbase
from hnccorr.utils import eight_neighborhood, add_offset_set_coordinates
from scipy.ndimage import gaussian_filter

from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation.modules.schemas import HNC_args
from ophys_etl.modules.segmentation.modules.calculate_edges import \
        CalculateEdges
from ophys_etl.modules.segmentation.graph_utils.plotting import \
        graph_to_img, add_rois_to_axes


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
            valid=True,
            mask=[i.tolist() for i in mask])
    return roi


def plot_seeds_and_rois(axes, seed_h5_path, rois_path):
    with open("sub64_rois.json", "r") as f:
        rois = json.load(f)

    with h5py.File("./sub64_seeds.h5", "r") as f:
        seeds = f["seeds"]
        s_coords = seeds["coordinates"][()]
        s_excluded = seeds["excluded"][()]
        seed_img = f["seed_image"][()]

    axes.imshow(seed_img, cmap="gray")
    for c, e in zip(s_coords, s_excluded):
        marker = "o"
        if e:
            marker = "x"
        axes.plot(c[1], c[0], marker=marker, color="b")
    add_rois_to_axes(axes, rois, seed_img.shape)


class AllenLocalCorrelationSeeder:
    """follows the style and interfaces of HNCcorr LocalCorrelationSeeder
    """
    def __init__(self,
                 seeder_mask_size,
                 percentage_of_seeds,
                 seeder_exclusion_padding,
                 seeder_grid_size,
                 video_path):
        self.seeder_mask_size = seeder_mask_size
        self.percentage_of_seeds = percentage_of_seeds
        self.seeder_exclusion_padding = seeder_exclusion_padding
        self.seeder_grid_size = seeder_grid_size
        self.video_path = video_path
        self._seeds = None
        self._current_index = None
        self._excluded_pixels = set()

    def select_seeds(self, movie):
        """sets self._seeds
        """
        # create a graph and then an image/array based on attribute
        attribute = "filtered_hnc_Gaussian"
        graph_file = tempfile.NamedTemporaryFile(suffix=".pkl")
        args = {
                "video_path": str(self.video_path),
                "graph_output": graph_file.name,
                "attribute": attribute,
                "filter_fraction": 0.2,
                "neighborhood_radius": 7,
                "n_parallel_workers": 4}
        ce = CalculateEdges(input_data=args, args=[])
        ce.run()
        g = nx.read_gpickle(graph_file.name)
        graph_file.close()
        img = graph_to_img(g, attribute_name=attribute)
        img = gaussian_filter(img, mode="constant", sigma=3)

        row = 0
        seeds = dict()
        while row < img.shape[0]:
            col = 0
            while col < img.shape[1]:
                block = img[row: (row + self.seeder_grid_size),
                            col: (col + self.seeder_grid_size)]
                inds = np.array(
                        np.unravel_index(np.argmax(block), block.shape))
                inds += np.array([row, col])
                seeds[tuple(inds)] = block.max()
                col += self.seeder_grid_size
            row += self.seeder_grid_size

        best_per_grid_block_sorted = sorted(
            [(key, val) for key, val in seeds.items()],
            key=lambda x: x[1],
            reverse=True)

        num_keep = int(self.percentage_of_seeds *
                       len(best_per_grid_block_sorted))

        # store best seeds
        self._seeds = [{"coords": (int(seed[0]), int(seed[1])), "value": val}
                       for seed, val in best_per_grid_block_sorted[:num_keep]]
        self._seed_img = img
        self.reset()

    def reset(self):
        """Reinitialize the sequence of seed pixels and empties
        `_excluded_seeds`.
        """
        self._current_index = 0
        self._excluded_pixels = set()

    def next(self):
        """Provides next seed pixel for segmentation.
        Returns the movie coordinates of the next available seed pixel for
        segmentation. Seed pixels that have previously been excluded will
        be ignored.
        Returns None when all seeds are exhausted.
        Returns:
            tuple or None: Coordinates of next seed pixel.
            None if no seeds remaining.
        """
        while self._current_index < len(self._seeds):
            center_seed = self._seeds[self._current_index]["coords"]
            self._current_index += 1

            if center_seed in self._excluded_pixels:
                self._seeds[self._current_index - 1]["excluded"] = True
            else:
                self._seeds[self._current_index - 1]["excluded"] = False
                return center_seed
        return None

    def exclude_pixels(self, pixels):
        """Excludes pixels from being returned by `next()` method.
        All pixels within in the set `pixels` as well as pixels that are
        within an L-
        infinity distance of `_padding` from any excluded pixel are excluded
        as seeds.
        Method enables exclusion of pixels in previously segmented cells
        from serving
        as new seeds. This may help to prevent repeated segmentation of the
        cell.
        Args:
            pixels (set): Set of pixel coordinates to exclude.
        Returns:
            None
        """
        neighborhood = eight_neighborhood(2, self.seeder_exclusion_padding)

        padded_pixel_sets = [
            add_offset_set_coordinates(neighborhood, pixel) for pixel in pixels
        ]

        self._excluded_pixels = self._excluded_pixels.union(
            pixels.union(*padded_pixel_sets)
        )


def hnc_construct(hnc_args: dict, video_path: Path):
    """
    construct an  HNCcorr object given parameters

    Parameters
    ----------
    hnc_args: dict
        will be validated by HNC_args schema
    """

    # validate against schema
    args = HNC_args().load(data=hnc_args)

    # turn into an object
    config = hncbase.HNCcorrConfig(**args)

    # construct just as in HNCcorr.from_config() but with our own
    # seeder object
    edge_selector = hncbase.SparseComputationEmbeddingWrapper(
            config.sparse_computation_dimension,
            config.sparse_computation_grid_distance
        )
    hnc_obj = hncbase.HNCcorr(
            # hncbase.LocalCorrelationSeeder(
            #     config.seeder_mask_size,
            #     config.percentage_of_seeds,
            #     config.seeder_exclusion_padding,
            #     config.seeder_grid_size
            #     ),
            AllenLocalCorrelationSeeder(
                config.seeder_mask_size,
                config.percentage_of_seeds,
                config.seeder_exclusion_padding,
                config.seeder_grid_size,
                video_path,
            ),
            hncbase.SizePostprocessor(
                config.postprocessor_min_cell_size,
                config.postprocessor_max_cell_size,
                config.postprocessor_preferred_cell_size,
            ),
            hncbase.HncParametricWrapper(0, 1),
            hncbase.PositiveSeedSelector(config.positive_seed_radius),
            hncbase.NegativeSeedSelector(
                config.negative_seed_circle_radius,
                config.negative_seed_circle_count
            ),
            hncbase.GraphConstructor(
                edge_selector,
                lambda a, b: hncbase.exponential_distance_decay(
                    a, b, config.gaussian_similarity_alpha
                ),
            ),
            hncbase.Candidate,
            hncbase.Patch,
            hncbase.CorrelationEmbedding,
            config.patch_size)

    return hnc_obj

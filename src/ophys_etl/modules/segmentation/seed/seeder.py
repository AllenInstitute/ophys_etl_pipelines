import h5py
import numpy as np
from typing import Tuple, Optional, List, Set
from collections.abc import Iterator
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from ophys_etl.modules.segmentation.seed.utils import (Seed,
                                                       dilated_coordinates)


class SeederBase(Iterator):
    """generic base class for seeding, and supports HNCcorr style of seeder

    Parameters
    ----------
    fov_shape: Tuple[int, int]
        the (nrows, ncols) shape of the field-of-view. Used by
        'dilated_coordinates()` to restrict excluded coordinates to
        a certain FOV
    exclusion_buffer: int
        the number of pixels beyond an ROI (by binary dilation) for ignoring
        subsequent seeds.

    Attributes
    ----------
    _candidate_seeds: List[Seed]:
        the list of seeds created by 'select_seeds'
    _provided_seeds: List[Seed]:
        the list of seeds that have been served by the iterator
    _excluded_seeds: List[Seed]:
        the list of candidate seeds that were not served by the iterator
    _excluded_pixels: Set[Tuple[Int, Int]]
        list of current excluded pixels, excluded by 'exclude_pixels()'
    _exclusion_buffer: int
        see Parameter 'exclusion_buffer'

    Notes
    -----
    The expected use of this class's children:
    ```
    seeder = SeederChildClass(*args)
    segmenter = SegmentClass(seeder)

    # segmenter uses seeder as iterator,updating exclusions
    segmenter.run()

    with h5py.File("segmentation_log.h5", "w") as f:
        segmenter.log_to_h5_group(f, "segmenting")

        # log seeder after segmenter has finished using it
        seeder.log_to_h5_group(f, "seeding")
    ```

    """
    def __init__(self,
                 exclusion_buffer: int = 1):
        self._candidate_seeds: List[Seed] = []
        self._provided_seeds: List[Seed] = []
        self._excluded_seeds: List[Seed] = []
        self._exclusion_buffer = exclusion_buffer
        self._excluded_pixels: Set[Tuple[int, int]] = set()

    def exclude_pixels(self, pixels):
        """applies dilation with a 3x3 (default) square structuring element
        and adds new pixels to self._excluded_pixels

        Parameters
        ----------
        pixels (set): Set of pixel coordinates to exclude.

        Returns
        -------
        None

        """
        dilated_pixels = dilated_coordinates(pixels,
                                             self._exclusion_buffer)
        self._excluded_pixels.update(dilated_pixels)

    def __next__(self):
        """returns the next valid seed

        Notes
        -----
        A seed can be invalid either because it was too low on the list of
        seeds sorted by metric (excluded_by_fraction) or if it was within
        a buffer of an ROI and is in _excluded_pixels (excluded_by_roi)

        """
        if len(self._candidate_seeds) == 0:
            raise StopIteration

        next_seed = self._candidate_seeds.pop(0)

        if next_seed['coordinates'] in self._excluded_pixels:
            next_seed['exclusion_reason'] = "excluded by ROI"
            self._excluded_seeds.append(next_seed)
            return self.__next__()

        self._provided_seeds.append(next_seed)
        return next_seed['coordinates']

    def select_seeds(self):
        """method for selecting seeds. This method should be implemented
        by subclasses.

        Notes
        -----
        implementations of this method should populate `self._seeds`
        as List[Seed]
        """
        raise NotImplementedError(f"class {type(self).__name__} does not "
                                  "implement the method 'select_seeds()'")


class ImageMetricSeeder(SeederBase):
    """Finds seeds based on an image input and non-overlapping blocks
    of that image

    Parameters
    ----------
    keep_fraction: float
        when sorted by the image pixel values, the top keep_fraction
        seeds will be valid and the bottom (1.0 - keep_fraction) will
        be marked as 'excluded_by_fraction'
    seeder_grid_size: int
        this follow the HNCcorr method of considering candidate seeds
        from every non-overlapping seeder_grid_size x seeder_grid_size
        subimage

    Attributes
    ----------
    _keep_fraction
        see Parameter 'keep_fraction'
    _seeder_grid_size
        see Parameter 'seeder_grid_size'
    _seed_image
        the input image to 'select_seeds' which is stored for access
        by 'log_to_h5_group()'

    Notes
    -----
    This class is largely inspired by HNCcorr and is written to be
    usable with little modification by that algorithm.
    https://github.com/hochbaumGroup/HNCcorr
    Inheriting from 'Iterator' directly, adding attributes, a logging method,
    and operating on any input image (separating out the computation of that
    image) are the notable changes from that inspiration.

    """
    def __init__(self,
                 keep_fraction: float = 0.4,
                 seeder_grid_size: Optional[int] = None,
                 **kwargs):
        self._seeder_grid_size = seeder_grid_size
        self._keep_fraction = keep_fraction
        self._seed_image: np.ndarray = None
        super(ImageMetricSeeder, self).__init__(**kwargs)

    def select_seeds(self, image: np.ndarray, sigma: Optional[float] = None):
        """select seeds from an image

        Parameters
        ----------
        image: np.ndarray
            an image whose intensity values are the metrics for determining
            seeds
        sigma: float
            if provided, will be the Gaussian filter sigma for a smoothing
            operation on the image before any additional processing.

        """
        if sigma is not None:
            image = gaussian_filter(image, mode="constant", sigma=sigma)
        self._seed_image = image

        seeds = []

        if self._seeder_grid_size is None:
            # every pixel in image is used
            rows, cols = np.mgrid[0: image.shape[0], 0: image.shape[1]]
            for r, c, v in zip(rows.flat, cols.flat, image.flat):
                seeds.append(
                        Seed(coordinates=(int(r), int(c)),
                             value=v,
                             excluded_by_roi=False,
                             excluded_by_quantile=False))
        else:
            # only max value pixels in blockwise neighborhoods (HNCcorr-like)
            row = 0
            while row < image.shape[0]:
                col = 0
                while col < image.shape[1]:
                    block = image[row: (row + self._seeder_grid_size),
                                  col: (col + self._seeder_grid_size)]
                    inds = np.array(
                            np.unravel_index(np.argmax(block), block.shape))
                    inds += np.array([row, col])
                    inds = tuple([int(i) for i in inds])
                    seeds.append(
                            Seed(coordinates=inds,
                                 value=block.max(),
                                 excluded_by_roi=False,
                                 excluded_by_quantile=False))
                    col += self._seeder_grid_size
                row += self._seeder_grid_size

        seeds_sorted = sorted(seeds,
                              key=lambda x: x['value'],
                              reverse=True)
        num_keep = int(self._keep_fraction * len(seeds_sorted))
        self._candidate_seeds = seeds_sorted[:num_keep]
        self._excluded_seeds = seeds_sorted[num_keep:]
        for i in range(len(self._excluded_seeds)):
            self._excluded_seeds[i]["exclusion_reason"] = \
                    "excluded by fraction"

    def log_to_h5_group(self,
                        h5file: h5py.File,
                        group_name: str = "seeding"):
        """records some attributes of this seeder to an hdf5 group for
        inspection purposes.

        Parameters
        ----------
        h5file: h5py.File
            an h5py.File object which has been opened in a writable mode
        group_name: str
            the name of the group to which this object's logged attrbiutes
            will be inserted

        """
        # collect any candidates not even considered (perhaps a segmenter
        # can impose a stopping condition different from running out of seeds
        for seed in self._candidate_seeds:
            seed["exclusion_reason"] = "never considered"
            self._excluded_seeds.append(seed)

        group = h5file.create_group(group_name)
        group.create_dataset(
                "provided_seeds",
                data=np.array([i['coordinates']
                               for i in self._provided_seeds]))
        group.create_dataset(
                "excluded_seeds",
                data=np.array([i['coordinates']
                               for i in self._excluded_seeds]))
        group.create_dataset(
                "exclusion_reason",
                data=np.array([i['exclusion_reason'].encode("utf-8")
                               for i in self._excluded_seeds]))
        group.create_dataset(
                "seed_image",
                data=self._seed_image)


class BatchImageMetricSeeder(ImageMetricSeeder):
    """Can serve up to 'n_samples' size batches of seeds that exceed a
    minimum distance criteria. This is useful for sending to parallel
    jobs where the potential exclusion zones are unlikely reach the other
    the other seeds.

    Parameters
    ----------
    n_samples: int
        the maximum number of seeds to provided per __next__() call.
    minimum_distance: float
        the minimum euclidean distance to allow in a single batch

    """
    def __init__(self, n_samples: int,
                 minimum_distance: float,
                 **kwargs):
        self._n_samples = n_samples
        self._minimum_distance = minimum_distance
        super(BatchImageMetricSeeder, self).__init__(**kwargs)

    def __next__(self):
        """returns the next 'n_samples' valid seeds, subject to minimum
        distance constraint.

        """
        if len(self._candidate_seeds) == 0:
            raise StopIteration

        next_seeds = []
        i = 0
        while len(next_seeds) < self._n_samples:
            if i > (len(self._candidate_seeds) - 1):
                # ran out of seeds before hitting 'n_samples'
                break

            candidate = self._candidate_seeds[i]

            if candidate['coordinates'] in self._excluded_pixels:
                # move candidate to excluded
                candidate['exclusion_reason'] = "excluded by ROI"
                self._excluded_seeds.append(self._candidate_seeds.pop(i))
                continue

            coords = np.array(candidate['coordinates'])
            next_coords = [np.array(i['coordinates']) for i in next_seeds]
            if len(next_coords) == 0:
                # first entry in next_seeds
                next_seeds.append(self._candidate_seeds.pop(i))
                continue

            # 1 or more seeds already
            dists = cdist([coords], next_coords, metric="euclidean")[0]
            if dists.min() >= self._minimum_distance:
                next_seeds.append(self._candidate_seeds.pop(i))
            else:
                # move down the list
                i += 1

        self._provided_seeds.extend(next_seeds)
        return [i['coordinates'] for i in next_seeds]

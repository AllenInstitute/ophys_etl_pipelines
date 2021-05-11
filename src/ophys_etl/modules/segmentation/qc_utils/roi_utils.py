import matplotlib
from typing import List, Tuple, Callable, Optional
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI,
    OphysMovie,
    find_overlapping_roi_pairs)


def add_roi_boundaries_to_img(img: np.ndarray,
                              roi_list: List[OphysROI],
                              color: Tuple[int] = (255, 0, 0),
                              alpha: float = 0.25) -> np.ndarray:
    """
    Add colored ROI boundaries to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi_list: List[OphysROI]
        list of ROIs to add to image

    color: Tuple[int]
        color of ROI border as RGB tuple (default: (255, 0, 0))

    alpha: float
        transparency factor to apply to ROI (default=0.25)

    Returns
    -------
    new_img: np.ndarray
        New image with ROI borders superimposed
    """

    new_img = np.copy(img)
    for roi in roi_list:
        bdry = roi.boundary_mask
        for icol in range(roi.width):
            for irow in range(roi.height):
                if not bdry[irow, icol]:
                    continue
                yy = roi.y0 + irow
                xx = roi.x0 + icol
                for ic in range(3):
                    old_val = np.round(img[yy, xx, ic]*(1.0-alpha)).astype(int)
                    new_img[yy, xx, ic] = old_val
                    new_val = np.round(alpha*color[ic]).astype(int)
                    new_img[yy, xx, ic] += new_val

    new_img = np.where(new_img <= 255, new_img, 255)
    return new_img


def roi_thumbnail(movie: OphysMovie,
                  roi: OphysROI,
                  timestamps: Optional[np.ndarray],
                  reducer: Callable = np.mean,
                  slop: int = 20,
                  roi_color: Tuple[int] = (255, 0, 0),
                  alpha=0.5) -> np.ndarray:
    """
    Get the thumbnail of an ROI from an OphysMovie

    Parameters
    ----------
    movie: OphysMovie

    roi: OphysROI

    timestamps: Optional[np.ndarray]
        The timestamps that you want to select when building
        the thumbnail

    reducer: Callable
        The function that will be used to convert
        OphysMovie.data[timestamps,:,:] into an image.
        Must accept the movie array and the kwargs 'axis'
        (Default: np.mean)

    slop: int
        The number of pixels beyond the ROI to return
        in the thumbnail

    roi_color: Tuple[int]
        color of ROI border as RGB tuple (default: (255, 0, 0))

    alpha: float
        transparency factor to apply to ROI (default=0.5)

    Returns
    -------
    thumbnail: np.ndarray
        An RGB representation of the thumbnail with the ROI
        border drawn around it
    """

    if timestamps is not None:
        clipped_movie = movie.data[timestamps, :, :]
    else:
        clipped_movie = movie.data

    x0 = roi.x0-slop//2
    x1 = roi.x0+roi.width+slop//2
    y0 = roi.y0-slop//2
    y1 = roi.y0+roi.height+slop//2

    dx = roi.width+slop
    dy = roi.height+slop
    if x0 < 0:
        x0 = 0
        x1 = x0+dx
    if y0 < 0:
        y0 = 0
        y1 = y0+dy
    if x1 >= movie.data.shape[1]:
        x1 = movie.data.shape[1]-1
        x0 = x1-dx
    if y1 >= movie.data.shape[0]:
        y1 = movie.data.shape[0]
        y0 = y1-dy

    clipped_movie = clipped_movie[:, y0:y1, x0:x1]
    mean_img = reducer(clipped_movie, axis=0)
    del clipped_movie

    thumbnail = np.zeros((mean_img.shape[0], mean_img.shape[1], 3),
                         dtype=int)

    v = mean_img.max()
    for ic in range(3):
        thumbnail[:, :, ic] = np.round(255*(mean_img/v)).astype(int)
    thumbnail = np.where(thumbnail <= 255, thumbnail, 255)

    # need to re-center the ROI
    new_roi = OphysROI(x0=roi.x0-x0,
                       y0=roi.y0-y0,
                       width=roi.width,
                       height=roi.height,
                       mask_matrix=roi.mask_matrix,
                       valid_roi=True,
                       roi_id=-999)

    thumbnail = add_roi_boundaries_to_img(thumbnail,
                                          roi_list=[new_roi],
                                          color=roi_color,
                                          alpha=alpha)

    return thumbnail


class ROIExaminer(object):

    def __init__(self, movie_path):
        self.ophys_movie = OphysMovie(str(movie_path),
                                      motion_border={'x0': 0,
                                                     'x1': 0,
                                                     'y0': 0,
                                                     'y1': 0})
        print('please wait while we load the movie data')
        self.ophys_movie.load_movie_data()
        print('now we will generate the maximum projection image')
        self.ophys_movie.get_max_rgb(keep_data=True)

        self._roi_from_id = None
        self._color_from_subset = None

    def purge_movie_data(self):
        self.ophys_movie.purge_movie()

    def load_rois_to_compare(self, roi_sets: List[Tuple]):
        """
        Load lists of ROIs to compare.

        Parameters
        ----------
        roi_set: List[Tuple]
            Each tuple consists of a color, which is a tuple representing
            the RGB color you want to signify that set of ROIs and a list
            of OphysROIs

        Raises
        ------
        RuntimeError
            If the ROI IDs in your sets of ROIs are not unique, an error
            will be raised. It will be easier to compare ROIs if they
            all have unique IDs.
        """

        self._roi_from_id = {}
        self._roi_from_subset = {}
        self._color_from_subset = {}
        all_roi_list = []
        for i_subset, roi_subset in enumerate(roi_sets):
            color = roi_subset[0]
            if not isinstance(color, tuple):
                raise RuntimeError("The proper form for roi_sets is "
                                   "[(color, ROI_list), "
                                   "(color, ROI_list)....]")

            self._color_from_subset[i_subset] = color
            self._roi_from_subset[i_subset] = roi_subset[1]
            for roi in roi_subset[1]:
                if roi.roi_id in self._roi_from_id:
                    other = self._roi_from_id[roi.roi_id]
                    raise RuntimeError(f"ROI ID {roi.roi_id} occurs "
                                       "at least twise in your roi_sets. "
                                       f"Once in subset {i_subset}; "
                                       "once in subset "
                                       f"{other['subset']}")
                obj = {}
                obj['roi'] = roi
                obj['subset'] = i_subset
                self._roi_from_id[roi.roi_id] = obj
                all_roi_list.append(roi)

        print('please wait while we extract the traces for these ROIs')
        trace_struct = self.ophys_movie.get_trace(all_roi_list)
        self._trace_from_id = {}
        for roi_id in trace_struct['roi'].keys():
            tr = trace_struct['roi'][roi_id]['signal']
            self._trace_from_id[roi_id] = tr
        return None

    def _max_projection_with_roi(self, rois_and_colors):
        """
        rois_and_colors is a list of dicts keyed on 'color'
        and 'rois'
        """
        output_img = self.ophys_movie.get_max_rgb()
        for obj in rois_and_colors:
            output_img = add_roi_boundaries_to_img(output_img,
                                                   roi_list=obj['rois'],
                                                   alpha=0.4,
                                                   color=obj['color'])
        return output_img

    def _add_labels(self, axis, rois_and_colors):

        n_roi = 0
        for subset in rois_and_colors:
            n_roi += len(subset['rois'])
        nx = -999.0*np.ones(n_roi, dtype=float)
        ny = -999.0*np.ones(n_roi, dtype=float)
        roi_ct = 0

        rng = np.random.RandomState(44)
        for subset in rois_and_colors:
            color_hex = '#%02x%02x%02x' % subset['color']
            for roi in subset['rois']:
                xx = roi.centroid_x
                yy = roi.centroid_y
                dd = None
                n_iter = 0

                # add random salt in case two labels would
                # be right on top of each other
                while (dd is None or dd < 50) and n_iter < 20:
                    if n_iter > 0:
                        _x = rng.normal()
                        _y = rng.normal()
                        _r = rng.normal(loc=n_iter, scale=5)
                        n = np.sqrt(_x**2+_y**2)
                        _x *= _r/n
                        _y *= _r/n
                        xx = roi.centroid_x + _x
                        yy = roi.centroid_y + _y

                    dd = np.min((xx-nx)**2+(yy-ny)**2)
                    n_iter += 1

                nx[roi_ct] = xx
                ny[roi_ct] = yy
                roi_ct += 1

                axis.text(xx, yy,
                          f'{roi.roi_id}',
                          color=color_hex,
                          fontsize=15)

        return axis

    def plot_roi(self,
                 subset_list: List[int],
                 axis: matplotlib.axes.Axes,
                 labels=False):

        for subset in subset_list:
            if subset not in self._color_from_subset:
                valid = list(self._color_from_subset.keys())
                valid.sort()
                raise RuntimeError(f"No data for subset {subset}. "
                                   f"Only subsets {valid} have been "
                                   "loaded")
        subset_args = []
        for subset in subset_list:
            obj = {}
            obj['color'] = self._color_from_subset[subset]
            obj['rois'] = self._roi_from_subset[subset]
            subset_args.append(obj)

        img_arr = self._max_projection_with_roi(subset_args)
        axis.imshow(img_arr)
        if not labels:
            return axis

        axis = self._add_labels(axis, subset_args)

        return axis

    def plot_distinct_rois(self,
                           subset_list: List[int],
                           axis: matplotlib.axes.Axes,
                           threshold: float = 0.5,
                           labels=False):

        if len(subset_list) > 2:
            raise RuntimeError("plot_distinct_rois only defined for "
                               f"2 subsets; you gave {subset_list}")

        roi_list_0 = self._roi_from_subset[subset_list[0]]
        roi_list_1 = self._roi_from_subset[subset_list[1]]
        overlap = find_overlapping_roi_pairs(roi_list_0,
                                             roi_list_1)

        overlap_rois = set()
        for pair in overlap:
            if pair[2] >= threshold or pair[3] >= threshold:
                overlap_rois.add(pair[0])
                overlap_rois.add(pair[1])

        rois_and_colors = []
        for subset in subset_list:
            color = self._color_from_subset[subset]
            roi_list = []
            for roi in self._roi_from_subset[subset]:
                if roi.roi_id not in overlap_rois:
                    roi_list.append(roi)
            if len(roi_list) > 0:
                rois_and_colors.append({'color': color, 'rois': roi_list})
        img_arr = self._max_projection_with_roi(rois_and_colors)
        axis.imshow(img_arr)
        if not labels:
            return axis
        axis = self._add_labels(axis, rois_and_colors)
        return axis

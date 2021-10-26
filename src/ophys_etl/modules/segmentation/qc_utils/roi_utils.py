import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx
from matplotlib import figure, cm as mplt_cm
from typing import List, Tuple, Callable, Optional, Union, Dict
import numpy as np
import pathlib

from ophys_etl.types import ExtractROI

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    convert_roi_keys,
    extract_roi_to_ophys_roi,
    mean_metric_from_roi, do_rois_abut)

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysMovie,
    find_overlapping_roi_pairs)

import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


def add_roi_mask_to_img(
        img: np.ndarray,
        roi: OphysROI,
        color: Tuple[int],
        alpha: float) -> np.ndarray:
    """
    Add colored ROI mask to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi: OphysROI

    color: Tuple[int]
        RGB color of ROI

    alpha: float

    Returns
    -------
    img: np.ndarray

    Note
    ----
    While this function does return an image, it also operates
    on img in place
    """
    rows = roi.global_pixel_array[:, 0]
    cols = roi.global_pixel_array[:, 1]
    for ic in range(3):
        old_vals = img[rows, cols, ic]
        new_vals = np.round(alpha*color[ic]+(1.0-alpha)*old_vals).astype(int)
        img[rows, cols, ic] = new_vals
    img = np.where(img >= 255, 255, img)
    return img


def add_roi_boundary_to_img(
        img: np.ndarray,
        roi: OphysROI,
        color: Tuple[int, int, int],
        alpha: float) -> np.ndarray:
    """
    Add colored ROI boundary to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi: OphysROI

    color: Tuple[int]
        RGB color of ROI

    alpha: float

    Returns
    -------
    img: np.ndarray

    Note
    ----
    While this function does return an image, it also operates
    on img in place
    """
    bdry = roi.boundary_mask
    valid = np.argwhere(bdry)
    rows = np.array([r+roi.y0 for r in valid[:, 0]])
    cols = np.array([c+roi.x0 for c in valid[:, 1]])
    for ic in range(3):
        old_vals = img[rows, cols, ic]
        new_vals = np.round(alpha*color[ic]+(1.0-alpha)*old_vals).astype(int)
        img[rows, cols, ic] = new_vals
    img = np.where(img >= 255, 255, img)
    return img


def add_list_of_roi_boundaries_to_img(
        img: np.ndarray,
        roi_list: Union[List[OphysROI], List[Dict]],
        multicolor=True,
        color: Optional[Tuple[int, int, int]] = None,
        alpha: float = 0.25) -> np.ndarray:
    """
    Add colored ROI boundaries to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi_list: List[OphysROI]
        list of ROIs to add to image

    multicolor
        Whether to use a color scheme such that touching rois have different
        colors
        If True, color will be ignored

    color: Optional[Tuple[int]]
        color of ROI border as RGB tuple (default: (255, 0, 0))
        If given, multicolor will be set to False

    alpha: float
        transparency factor to apply to ROI (default=0.25)

    Returns
    -------
    new_img: np.ndarray
        New image with ROI borders superimposed
    """

    if color is not None:
        multicolor = False

    if color is None and not multicolor:
        raise ValueError('Either specify a color or set multicolor to True')

    new_img = np.copy(img)
    if len(roi_list) == 0:
        return new_img

    if not isinstance(roi_list[0], OphysROI):
        roi_list = convert_roi_keys(roi_list)
        roi_list = [OphysROI.from_schema_dict(roi)
                    for roi in roi_list]

    if multicolor:
        color_map = get_roi_color_map(roi_list=roi_list)
    else:
        color_map = {roi.roi_id: color for roi in roi_list}

    for roi in roi_list:
        new_img = add_roi_boundary_to_img(
                      new_img,
                      roi,
                      color_map[roi.roi_id],
                      alpha)
    return new_img


def add_labels_to_axes(axis: matplotlib.axes.Axes,
                       roi_list: Union[List[OphysROI], List[Dict]],
                       colors: Union[Tuple[int], List[Tuple[int]]],
                       fontsize: int = 15,
                       origin: Tuple[int, int] = (0, 0),
                       frame_shape: Optional[Tuple[int, int]] = None):
    """
    Add labels to a plot of ROIs

    Parameters
    ----------
    axis: matplotlib.axes.Axes
    rois: List[OphysROI]:
        the ROIs
    colors: Union[Tuple[int], List[Tuple[int]]]
        if not a list, all ROIs will get same color
    fontsize: int
        size of font used when putting labels on ROIs
    origin: Optional[Tuple[int, int]]
        origin of the image in the original image
    frame_shape: Optional[Tuple[int, int]]
        shape of the field of view (rows, cols)

    Returns
    -------
    matplotlib.axes.Axes
        The input axis with the figure added
    """

    if frame_shape is not None:
        rowmax = origin[0]+frame_shape[0]
        colmax = origin[1]+frame_shape[1]
    else:
        rowmax = None
        colmax = None

    n_roi = len(roi_list)

    if not isinstance(colors, list):
        colors = [colors] * n_roi

    if not isinstance(roi_list[0], OphysROI):
        roi_list = convert_roi_keys(roi_list)
        roi_list = [OphysROI.from_schema_dict(roi) for roi in roi_list]

    nx = -999.0*np.ones(n_roi, dtype=float)
    ny = -999.0*np.ones(n_roi, dtype=float)
    roi_ct = 0

    rng = np.random.RandomState(44)
    for color, roi in zip(colors, roi_list):
        color_hex = '#%02x%02x%02x' % color
        xx = roi.centroid_x-origin[1]
        yy = roi.centroid_y-origin[0]
        dd = None
        n_iter = 0

        # add random salt in case two labels would
        # be right on top of each other
        keep_going = True
        while keep_going:
            keep_going = False
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
            if n_iter > 20:
                if rowmax is not None and yy > rowmax:
                    keep_going = True
                elif colmax is not None and xx > colmax:
                    keep_going = True
                elif (dd is None or dd < 100):
                    keep_going = True

        nx[roi_ct] = xx
        ny[roi_ct] = yy
        roi_ct += 1

        axis.text(xx, yy,
                  f'{roi.roi_id}',
                  color=color_hex,
                  fontsize=fontsize,
                  clip_on=True)

    return axis


def roi_thumbnail(movie: OphysMovie,
                  roi: OphysROI,
                  timestamps: Optional[np.ndarray],
                  reducer: Callable = np.mean,
                  slop: int = 20,
                  roi_color: Tuple[int, int, int] = (255, 0, 0),
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
    if x1 >= movie.data.shape[2]:
        x1 = movie.data.shape[2]-1
        x0 = x1-dx
    if y1 >= movie.data.shape[1]:
        y1 = movie.data.shape[1]-1
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

    thumbnail = add_list_of_roi_boundaries_to_img(
                                          thumbnail,
                                          roi_list=[new_roi],
                                          color=roi_color,
                                          alpha=alpha)

    return thumbnail


class ROIExaminer(object):
    """
    A class for comparing ROIs found by different segmentation schemes
    to the movie they were run on

    Parameters
    ----------
    movie_path: str or pathlib.Path
        The path to the motion corrected movie to be used as the baseline
        for ROI comparison
    """

    def __init__(self, movie_path: Union[str, pathlib.Path]):
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

    def load_rois_to_compare(self, roi_sets: List[Tuple]) -> None:
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

    def _max_projection_with_roi(self,
                                 rois_and_colors: List,
                                 alpha: float = 0.5) -> np.ndarray:
        """
        Generate an RGB image of the maximum projection with
        ROIs superimposed

        Parameters
        ----------
        rois_and_colors: List
            Each element in the list is a Dict representing
            a set of ROIs. 'color' points to a tuple representing
            the color of that set of ROIs. 'rois' points to a list
            of OphysROIs.

        alpha: float
            The transparency factor applied to the ROIs
            (default: 0.5)

        Return
        ------
        np.ndarray
            An RGB image of the maximum projection image with
            ROIs superimposed.
        """
        output_img = self.ophys_movie.get_max_rgb()
        for obj in rois_and_colors:
            output_img = add_list_of_roi_boundaries_to_img(
                                                   output_img,
                                                   roi_list=obj['rois'],
                                                   alpha=alpha)
        return output_img

    def plot_rois(self,
                  subset_list: List[int],
                  axis: matplotlib.axes.Axes,
                  labels: bool = False,
                  alpha: float = 0.5) -> matplotlib.axes.Axes:
        """
        Plot the maximum projection image with ROIs superimposed

        Parameters
        ----------
        subset_list: List[int]
            The subsets of ROIs to be plotted (i.e. indexes referring
            to the sets of ROIs loaded with load_rois_to_compare)

        axis: matplotlib.axes.Axes
            The axis in which to plot

        labels: boolean
            If True, label the ROIs with roi_id
            (default=False)

        alpha: float
            The transparency factor to apply to the ROIs
            (default=0.5)

        Returns
        -------
        matplotlib.axes.Axes
            The input axis with the plot added to it
        """

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

        img_arr = self._max_projection_with_roi(subset_args, alpha=alpha)
        axis.imshow(img_arr)
        if not labels:
            return axis

        for subset in subset_args:
            axis = add_labels_to_axes(axis, subset["rois"], subset["color"])

        return axis

    def _plot_union_of_rois(self,
                            subset_list: List[int],
                            axis: matplotlib.axes.Axes,
                            threshold: float = 0.5,
                            labels=False,
                            alpha=0.5,
                            plot_overlapping=True):
        """
        Actually plot either the union or the complement of
        the union of two sets of ROIs

        Parameters
        ----------
        subset_list: List[int]
            The subsets of ROIs to be compared (i.e.
            indexes referring to the sets of ROIs loaded with
            load_rois_to_compare; must be exactly 2
            of them)

        axis: matplotlib.axes.Axes
            The axis in which to plot

        threshold: float
            The fraction of overlapping area that at least one ROI
            must meet for a pair of ROIs to be considered overlapping
            (default=0.5)

        labels: boolean
            If True, label the ROIs with roi_id
            (default=False)

        alpha: float
            The transparency factor to apply to the ROIs
            (default=0.5)

        plot_overlapping: boolean
            If True, plot ROIs that overlap; if False, plot
            ROIs that don't overlap (default: True)

        Returns
        -------
        matplotlib.axes.Axes
            The input axis with the plot added to it
        """

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
                if not plot_overlapping and roi.roi_id not in overlap_rois:
                    roi_list.append(roi)
                elif plot_overlapping and roi.roi_id in overlap_rois:
                    roi_list.append(roi)

            if len(roi_list) > 0:
                rois_and_colors.append({'color': color, 'rois': roi_list})

        img_arr = self._max_projection_with_roi(rois_and_colors,
                                                alpha=alpha)
        axis.imshow(img_arr)
        if not labels:
            return axis
        for subset in rois_and_colors:
            axis = add_labels_to_axes(axis, subset["rois"], subset["color"])
        return axis

    def plot_distinct_rois(self,
                           subset_list: List[int],
                           axis: matplotlib.axes.Axes,
                           threshold: float = 0.5,
                           labels: bool = False,
                           alpha: float = 0.5):
        """
        Plot the maximum projection image with the non-overlapping
        ROIs from two subsets of ROIs superimposed (i.e. ROIs that
        exist in one set but not another)

        Parameters
        ----------
        subset_list: List[int]
            The subsets of ROIs to be compared (i.e.
            indexes referring to the sets of ROIs loaded with
            load_rois_to_compare; must be exactly 2
            of them)

        axis: matplotlib.axes.Axes
            The axis in which to plot

        threshold: float
            The fraction of overlapping area that at least one ROI
            must meet for a pair of ROIs to be considered overlapping
            (default=0.5)

        labels: boolean
            If True, label the ROIs with roi_id
            (default=False)

        alpha: float
            The transparency factor to apply to the ROIs
            (default=0.5)

        Returns
        -------
        matplotlib.axes.Axes
            The input axis with the plot added to it
        """

        return self._plot_union_of_rois(subset_list,
                                        axis,
                                        threshold=threshold,
                                        labels=labels,
                                        plot_overlapping=False,
                                        alpha=alpha)

    def plot_overlapping_rois(self,
                              subset_list: List[int],
                              axis: matplotlib.axes.Axes,
                              threshold: float = 0.5,
                              labels: bool = False,
                              alpha: float = 0.5) -> matplotlib.axes.Axes:
        """
        Plot the maximum projection image with the overlapping ROIs
        from two subsets of ROIs superimposed

        Parameters
        ----------
        subset_list: List[int]
            The subsets of ROIs to be compared (i.e.
            indexes referring to the sets of ROIs loaded with
            load_rois_to_compare; must be exactly 2
            of them)

        axis: matplotlib.axes.Axes
            The axis in which to plot

        threshold: float
            The fraction of overlapping area that at least one ROI
            must meet for a pair of ROIs to be considered overlapping
            (default=0.5)

        labels: boolean
            If True, label the ROIs with roi_id
            (default=False)

        alpha: float
            The transparency factor to apply to the ROIs
            (default=0.5)

        Returns
        -------
        matplotlib.axes.Axes
            The input axis with the plot added to it
        """

        return self._plot_union_of_rois(subset_list,
                                        axis,
                                        threshold=threshold,
                                        labels=labels,
                                        plot_overlapping=True,
                                        alpha=alpha)

    def plot_thumbnail_and_trace(self,
                                 roi_id: int,
                                 timestamps: Optional[np.ndarray] = None
                                 ) -> None:
        """
        Plot a thumbnail of an ROI next to its trace. Note, the thumbnail
        will be rescaled to the local maximum projection value in an
        attempt to highlight contrast in dark regions of the field of view.

        Parameters
        ----------
        roi_id: int

        timestamps: Optional[np.ndarray]
            The timestamps from the trace that will be stacked when
            making the thumbnail.

        Returns
        -------
        None

        Note
        ----
        Because this method is meant to be called in a notebook, it
        merely instantiates a matplotlib.figure.Figure without
        returning it. This is sufficient for the image to be
        displayed in a notebook with `%matplotlib inline`
        """
        if roi_id not in self._roi_from_id:
            raise RuntimeError(f"{roi_id} is not a valid ROI")

        fig = plt.figure(constrained_layout=True, figsize=(30, 10))

        trace = self._trace_from_id[roi_id]
        roi = self._roi_from_id[roi_id]['roi']
        subset = self._roi_from_id[roi_id]['subset']
        color = self._color_from_subset[subset]

        grid = gridspec.GridSpec(1, 4, figure=fig)
        thumbnail_axis = fig.add_subplot(grid[0, 0])
        trace_axis = fig.add_subplot(grid[0, 1:])

        thumbnail = roi_thumbnail(self.ophys_movie,
                                  roi,
                                  timestamps=timestamps,
                                  reducer=np.max,
                                  roi_color=color)
        thumbnail_axis.imshow(thumbnail)
        thumbnail_axis.set_title(f'roi {roi.roi_id}', fontsize=30)
        thumbnail_axis.tick_params(axis='both', labelsize=0)
        tt = np.arange(len(trace))
        tmin = trace[trace > 1.0e-6].min()
        tmax = trace.max()
        trace_axis.plot(tt, trace,
                        color='#%02x%02x%02x' % color)
        trace_axis.set_ylim(tmin, tmax)
        trace_axis.tick_params(axis='both', labelsize=20)
        return None


def add_rois_to_axes(
        axes: matplotlib.axes.Axes,
        roi_list: List[ExtractROI],
        shape: Tuple[int, int],
        rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
        ) -> None:
    """

    Parameters
    ----------
    axes: matplotlib.axes.Axes
        the axes to add to
    roi_list: List[ExtractROI]
        the ROIs to add
    shape: Tuple[int, int]
        shape of the FOV
    rgba: Tuple[float, float, float, float]
        0.0 - 1.0  RGBA values for the outlines

    """
    bdry_pixels = np.zeros((*shape, 4), dtype=float)
    for roi in roi_list:
        ophys_roi = OphysROI(
                        roi_id=0,
                        x0=roi['x'],
                        y0=roi['y'],
                        width=roi['width'],
                        height=roi['height'],
                        valid_roi=False,
                        mask_matrix=roi['mask'])

        bdry = ophys_roi.boundary_mask
        for ir in range(ophys_roi.height):
            for ic in range(ophys_roi.width):
                if bdry[ir, ic]:
                    bdry_pixels[ir+ophys_roi.y0,
                                ic+ophys_roi.x0] = rgba
    axes.imshow(bdry_pixels)


def create_roi_plot(plot_path: pathlib.Path,
                    img_data: np.ndarray,
                    roi_list: List[ExtractROI]) -> None:
    """
    Generate a side-by-side plot comparing the image data
    used to seed ROI generation with the borders of the
    discovered ROIs

    Parameters
    ----------
    plot_path: pathlib.Path
        Path to file where plot will be saved

    img_data: np.ndarray
        The baseline image over which to plot the ROIs

    roi_list: List[ExtractROI]

    Returns
    -------
    None

    """
    fig = figure.Figure(figsize=(40, 20))
    axes = [fig.add_subplot(1, 2, i) for i in [1, 2]]
    axes[0].imshow(img_data)
    axes[1].imshow(img_data)
    add_rois_to_axes(axes[1], roi_list, img_data.shape)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    return None


def roi_average_metric(roi_list: List[ExtractROI],
                       metric_image: np.ndarray) -> Dict[int, float]:
    """calculate the average metric for a list of ROIs
    given an image representation of the metric

    Parameters
    ----------
    roi_list: List[ExtractROI]
        the list of ROIs
    metric_image: np.ndarray
        a 2D array of values from which to calculate the average
        metric, from each ROI mask

    Returns
    -------
    average_metric: Dict[int, float]
        keys: roi[i]["id"]
        values: average metric within roi[i]["mask"]

    """
    average_metric: Dict[int, float] = dict()
    for roi in roi_list:
        ophys_roi = extract_roi_to_ophys_roi(roi)
        average_metric[roi["id"]] = mean_metric_from_roi(
                                         ophys_roi,
                                         metric_image)

    return average_metric


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


def get_roi_color_map(
        roi_list: List[OphysROI]) -> Dict[int, Tuple[int, int, int]]:
    """
    Take a list of OphysROI and return a dict mapping ROI ID
    to RGB color so that no ROIs that touch have the same color

    Parametrs
    ---------
    roi_list: List[OphysROI]

    Returns
    -------
    color_map: Dict[int, Tuple[int, int, int]]
    """
    roi_graph = networkx.Graph()
    for roi in roi_list:
        roi_graph.add_node(roi.roi_id)
    for ii in range(len(roi_list)):
        roi0 = roi_list[ii]
        for jj in range(ii+1, len(roi_list)):
            roi1 = roi_list[jj]

            # value of 5 is so that singleton ROIs that
            # are near each other do not get assigned
            # the same color
            abut = do_rois_abut(roi0, roi1, 5.0)
            if abut:
                roi_graph.add_edge(roi0.roi_id, roi1.roi_id)
                roi_graph.add_edge(roi1.roi_id, roi0.roi_id)

    nx_coloring = networkx.greedy_color(roi_graph)
    n_colors = len(set(nx_coloring.values()))

    mplt_color_map = mplt_cm.jet

    # create a list of colors based on the matplotlib color map
    raw_color_list = []
    for ii in range(n_colors):
        color = mplt_color_map(0.8*(1.0+ii)/(n_colors+1.0))
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        raw_color_list.append(color)

    # re-order colors so that colors that are adjacent in index
    # have higher contrast
    step = max(n_colors//3, 1)
    color_list = []
    for i0 in range(step):
        for ii in range(i0, n_colors, step):
            this_color = raw_color_list[ii]
            color_list.append(this_color)

    # reverse color list, since matplotlib.cm.jet will
    # assign a dark blue as color_list[0], which isn't
    # great for contrast
    color_list.reverse()

    color_map = {}
    for roi_id in nx_coloring:
        color_map[roi_id] = color_list[nx_coloring[roi_id]]
    return color_map

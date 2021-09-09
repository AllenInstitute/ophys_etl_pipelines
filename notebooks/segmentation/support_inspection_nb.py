import h5py
import multiprocessing
import ipywidgets as widgets
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import display
from functools import partial
from pathlib import Path

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
        add_list_of_roi_boundaries_to_img, add_labels_to_axes,
        convert_roi_keys)
from ophys_etl.modules.segmentation.processing_log import \
        SegmentationProcessingLog
from ophys_etl.modules.segmentation.qc_utils.graph_plotting import \
        draw_graph_edges


def new_background_selector(nrows, ncols, background_paths):
    """creates a list of widgets for selecting subplot background images.

    Parameters
    ----------
    nrows: int
        the number of subplot rows
    ncols: int
        the number of subplot columns
    background_paths: List[str]
        the list of paths

    Returns
    -------
    background_selector: List[widgets.Dropdown]
        there is one dropdown widget per ROI inspection subplot.
        each dropdown widget lists the background_paths options.
        The displayed key is the 'path.name' and the selected value is 'path'

    """
    background_selector = [
        widgets.Dropdown(
            options=[(None, None)] + [(p.name, p) for p in background_paths],
            description=f"({i}, {j})",
            layout=widgets.Layout(display='flex', align_items='flex-start')
        )
        for i in range(nrows)
        for j in range(ncols)]
    return background_selector


def new_processing_log_selector(nrows, ncols, processing_logs):
    """creates 2 lists of widgets for selecting foreground ROIs.

    Parameters
    ----------
    nrows: int
        the number of subplot rows
    ncols: int
        the number of subplot columns
    processing_logs: List[str]
        the list of paths

    Returns
    -------
    foreground_selector: List[widgets.Dropdown]
        there is one dropdown widget per ROI inspection subplot.
        each dropdown widget lists the processing_log option
        The displayed key is the 'path.name' and the selected value is 'path'
    dataset_selector: List[widgets.Dropdown]
        for the selected foreground, or procesing log, will display the
        h5 group names which are available that contain ROIs.

    """
    foreground_selector = [
        widgets.Dropdown(
            options=[(None, None)] + [(f.name, f) for f in processing_logs],
            description=f"({i}, {j})")
        for i in range(nrows)
        for j in range(ncols)]
    dataset_selector = [
        widgets.Dropdown(options=[], layout=widgets.Layout(width='150px'))
        for i in range(nrows)
        for j in range(ncols)]
    return foreground_selector, dataset_selector


def new_plot_update_buttons(nrows, ncols):
    """ list of update buttons per subplot

    Parameters
    ----------
    nrows: int
        the number of subplot rows
    ncols: int
        the number of subplot columns

    Returns
    -------
    buttons: List[widgets.Button]
        the update buttons

    """
    buttons = [
        widgets.Button(description="Update")
        for i in range(nrows)
        for j in range(ncols)]
    return buttons


def update_plot(widget, fig, axes, background_widget, log_widget,
                dataset_widget, label_widget, valid_widget):
    """updates the plots for ROI inspection

    Parameters
    ----------
    widget:
        I believe this needs to be here for this function to be used as an
        argument for button.on_click()
    fig: matplotlib.figure.Figure
        the figure for plotting into
    axes: matplotlib.axes.Axes
        the axes for plotting into
    background_widget:
        the widget controlling this axes' background
    log_widget:
        the widget controlling this axes' foreground (1 of 2)
    dataset_widget:
        the widget controlling this axes' foreground (2 of 2)
    label_widget:
        the widget controlling whether labels should be applied to these
        axes.
    valid_widget:
        the widget controlling whether only valid ROIs should be plotted.

    Notes
    -----
    - the labels can make the plot difficult to read, and could be improved.

    """
    background_path = background_widget.value
    if background_path is None:
        im = np.ones((512, 512, 3), dtype="uint8") * 255
    else:
        if background_path.suffix == ".pkl":
            graph = nx.read_gpickle(background_path)
            edge = list(graph.edges(data=True))[0]
            attribute_name = list(edge[2].keys())[0]
            axes.cla()
            draw_graph_edges(fig, axes, graph, attribute_name, colorbar=False)
            title = f"{background_path.name}"
            axes.set_title(title, fontsize=10)
            fig.tight_layout()
            return

        im = plt.imread(background_path)
        if im.ndim == 2:
            im = np.dstack([im, im, im])

    log_path = log_widget.value
    dataset = dataset_widget.value
    valid_only = valid_widget.value
    if (log_path is not None) & (dataset is not None):
        processing_log = SegmentationProcessingLog(log_path)
        rois = processing_log.get_rois_from_group(
                dataset, valid_only=valid_only)
        im = add_list_of_roi_boundaries_to_img(im, rois)
    axes.cla()
    axes.imshow(im)
    title = ""
    if background_path is not None:
        title += f"{background_path.name}"
    if log_path is not None:
        if title != "":
            title += "\n"
        title += f"{log_path.name} - {dataset}"
    if title != "":
        title += "\n"
    title += f"valid_only: {valid_only}"
    axes.set_title(title, fontsize=10)

    if label_widget.value:
        add_labels_to_axes(axes, rois, (255, 0, 0), fontsize=6)
    fig.tight_layout()


def roi_viewer(inspection_manifest, nrows=1, ncols=1):
    """displays a figure and selector boxes for viewing ROIs

    Parameters
    ----------
    inspection_manifest: dict
        {'videos': a list of video paths/str (not used here),
         'processing_logs: a list of processing log paths/str,
         'backgounds: a list of background paths/str}
    nrows: int
        how many rows of subplots
    ncols: int
        how many columns of subplots

    """
    # erase old figure
    fig = plt.figure(1)
    plt.close(fig)

    # make new figure
    fig, axes = plt.subplots(
            nrows, ncols, clear=True, sharex=True, sharey=True,
            num=1, squeeze=False)
    plt.show()
    fig.tight_layout()

    # make selectors for each axis and attach to callbacks
    backgrounds = new_background_selector(
            nrows, ncols, inspection_manifest["backgrounds"])
    processing_logs, datasets = new_processing_log_selector(
            nrows, ncols, inspection_manifest["processing_logs"])

    def on_change_logs(index):
        def on_change(change):
            """open the log and see what groups have ROIs in them.
            display the available groups in the datasets widget.
            """
            if change['type'] == 'change' and change['name'] == 'value':
                options = []
                with h5py.File(processing_logs[index].value, "r") as f:
                    for key in f.keys():
                        if isinstance(f[key], h5py.Group):
                            if "rois" in f[key]:
                                options.append(key)
                datasets[index].options = options
        return on_change

    for i in range(len(processing_logs)):
        processing_logs[i].observe(on_change_logs(i))

    label_checks = [
            widgets.Checkbox(
                description="yes",
                layout=widgets.Layout(width='150px'))
            for i in range(nrows*ncols)]
    valid_only = [
            widgets.Checkbox(
                description="valid",
                layout=widgets.Layout(width='150px'))
            for i in range(nrows*ncols)]
    partials = []
    for ax, bgw, logw, dataw, lw, vw in zip(axes.flat,
                                            backgrounds,
                                            processing_logs,
                                            datasets,
                                            label_checks,
                                            valid_only):
        partials.append(partial(update_plot,
                                fig=fig,
                                axes=ax,
                                log_widget=logw,
                                dataset_widget=dataw,
                                background_widget=bgw,
                                label_widget=lw,
                                valid_widget=vw))
    update_buttons = new_plot_update_buttons(nrows, ncols)
    for partial_fun, button in zip(partials, update_buttons):
        button.on_click(partial_fun)

    # group the selectors and display
    background_box = widgets.VBox(
            [widgets.Label("backgrounds")] + backgrounds)
    log_selection = widgets.VBox(
            [widgets.Label("processing logs")] + processing_logs)
    dataset_selection = widgets.VBox(
            [widgets.Label("datasets")] + datasets)
    button_box = widgets.VBox(
            [widgets.Label("update buttons")] + update_buttons)
    label_box = widgets.VBox(
            [widgets.Label("include labels")] + label_checks)
    valid_box = widgets.VBox(
            [widgets.Label("valid only")] + valid_only)
    selector_box = widgets.HBox([background_box,
                                 log_selection,
                                 dataset_selection,
                                 label_box,
                                 valid_box,
                                 button_box])
    display(selector_box)


def all_roi_dicts(inspection_manifest):
    """returns a dictionary
    keys are log_path.fname-hdf5_group
    values are the deserialized ROI List[Dict] for that key
    """
    results = dict()
    for log in inspection_manifest["processing_logs"]:
        groups = []
        with h5py.File(log, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    if "rois" in f[key]:
                        groups.append(key)
        splog = SegmentationProcessingLog(log)
        for group in groups:
            results[f"{log.name}-{group}"] = splog.get_rois_from_group(group)
    return results


def get_movie_widget_list(video_list):
    """a list of checkboxes for available movies
    NOTE: dscription_tooltip is the full path for retrieval later.
    """
    movie_widget_list = [
        widgets.Checkbox(
            value=True,
            description=Path(f).name,
            description_tooltip=str(f),
            layout={'width': 'max-content'}
        )
        for f in video_list]
    return movie_widget_list


def get_roi_dropdowns(rois_dict):
    """a dropdown listing the available ROI IDs per dataset
    """
    roi_drops = [
        widgets.Dropdown(
            options=np.sort([-1] + [i["id"] for i in v]),
            description=k,
            layout={'width': 'max-content'},
            style={'description_width': 'initial'}
        )
        for k, v in rois_dict.items()]
    return roi_drops


def get_trace_selection_widgets(inspection_manifest):
    """returns the trace selection widgets, grouped in one box to
    display (all_widgets) and individually to help grab items
    elsewhere.

    Notes
    -----
    - admittedly, this could be cleaner, probably ROIViewer and TraceViewer
    should both be classes so things don't need to be passed around so much.

    """
    movie_widget_list = get_movie_widget_list(inspection_manifest["videos"])
    movie_list = widgets.VBox(movie_widget_list)

    rois_dict = all_roi_dicts(inspection_manifest)
    roi_drops = get_roi_dropdowns(rois_dict)
    roi_list = widgets.VBox(roi_drops)

    movies_and_rois = widgets.HBox(
        [widgets.VBox([widgets.HTML(value="<b>available movies</b>"),
                       movie_list]),
         widgets.VBox([widgets.HTML(value="<b>available ROIs</b>"),
                       roi_list])],
        layout={'width': 'max-content'})
    trace_grouping = widgets.Dropdown(
        options=[
            ("group traces by ROI", 0),
            ("group traces by movie", 1)])
    all_widgets = widgets.VBox([movies_and_rois, trace_grouping])
    return rois_dict, all_widgets, roi_drops, movie_widget_list, trace_grouping


def extents_from_roi(roi):
    """get bounding box extents for an ROI
    """
    xmin = roi["x"]
    xmax = xmin + roi["width"]
    ymin = roi["y"]
    ymax = ymin + roi["height"]
    return xmin, xmax, ymin, ymax


def get_trace(movie_path, roi):
    """extract a trace given a movie path and an ROI
    """
    xmin, xmax, ymin, ymax = extents_from_roi(roi)
    with h5py.File(movie_path, "r") as f:
        data = f["data"][:, ymin: ymax, xmin: xmax]
    data = data.reshape(data.shape[0], -1)
    mask = np.array(roi["mask_matrix"]).reshape(data.shape[1])
    npix = np.count_nonzero(mask)
    trace = data[:, mask].sum(axis=1) / npix
    return trace


def trace_plot_callback(rois_dict, roi_drops,
                        movie_widget_list, trace_grouping):
    """plot traces of selected ROIs from selected movies
    grouped by choice in trace_grouping.
    """
    # determine which ROIs are selected
    rois_lookup = dict()
    for roi_select in roi_drops:
        if roi_select.value != -1:
            rois_lookup[roi_select.description] = int(roi_select.value)
    for k, v in list(rois_lookup.items()):
        j = rois_dict[k]
        j = convert_roi_keys(j)
        for i in j:
            if i["id"] == v:
                rois_lookup[k] = i

    # determine which movie paths are selected
    movie_paths = []
    for movie_widget in movie_widget_list:
        if movie_widget.value:
            movie_paths.append(Path(movie_widget.description_tooltip))

    # get all combinations of ROIs and movie paths
    trace_list = []
    for roi_source, roi in rois_lookup.items():
        for movie_path in movie_paths:
            trace_list.append(
                {
                    "roi_source": roi_source,
                    "roi": roi,
                    "roi_id": roi["id"],
                    "movie_path": movie_path,
                    "movie_label": movie_path.name,
                    "roi_label": f"{roi_source}_{roi['id']}"
                }
            )

    # load traces in parallel
    args = [(i["movie_path"], i["roi"]) for i in trace_list]
    with multiprocessing.Pool(4) as pool:
        results = pool.starmap(get_trace, args)
    for i, result in enumerate(results):
        trace_list[i]["trace"] = result

    # group according to selected method
    df = pd.DataFrame.from_records(trace_list)
    if trace_grouping.value == 0:
        groups = df.groupby(["roi_source", "roi_id"])
        label = "movie_label"
    elif trace_grouping.value == 1:
        groups = df.groupby(["movie_label"])
        label = "roi_label"

    fig2, axes2 = plt.subplots(
            len(groups), 1, clear=True,
            sharex=True, sharey=False, squeeze=False)
    for group, ax in zip(groups, axes2.flat):
        if isinstance(group[0], tuple):
            ylab = "\n".join([f"{i}" for i in group[0]])
        else:
            ylab = group[0]
        ax.set_ylabel(ylab, fontsize=6)
        for entry in group[1].iterrows():
            ax.plot(entry[1]["trace"], linewidth=0.4, label=entry[1][label])

    axes2.flat[0].legend(fontsize=6)
    fig2.tight_layout()
    plt.show()

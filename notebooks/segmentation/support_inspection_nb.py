import h5py
import ipywidgets as widgets
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from IPython.core.display import display
from functools import partial

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
        add_list_of_roi_boundaries_to_img, add_labels_to_axes)
from ophys_etl.modules.segmentation.processing_log import \
        SegmentationProcessingLog
from ophys_etl.modules.segmentation.qc_utils.graph_plotting import \
        draw_graph_edges


def new_background_selector(nrows, ncols, background_paths):
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
    buttons = [
        widgets.Button(description="Update")
        for i in range(nrows)
        for j in range(ncols)]
    return buttons


def update_plot(widget, fig, axes, background_widget, log_widget,
                dataset_widget, label_widget):
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
    if (log_path is not None) & (dataset is not None):
        processing_log = SegmentationProcessingLog(log_path)
        rois = processing_log.get_rois_from_group(dataset)
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
    axes.set_title(title, fontsize=10)

    if label_widget.value:
        add_labels_to_axes(axes, rois, (255, 0, 0), fontsize=6)
    fig.tight_layout()


def roi_viewer(inspection_manifest, nrows=1, ncols=1):
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

    label_checks = [widgets.Checkbox(description="include labels")
                    for i in range(nrows*ncols)]
    partials = []
    for ax, bgw, logw, dataw, lw in zip(axes.flat,
                                        backgrounds,
                                        processing_logs,
                                        datasets,
                                        label_checks):
        partials.append(partial(update_plot,
                                fig=fig,
                                axes=ax,
                                log_widget=logw,
                                dataset_widget=dataw,
                                background_widget=bgw,
                                label_widget=lw))
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
    selector_box = widgets.HBox([background_box,
                                 log_selection,
                                 dataset_selection,
                                 label_box,
                                 button_box])
    display(selector_box)

from pathlib import Path
from timeit import default_timer as timer

import argschema
import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import coo_matrix

from ophys_etl.transforms.roi_transforms import (
    suite2p_rois_to_coo,
    binarize_roi_mask,
    crop_roi_mask
)
from ophys_etl.transforms.trace_transforms import extract_traces


def plot_binarized_vs_weighted_roi(weighted_mask: coo_matrix,
                                   binary_mask: coo_matrix,
                                   weighted_trace: np.ndarray,
                                   binary_trace: np.ndarray
                                   ) -> matplotlib.figure.Figure:
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=6)

    # Plot ROIs
    binary_roi_ax = fig.add_subplot(gs[:-1, -3:])
    weighted_roi_ax = fig.add_subplot(gs[:-1, :-3],
                                      sharex=binary_roi_ax,
                                      sharey=binary_roi_ax)

    weighted_roi_ax.set_xticks([])
    weighted_roi_ax.set_yticks([])
    weighted_roi_ax.set_title("Native Suite2P (weighted) ROI")
    weighted_roi_ax.imshow(weighted_mask.toarray())
    xmin, xmax, ymin, ymax = weighted_roi_ax.axis()

    binary_roi_ax.set_xticks([])
    binary_roi_ax.set_yticks([])
    binary_roi_ax.set_title("Binarized Suite2P ROI")
    binary_roi_ax.imshow(binary_mask.toarray())
    binary_roi_ax.set_xlim(xmin, xmax)
    binary_roi_ax.set_ylim(ymin, ymax)

    # Plot traces
    binary_trace_ax = fig.add_subplot(gs[-1, -3:])
    weighted_trace_ax = fig.add_subplot(gs[-1, :-3],
                                        sharex=binary_trace_ax,
                                        sharey=binary_trace_ax)

    weighted_trace_ax.set_ylabel("Weighted F")
    weighted_trace_ax.set_xlabel("Frame Number")
    weighted_trace_ax.plot(range(len(weighted_trace)), weighted_trace,
                           linewidth=0.5)

    binary_trace_ax.set_ylabel("Binarized F")
    binary_trace_ax.set_xlabel("Frame Number")
    binary_trace_ax.plot(range(len(binary_trace)), binary_trace,
                         linewidth=0.25)

    return fig


class RoiQcReportGeneratorSchema(argschema.ArgSchema):
    ophys_movie_path = argschema.fields.InputFile(
        required=True,
        description=("Path to input video (*.h5). In Allen production case, "
                     "assumed to be motion-corrected."))

    ophys_movie_h5_key = argschema.fields.Str(
        required=False,
        missing="data",
        default="data",
        description="h5py key for accessing video data array.")

    suite2p_stat_path = argschema.fields.InputFile(
        required=True,
        description=("Path to Suite2P `stat.npy` output file which contains"
                     "putative ROI data."))

    output_dir = argschema.fields.OutputDir(
        required=True,
        description="Directory to save binary vs weighted ROI comparison PDF.")


class RoiQcReportGenerator(argschema.ArgSchemaParser):
    default_schema = RoiQcReportGeneratorSchema

    def run(self):
        movie_path = self.args["ophys_movie_path"]

        # Load and convert Suite2P ROIs
        s2p_stat = np.load(self.args["suite2p_stat_path"], allow_pickle=True)

        with h5py.File(movie_path, "r") as movie_f:
            movie_shape = movie_f[self.args["ophys_movie_h5_key"]][0].shape

        weighted_rois = suite2p_rois_to_coo(s2p_stat, movie_shape)
        binary_rois = [binarize_roi_mask(w_roi) for w_roi in weighted_rois]

        # Extract traces
        with h5py.File(movie_path, "r") as movie_f:
            movie_frames = movie_f[self.args["ophys_movie_h5_key"]]

            self.logger.info("Extracting weighted traces, please be patient.")
            start = timer()
            weighted_traces = extract_traces(movie_frames, weighted_rois)
            end = timer()
            self.logger.info(f"Weighted trace extraction: {end - start} secs")

            self.logger.info("Extracting binary traces, please be patient.")
            start = timer()
            binary_traces = extract_traces(movie_frames, binary_rois)
            end = timer()
            self.logger.info(f"Binary trace extraction: {end - start} secs")

        # Crop ROIs
        cropped_weighted = [crop_roi_mask(w_roi) for w_roi in weighted_rois]
        # Because the roi bounds can differ a bit between weighted and
        # binarized rois, for display purposes we should binarize the cropped
        # weighted rois.
        cropped_binary = [binarize_roi_mask(cw_roi)
                          for cw_roi in cropped_weighted]

        # Make binary vs weighted ROI comparison plots
        save_dir = Path(self.args["output_dir"])

        # This might be a bit fragile...
        ophys_expt_name = movie_path.split("/")[-3]
        pdf_name = f"{ophys_expt_name}_weighted_vs_binary_rois.pdf"
        pdf_savepath = save_dir / pdf_name
        pdf = PdfPages(pdf_savepath)

        data_slug = zip(cropped_weighted, cropped_binary,
                        weighted_traces, binary_traces)

        self.logger.info("Making roi comparison plots.")
        for w_roi, b_roi, w_trace, b_trace in data_slug:
            fig = plot_binarized_vs_weighted_roi(w_roi, b_roi,
                                                 w_trace, b_trace)
            pdf.savefig(fig, dpi=400)
            plt.close(fig)
        pdf.close()
        self.logger.info(f"Wrote {str(pdf_savepath)}")


if __name__ == "__main__":  # pragma: no cover
    rqrg = RoiQcReportGenerator()
    rqrg.run()

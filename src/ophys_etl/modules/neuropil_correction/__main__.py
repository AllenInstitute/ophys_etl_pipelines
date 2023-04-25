import logging
import os
import shutil

import h5py
import numpy as np
from argschema import ArgSchemaParser

from ophys_etl.modules.neuropil_correction.schemas import (
    NeuropilCorrectionJobSchema,
    NeuropilCorrectionJobOutputSchema,
)
from ophys_etl.modules.neuropil_correction.utils import (
    debug_plot,
    estimate_contamination_ratios,
    fill_unconverged_r,
)


class NeuropilCorrectionRunner(ArgSchemaParser):

    default_schema = NeuropilCorrectionJobSchema
    default_output_schema = NeuropilCorrectionJobOutputSchema

    def run(self):

        #######################################################################
        # prelude -- get processing metadata

        trace_file = self.args["roi_trace_file"]
        neuropil_file = self.args["neuropil_trace_file"]
        storage_dir = self.args["storage_directory"]

        plot_dir = os.path.join(storage_dir, "neuropil_subtraction_plots")
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)

        os.makedirs(plot_dir)

        logging.info("Neuropil correcting '%s'", trace_file)

        #######################################################################
        # process data
        try:
            roi_traces = h5py.File(trace_file, "r")
        except Exception as e:
            logging.error(
                f"Error: unable to open ROI trace file {trace_file}", e
            )
            raise

        try:
            neuropil_traces = h5py.File(neuropil_file, "r")
        except Exception as e:
            logging.error(
                f"Error: unable to open neuropil trace file {neuropil_file}", e
            )
            raise

        # get number of traces, length, etc.
        num_traces, T = roi_traces["data"].shape
        T_orig = T
        T_cross_val = int(T / 2)
        if T - T_cross_val > T_cross_val:
            T = T - 1

        # make sure that ROI and neuropil trace files are organized the same
        n_id = neuropil_traces["roi_names"][:].astype(str)
        r_id = roi_traces["roi_names"][:].astype(str)
        logging.info("Processing %d traces", len(n_id))
        assert len(n_id) == len(
            r_id
        ), "Input trace files are not aligned (ROI count)"
        for i in range(len(n_id)):
            assert (
                n_id[i] == r_id[i]
            ), "Input trace files are not aligned (ROI IDs)"

        # initialize storage variables and analysis routine
        r_array = np.empty(num_traces)
        RMSE_array = np.ones(num_traces, dtype=float) * -1
        roi_names = n_id
        corrected = np.zeros((num_traces, T_orig))
        r_vals = [None] * num_traces

        for n in range(num_traces):
            roi = roi_traces["data"][n]
            neuropil = neuropil_traces["data"][n]

            if np.any(np.isnan(neuropil)):
                logging.warning(
                    "neuropil trace for roi %d contains NaNs, skipping", n
                )
                continue

            if np.any(np.isnan(roi)):
                logging.warning(
                    "roi trace for roi %d contains NaNs, skipping", n
                )
                continue

            r = np.nan

            logging.info("Correcting trace %d (roi %s)", n, str(n_id[n]))
            results = estimate_contamination_ratios(roi, neuropil)
            logging.info(
                "r=%f err=%f it=%d",
                results["r"],
                results["err"],
                results["it"],
            )

            r = results["r"]
            fc = roi - r * neuropil
            RMSE_array[n] = results["err"]
            r_vals[n] = results["r_vals"]

            if r > 1:
                logging.info(f"Estimated r value > 1, r = {r}")
            debug_plot(
                os.path.join(plot_dir, "initial_%04d.png" % n),
                roi,
                neuropil,
                fc,
                r,
                results["r_vals"],
                results["err_vals"],
            )

            # mean of the corrected trace must be positive
            if fc.mean() > 0:
                r_array[n] = r
                corrected[n, :] = fc
            else:
                logging.warning(
                    "fc has negative baseline, skipping this r value"
                )

        # fill in empty r values
        for n in range(num_traces):
            roi = roi_traces["data"][n]
            neuropil = neuropil_traces["data"][n]

            if r_array[n] is np.nan:
                logging.warning(
                    "fc had negative baseline %d. Setting r to zero.", n
                )
                r_array[n] = 0
                corrected[n, :] = roi

            # save a debug plot
            debug_plot(
                os.path.join(plot_dir, "final_%04d.png" % n),
                roi,
                neuropil,
                corrected[n, :],
                r_array[n],
            )

            # one last sanity check
            eps = -0.0001
            if np.mean(corrected[n, :]) < eps:
                raise Exception(
                    "Trace %d baseline is still negative value after"
                    "correction" % n
                )

            if r_array[n] < 0.0:
                raise Exception("Trace %d ended with negative r" % n)

        r_array = r_array.astype(float)

        # flag cells with unconverged r values (r>1) and fill in with
        # mean r value across all other cells of the same experiment.
        # recalculate neuropil_corrected trace and RMSE
        if any(r_array > 1):
            logging.info(
                f"Number of unconverged r values > 1: {sum(r_array > 1)}"
                "Filling in unconverged r values with mean r value"
                "Recalculating corrected trace and RMSE"
            )
            corrected, r_array, RMSE_array = fill_unconverged_r(
                corrected,
                roi_traces["data"][()],
                neuropil_traces["data"][()],
                r_array,
            )

        ######################################################################
        # write out processed data
        try:
            neuropil_correction_output = os.path.join(
                storage_dir, "neuropil_correction.h5"
            )
            hf = h5py.File(neuropil_correction_output, "w")
            hf.create_dataset("r", data=r_array)
            hf.create_dataset("RMSE", data=RMSE_array)
            hf.create_dataset("FC", data=corrected, compression="gzip")
            hf.create_dataset("roi_names", data=roi_names.astype(np.string_))

            for n in range(num_traces):
                r = r_vals[n]
                if r is not None:
                    hf.create_dataset("r_vals/%d" % n, data=r)
            hf.close()
        except Exception as e:
            logging.error("Error creating output h5 file", e)
            raise

        roi_traces.close()
        neuropil_traces.close()
        self.output(
            {
                "neuropil_subtraction_plots": plot_dir,
                "neuropil_correction_trace_file": neuropil_correction_output,
            }
        )

        logging.info("finished")


if __name__ == "__main__":
    neuropil_correction_job = NeuropilCorrectionRunner()
    neuropil_correction_job.run()

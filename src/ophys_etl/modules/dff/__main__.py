import time
from argschema import ArgSchemaParser
import multiprocessing
import h5py
import numpy as np
from functools import partial
from pathlib import Path
from typing import Tuple

from ophys_etl.modules.dff.schemas import DffJobSchema, DffJobOutputSchema
from ophys_etl.utils.traces import medfilt, noise_std


def compute_dff_trace(corrected_fluorescence_trace: np.ndarray,
                      long_filter_length: int,
                      short_filter_length: int
                      ) -> Tuple[np.ndarray, float, int]:
    """
    Compute the "delta F over F" from the fluorescence trace.
    Uses configurable length median filters to compute baseline for
    baseline-subtraction and short timescale detrending.
    Returns the artifact-corrected and detrended dF/F, along with
    additional metadata for QA: the estimated standard deviation of
    the noise ("sigma_dff") and the number of frames where the
    computed baseline was less than the standard deviation of the noise.

    Parameters
    ----------
    corrected_fluorescence_trace: np.array
        1d numpy array of the neuropil-corrected fluorescence trace
    long_filter_length: int
        Length (in number of elements) of the long median filter used
        to compute a rolling baseline. Must be an odd number.
    short_filter_length: int (default=31)
        Length (in number of elements) for a short median filter used
        for short timescale detrending.
    Returns
    -------
    np.ndarray:
        The "dff" (delta_fluorescence/fluorescence) trace, 1d np.array
    float:
        The estimated standard deviation of the noise in the dff trace
    int:
        Number of frames where the baseline (long timescape median
        filter) was less than or equal to the estimated noise of the
        `corrected_fluorescence_trace`.
    """
    sigma_f = noise_std(corrected_fluorescence_trace, short_filter_length)

    # Long timescale median filter for baseline subtraction
    baseline = medfilt(corrected_fluorescence_trace, long_filter_length)
    dff = ((corrected_fluorescence_trace - baseline)
           / np.maximum(baseline, sigma_f))
    num_small_baseline_frames = np.sum(baseline <= sigma_f)

    sigma_dff = noise_std(dff, short_filter_length)

    # Short timescale detrending
    filtered_dff = medfilt(dff, short_filter_length)
    # Constrain to 2.5x the estimated noise of dff
    filtered_dff = np.minimum(filtered_dff, 2.5*sigma_dff)
    detrended_dff = dff - filtered_dff

    return detrended_dff, sigma_dff, num_small_baseline_frames


def job_call(index: int, input_file: Path, key: str,
             long_filter: int, short_filter: int):
    with h5py.File(input_file, "r") as f:
        trace = f[key][index]
    dff, sigma_dff, small_baseline = compute_dff_trace(
            trace, long_filter, short_filter)
    return dff, sigma_dff, small_baseline


class DffJob(ArgSchemaParser):
    """
    This is the job runner for the dF/F computation from F (fluorescence)
    traces. The primary data input is the h5 file produced by neuropil
    subtraction, which contains the neuropil-corrected fluorescence trace.
    (by default in the "CF" key).

    NOTE: There has historically been no data saved in the output json
    for this job in the AllenSDK, so the output json is just a
    placeholder (empty dictionary).
    """
    default_schema = DffJobSchema
    default_output_schema = DffJobOutputSchema

    def run(self):
        # Set up file and data pointers
        with h5py.File(self.args["input_file"], "r") as f:
            traces_shape = f[self.args["input_dataset"]].shape
            roi_dataset = f[self.args['roi_field']][()]
        roi_shape = roi_dataset.shape

        # Check for id mapping mismatches
        if roi_shape[0] != traces_shape[0]:
            raise ValueError(
                f"Can't associate ROIs of shape {roi_dataset.shape} "
                f"to traces of shape {traces_shape}")

        job_partial = partial(job_call,
                              input_file=Path(self.args['input_file']),
                              key=self.args["input_dataset"],
                              long_filter=self.args["long_filter_frames"],
                              short_filter=self.args["short_filter_frames"])
        with multiprocessing.Pool(self.args['n_parallel_workers']) as pool:
            results = [i for i in pool.imap(
                       job_partial, np.arange(traces_shape[0]), chunksize=10)]

        dff, sigma_dff, small_baseline = list(zip(*results))

        with h5py.File(self.args["output_file"], "w") as output_h5:
            output_h5.create_dataset(self.args["roi_field"],
                                     data=roi_dataset),
            output_h5.create_dataset(self.args["output_dataset"], data=dff)
            output_h5.create_dataset(self.args["sigma_dataset"],
                                     data=sigma_dff)
            output_h5.create_dataset(self.args["baseline_frames_dataset"],
                                     data=small_baseline)

        self.logger.info("Dff traces complete.")

        self.output({
            "output_file": self.args["output_file"],
            "created_at": int(time.time())
            }, indent=2)


if __name__ == "__main__":    # pragma: nocover
    dff_job = DffJob()
    dff_job.run()

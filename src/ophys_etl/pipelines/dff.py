# Compute deltaF/F
import time

from argschema import ArgSchema, ArgSchemaParser, fields
import multiprocessing
import h5py
from marshmallow import post_load
import numpy as np
from functools import partial
from pathlib import Path

from ophys_etl.transforms.trace_transforms import compute_dff_trace
from ophys_etl.schemas.fields import H5InputFile


class DffJobOutputSchema(ArgSchema):
    output_file = H5InputFile(
        required=True,
        description=("Path to output h5 file containing the df/f traces for "
                     "each ROI.")
    )
    created_at = fields.Int(
        required=True,
        description=("Epoch time (in seconds) that the file was created.")
    )


class DffJobSchema(ArgSchema):
    input_file = H5InputFile(
        required=True,
        description=("Input h5 file containing fluorescence traces and the "
                     "associated ROI IDs (in datasets specified by the keys "
                     "'input_dataset' and 'roi_field', respectively.")
        )
    output_file = fields.OutputFile(
        required=True,
        description="h5 file to write the results of dff computation."
        )
    movie_frame_rate_hz = fields.Float(
        required=True,
        description=("Acquisition frame rate for the trace data in "
                     "`input_dataset`")
    )
    log_level = fields.Int(
        required=False,
        default=20      # logging.INFO
        )
    input_dataset = fields.Str(
        required=False,
        default="FC",
        description="Key of h5 dataset to use from `input_file`."
    )
    roi_field = fields.Str(
        required=False,
        default="roi_names",
        description=("The h5 dataset key in both the `input_file` and "
                     "`output_file` containing ROI IDs associated with "
                     "traces.")
        )
    output_dataset = fields.Str(
        required=False,
        default="data",
        description=("h5 dataset key used to store the computed dff traces "
                     "in `output_file`.")
    )
    sigma_dataset = fields.Str(
        required=False,
        default="sigma_dff",
        description=("h5 dataset key used to store the estimated noise "
                     "standard deviation for the dff traces in `output_file`.")
    )
    baseline_frames_dataset = fields.Str(
        required=False,
        default="num_small_baseline_frames",
        description=("h5 dataset key used to store the number of small "
                     "baseline frames (where the computed baseline of the "
                     "fluorescence trace was smaller than its estimated "
                     "noise standard deviation) in `output_file`.")
    )
    long_baseline_filter_s = fields.Int(
        required=False,
        default=600,
        description=("Number of seconds to use in the rolling median "
                     "filter for for computing the baseline activity. "
                     "The length of the filter is the frame rate of the "
                     "signal in Hz * the long baseline filter seconds ("
                     "+1 if the result is even, since the median filter "
                     "length must be odd).")
    )
    short_filter_s = fields.Float(
        required=False,
        default=3.333,
        description=("Number of seconds to use in the rolling median "
                     "filter for the short timescale detrending. "
                     "The length of the filter is the frame rate of the "
                     "signal in Hz * the short baseline filter seconds ("
                     "+1 if the result is even, since the median filter "
                     "length must be odd).")
    )
    n_parallel_workers = fields.Int(
        required=False,
        default=1,
        description="number of parallel workers")

    @post_load
    def filter_s_to_frames(self, item, **kwargs):
        """Convert number of seconds to number of frames for the
        filters `short_filter_s`, `long_baseline_filter_s`. If the
        number of frames is even, add 1."""
        short_frames = int(np.round(
            item["movie_frame_rate_hz"] * item["short_filter_s"]))
        long_frames = int(np.round(
            item["movie_frame_rate_hz"] * item["long_baseline_filter_s"]))
        # Has to be odd
        item["short_filter_frames"] = (
            short_frames if short_frames % 2 else short_frames + 1)
        item["long_filter_frames"] = (
            long_frames if long_frames % 2 else long_frames + 1)
        return item


def job_call(index: int, input_file: Path, key: str,
             long_filter: int, short_filter: int):
    with h5py.File(input_file, "r") as f:
        trace = f[key][index]
    dff, sigma_dff, small_baseline = compute_dff_trace(
            trace, long_filter, short_filter)
    print(index)
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

from argschema import ArgSchema, fields
from marshmallow import post_load
import numpy as np

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

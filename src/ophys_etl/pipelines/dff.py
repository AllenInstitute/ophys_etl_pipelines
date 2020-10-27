# Compute deltaF/F
from argschema import ArgSchema, ArgSchemaParser, fields
from marshmallow import post_load
import numpy as np
import h5py

from ophys_etl.transforms.trace_transforms import compute_dff_trace


class DffJobSchema(ArgSchema):
    input_file = fields.InputFile(
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


class DffJob(ArgSchemaParser):
    """
    This is the job runner for the dF/F computation from F (fluorescence)
    traces. The primary data input is the h5 file produced by neuropil
    subtraction, which contains the "corrected fluorescence trace"
    (by default in the "CF" key).

    NOTE: There has historically been no data saved in the output json
    for this job in the AllenSDK, so the output json is just a
    placeholder (empty dictionary).
    """
    default_schema = DffJobSchema

    def run(self):
        # Set up file and data pointers
        input_h5 = h5py.File(self.args["input_file"], "r")
        output_h5 = h5py.File(self.args["output_file"], "w")
        roi_dataset = input_h5[self.args["roi_field"]]
        traces_dataset = input_h5[self.args["input_dataset"]]

        # Initialize storage
        dff_dataset = output_h5.create_dataset(
            self.args["output_dataset"], traces_dataset.shape)
        sigma_dffs = output_h5.create_dataset(
            self.args["sigma_dataset"], roi_dataset.shape)
        small_baselines = output_h5.create_dataset(
            self.args["baseline_frames_dataset"], roi_dataset.shape)
        # Copy over the roi names
        output_h5.create_dataset(self.args["roi_field"],
                                 data=input_h5[self.args["roi_field"]][()])

        # Check for id mapping mismatches
        if roi_dataset.shape[0] != traces_dataset.shape[0]:
            raise ValueError(
                f"Can't associate ROIs of shape {roi_dataset.shape} "
                f"to traces of shape {traces_dataset.shape}")

        # Run computation
        # The traces can be large, so load them into memory 'row-wise' and
        #   rely on reference counting for cleanup.
        # Default in h5 is C order storage so this should be fairly efficient
        trace_len = traces_dataset.shape[0]
        self.logger.info(f"Computing dff traces: 0/{trace_len}")
        for i in range(trace_len):
            if (i % 20 == 0) and (i != 0):
                self.logger.info(f"Working on {i}/{trace_len}...")
            trace = traces_dataset[i, :]
            dff, sigma_dff, small_baseline = compute_dff_trace(
                trace,
                self.args["long_filter_frames"],
                self.args["short_filter_frames"])
            sigma_dffs[i] = sigma_dff
            small_baselines[i] = small_baseline
            dff_dataset[i] = dff

        self.logger.info("Dff traces complete.")
        # Clean up
        output_h5.close()
        input_h5.close()

        # Output json placeholder; no info saved in SDK pipeline
        self.output({}, indent=2)


if __name__ == "__main__":    # pragma: nocover
    dff_job = DffJob()
    dff_job.run()

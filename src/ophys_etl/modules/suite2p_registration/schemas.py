import json
import tempfile
from pathlib import Path

import argschema
import marshmallow
import numpy as np
from ophys_etl.modules.suite2p_wrapper.schemas import Suite2PWrapperSchema
from ophys_etl.schemas.fields import ExistingFile, ExistingH5File


class Suite2PRegistrationInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")
    suite2p_args = argschema.fields.Nested(Suite2PWrapperSchema, required=True)
    movie_frame_rate_hz = argschema.fields.Float(
        required=True, description="frame rate of movie, usually 31Hz or 11Hz"
    )
    motion_corrected_output = argschema.fields.OutputFile(
        required=True,
        description="destination path for hdf5 motion corrected video.",
    )
    motion_diagnostics_output = argschema.fields.OutputFile(
        required=True,
        description=(
            "Desired save path for *.csv file containing motion "
            "correction offset data"
        ),
    )
    max_projection_output = argschema.fields.OutputFile(
        required=True,
        description=(
            "Desired path for *.png of the max projection of the "
            "motion corrected video."
        ),
    )
    avg_projection_output = argschema.fields.OutputFile(
        required=True,
        description=(
            "Desired path for *.png of the avg projection of the "
            "motion corrected video."
        ),
    )
    registration_summary_output = argschema.fields.OutputFile(
        required=True, description="Desired path for *.png for summary QC plot"
    )
    motion_correction_preview_output = argschema.fields.OutputFile(
        required=True, description="Desired path for *.webm motion preview"
    )
    movie_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.1,
        description=(
            "lower quantile threshold for avg projection "
            "histogram adjustment of movie"
        ),
    )
    movie_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.999,
        description=(
            "upper quantile threshold for avg projection "
            "histogram adjustment of movie"
        ),
    )
    preview_frame_bin_seconds = argschema.fields.Float(
        required=False,
        default=2.0,
        description=(
            "before creating the webm, the movies will be "
            "aveaged into bins of this many seconds."
        ),
    )
    preview_playback_factor = argschema.fields.Float(
        required=False,
        default=10.0,
        description=(
            "the preview movie will playback at this factor "
            "times real-time."
        ),
    )
    outlier_detrend_window = argschema.fields.Float(
        required=False,
        default=3.0,
        description=(
            "for outlier rejection in the xoff/yoff outputs "
            "of suite2p, the offsets are first de-trended "
            "with a median filter of this duration [seconds]. "
            "This value is ~30 or 90 samples in size for 11 and 31"
            "Hz sampling rates respectively."
        ),
    )
    outlier_maxregshift = argschema.fields.Float(
        required=False,
        default=0.05,
        description=(
            "units [fraction FOV dim]. After median-filter "
            "detrending, outliers more than this value are "
            "clipped to this value in x and y offset, independently."
            "This is similar to Suite2P's internal maxregshift, but"
            "allows for low-frequency drift. Default value of 0.05 "
            "is typically clipping outliers to 512 * 0.05 = 25 "
            "pixels above or below the median trend."
        ),
    )
    clip_negative = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "Whether or not to clip negative pixel "
            "values in output. Because the pixel values "
            "in the raw  movies are set by the current "
            "coming off a photomultiplier tube, there can "
            "be pixels with negative values (current has a "
            "sign), possibly due to noise in the rig. "
            "Some segmentation algorithms cannot handle "
            "negative values in the movie, so we have this "
            "option to artificially set those pixels to zero."
        ),
    )
    max_reference_iterations = argschema.fields.Int(
        required=False,
        default=8,
        description="Maximum number of iterations to preform when creating a "
        "reference image.",
    )
    auto_remove_empty_frames = argschema.fields.Boolean(
        required=False,
        default=True,
        allow_none=False,
        description="Automatically detect empty noise frames at the start and "
        "end of the movie. Overrides values set in "
        "trim_frames_start and trim_frames_end. Some movies "
        "arrive with otherwise quality data but contain a set of "
        "frames that are empty and contain pure noise. When "
        "processed, these frames tend to receive "
        "large random shifts that throw off motion border "
        "calculation. Turning on this setting automatically "
        "detects these frames before processing and removes them "
        "from reference image creation,  automated smoothing "
        "parameter searches, and finally the motion border "
        "calculation. The frames are still written however any "
        "shift estimated is removed and their shift is set to 0 "
        "to avoid large motion borders.",
    )
    trim_frames_start = argschema.fields.Int(
        required=False,
        default=0,
        allow_none=False,
        description="Number of frames to remove from the start of the movie "
        "if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still "
        "written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and "
        "trim_frames_start > 0",
    )
    trim_frames_end = argschema.fields.Int(
        required=False,
        default=0,
        allow_none=False,
        description="Number of frames to remove from the end of the movie "
        "if known. Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still "
        "written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and "
        "trim_frames_start > 0",
    )
    do_optimize_motion_params = argschema.fields.Bool(
        default=False,
        required=False,
        description="Do a search for best parameters of smooth_sigma and "
        "smooth_sigma_time. Adds significant runtime cost to "
        "motion correction and should only be run once per "
        "experiment with the resulting parameters being stored "
        "for later use.",
    )
    use_ave_image_as_reference = argschema.fields.Bool(
        default=False,
        required=False,
        description="Only available if `do_optimize_motion_params` is set. "
        "After the a best set of smoothing parameters is found, "
        "use the resulting average image as the reference for the "
        "full registration. This can be used as two step "
        "registration by setting by setting "
        "smooth_sigma_min=smooth_sigma_max and "
        "smooth_sigma_time_min=smooth_sigma_time_max and "
        "steps=1.",
    )
    n_batches = argschema.fields.Int(
        default=20,
        required=False,
        description="Number of batches of size suite2p_args['batch_size'] to "
        "load from the movie for smoothing parameter testing. "
        "Batches are evenly spaced throughout the movie.",
    )
    smooth_sigma_min = argschema.fields.Float(
        default=0.65,
        required=False,
        description="Minimum value of the parameter search for smooth_sigma.",
    )
    smooth_sigma_max = argschema.fields.Float(
        default=2.15,
        required=False,
        description="Maximum value of the parameter search for smooth_sigma.",
    )
    smooth_sigma_steps = argschema.fields.Int(
        default=4,
        required=False,
        description="Number of steps to grid between smooth_sigma and "
        "smooth_sigma_max. Large values will add significant time "
        "motion correction.",
    )
    smooth_sigma_time_min = argschema.fields.Float(
        default=0,
        required=False,
        description="Minimum value of the parameter search for "
        "smooth_sigma_time.",
    )
    smooth_sigma_time_max = argschema.fields.Float(
        default=6,
        required=False,
        description="Maximum value of the parameter search for "
        "smooth_sigma_time.",
    )
    smooth_sigma_time_steps = argschema.fields.Int(
        default=7,
        required=False,
        description="Number of steps to grid between smooth_sigma and "
        "smooth_sigma_time_max. Large values will add significant "
        "time motion correction.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpdir = None

    @marshmallow.pre_load
    def setup_default_suite2p_args(self, data: dict, **kwargs) -> dict:
        data["suite2p_args"]["log_level"] = data["log_level"]
        data["suite2p_args"]["roidetect"] = False
        data["suite2p_args"]["do_registration"] = 1
        data["suite2p_args"]["reg_tif"] = True
        data["suite2p_args"]["retain_files"] = ["*.tif", "ops.npy"]
        if "output_dir" not in data["suite2p_args"]:
            # send suite2p results to a temporary directory
            # the results of this pipeline will be formatted versions anyway
            if "tmp_dir" in data["suite2p_args"]:
                parent_dir = data["suite2p_args"]["tmp_dir"]
            else:
                parent_dir = None
            self.tmpdir = tempfile.TemporaryDirectory(dir=parent_dir)
            data["suite2p_args"]["output_dir"] = self.tmpdir.name
        if "output_json" not in data["suite2p_args"]:
            Suite2p_output = (
                Path(data["suite2p_args"]["output_dir"])
                / "Suite2P_output.json"
            )
            data["suite2p_args"]["output_json"] = str(Suite2p_output)
        return data

    @marshmallow.pre_load
    def check_movie_frame_rate(self, data, **kwargs):
        """
        Make sure that if movie_frame_rate_hz is specified in both
        the parent set of args and in suite2p_args, the values agree.

        If suite2p_args['movie_frame_rate_hz'] is not set, set it from
        self.args['movie_frame_rate_hz']
        """
        parent_val = data["movie_frame_rate_hz"]

        if "movie_frame_rate_hz" in data["suite2p_args"]:
            if data["suite2p_args"]["movie_frame_rate_hz"] is not None:
                s2p_val = data["suite2p_args"]["movie_frame_rate_hz"]
                if np.abs(s2p_val - parent_val) > 1.0e-10:
                    msg = "Specified two values of movie_frame_rate_hz in\n"
                    msg += json.dumps(data, indent=2, sort_keys=True)
                    raise ValueError(msg)

        data["suite2p_args"]["movie_frame_rate_hz"] = parent_val
        return data

    @marshmallow.post_load
    def check_trim_frames(self, data, **kwargs):
        """Make sure that if the user sets auto_remove_empty_frames
        and timing frames is already requested, raise an error.
        """
        if data["auto_remove_empty_frames"] and (
            data["trim_frames_start"] > 0 or data["trim_frames_end"] > 0
        ):
            msg = (
                "Requested auto_remove_empty_frames but "
                "trim_frames_start > 0 or trim_frames_end > 0. Please "
                "either request auto_remove_empty_frames or manually set "
                "trim_frames_start/trim_frames_end if number of frames to "
                "trim is known."
            )
            raise ValueError(msg)
        return data


class Suite2PRegistrationOutputSchema(argschema.schemas.DefaultSchema):
    motion_corrected_output = ExistingH5File(
        required=True,
        description="destination path for hdf5 motion corrected video.",
    )
    motion_diagnostics_output = ExistingFile(
        required=True,
        description=(
            "Path of *.csv file containing motion correction offsets"
        ),
    )
    max_projection_output = ExistingFile(
        required=True,
        description=(
            "Desired path for *.png of the max projection of the "
            "motion corrected video."
        ),
    )
    avg_projection_output = ExistingFile(
        required=True,
        description=(
            "Desired path for *.png of the avg projection of the "
            "motion corrected video."
        ),
    )
    registration_summary_output = ExistingFile(
        required=True, description="Desired path for *.png for summary QC plot"
    )
    motion_correction_preview_output = ExistingFile(
        required=True, description="Desired path for *.webm motion preview"
    )

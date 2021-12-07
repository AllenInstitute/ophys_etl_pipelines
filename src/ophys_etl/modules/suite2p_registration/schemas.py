import argschema
import marshmallow
import numpy as np
import json
import tempfile
from pathlib import Path

from ophys_etl.modules.suite2p_wrapper.schemas import Suite2PWrapperSchema
from ophys_etl.schemas.fields import ExistingFile, ExistingH5File


class Suite2PRegistrationInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")
    suite2p_args = argschema.fields.Nested(Suite2PWrapperSchema,
                                           required=True)
    movie_frame_rate_hz = argschema.fields.Float(
        required=True,
        description="frame rate of movie, usually 31Hz or 11Hz")
    motion_corrected_output = argschema.fields.OutputFile(
        required=True,
        description="destination path for hdf5 motion corrected video.")
    motion_diagnostics_output = argschema.fields.OutputFile(
        required=True,
        description=("Desired save path for *.csv file containing motion "
                     "correction offset data")
    )
    max_projection_output = argschema.fields.OutputFile(
        required=True,
        description=("Desired path for *.png of the max projection of the "
                     "motion corrected video."))
    avg_projection_output = argschema.fields.OutputFile(
        required=True,
        description=("Desired path for *.png of the avg projection of the "
                     "motion corrected video."))
    registration_summary_output = argschema.fields.OutputFile(
        required=True,
        description="Desired path for *.png for summary QC plot")
    motion_correction_preview_output = argschema.fields.OutputFile(
        required=True,
        description="Desired path for *.webm motion preview")
    movie_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.1,
        description=("lower quantile threshold for avg projection "
                     "histogram adjustment of movie"))
    movie_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.999,
        description=("upper quantile threshold for avg projection "
                     "histogram adjustment of movie"))
    preview_frame_bin_seconds = argschema.fields.Float(
        required=False,
        default=2.0,
        description=("before creating the webm, the movies will be "
                     "aveaged into bins of this many seconds."))
    preview_playback_factor = argschema.fields.Float(
        required=False,
        default=10.0,
        description=("the preview movie will playback at this factor "
                     "times real-time."))
    outlier_detrend_window = argschema.fields.Float(
        required=False,
        default=3.0,
        description=("for outlier rejection in the xoff/yoff outputs "
                     "of suite2p, the offsets are first de-trended "
                     "with a median filter of this duration [seconds]. "
                     "This value is ~30 or 90 samples in size for 11 and 31"
                     "Hz sampling rates respectively."))
    outlier_maxregshift = argschema.fields.Float(
        required=False,
        default=0.05,
        description=("units [fraction FOV dim]. After median-filter "
                     "detrending, outliers more than this value are "
                     "clipped to this value in x and y offset, independently."
                     "This is similar to Suite2P's internal maxregshift, but"
                     "allows for low-frequency drift. Default value of 0.05 "
                     "is typically clipping outliers to 512 * 0.05 = 25 "
                     "pixels above or below the median trend."))
    clip_negative = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=("Whether or not to clip negative pixel "
                     "values in output. Because the pixel values "
                     "in the raw  movies are set by the current "
                     "coming off a photomultiplier tube, there can "
                     "be pixels with negative values (current has a "
                     "sign), possibly due to noise in the rig. "
                     "Some segmentation algorithms cannot handle "
                     "negative values in the movie, so we have this "
                     "option to artificially set those pixels to zero."))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpdir = None

    @marshmallow.pre_load
    def setup_default_suite2p_args(self, data: dict, **kwargs) -> dict:
        data["suite2p_args"]["log_level"] = data["log_level"]
        data['suite2p_args']['roidetect'] = False
        data['suite2p_args']['do_registration'] = 1
        data['suite2p_args']['reg_tif'] = True
        data['suite2p_args']['retain_files'] = ["*.tif", "ops.npy"]
        if "output_dir" not in data["suite2p_args"]:
            # send suite2p results to a temporary directory
            # the results of this pipeline will be formatted versions anyway
            if 'tmp_dir' in data['suite2p_args']:
                parent_dir = data['suite2p_args']['tmp_dir']
            else:
                parent_dir = None
            self.tmpdir = tempfile.TemporaryDirectory(dir=parent_dir)
            data["suite2p_args"]["output_dir"] = self.tmpdir.name
        if "output_json" not in data["suite2p_args"]:
            Suite2p_output = (Path(data["suite2p_args"]["output_dir"])
                              / "Suite2P_output.json")
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
        parent_val = data['movie_frame_rate_hz']

        if 'movie_frame_rate_hz' in data['suite2p_args']:
            if data['suite2p_args']['movie_frame_rate_hz'] is not None:
                s2p_val = data['suite2p_args']['movie_frame_rate_hz']
                if np.abs(s2p_val-parent_val) > 1.0e-10:
                    msg = 'Specified two values of movie_frame_rate_hz in\n'
                    msg += json.dumps(data, indent=2, sort_keys=True)
                    raise ValueError(msg)

        data['suite2p_args']['movie_frame_rate_hz'] = parent_val
        return data


class Suite2PRegistrationOutputSchema(argschema.schemas.DefaultSchema):
    motion_corrected_output = ExistingH5File(
        required=True,
        description="destination path for hdf5 motion corrected video.")
    motion_diagnostics_output = ExistingFile(
        required=True,
        description=("Path of *.csv file containing motion correction offsets")
    )
    max_projection_output = ExistingFile(
        required=True,
        description=("Desired path for *.png of the max projection of the "
                     "motion corrected video."))
    avg_projection_output = ExistingFile(
        required=True,
        description=("Desired path for *.png of the avg projection of the "
                     "motion corrected video."))
    registration_summary_output = ExistingFile(
        required=True,
        description="Desired path for *.png for summary QC plot")
    motion_correction_preview_output = ExistingFile(
        required=True,
        description="Desired path for *.webm motion preview")

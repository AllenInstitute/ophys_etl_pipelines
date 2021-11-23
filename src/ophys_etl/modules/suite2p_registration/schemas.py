import argschema
import marshmallow
import tempfile
from pathlib import Path

from ophys_etl.modules.module_abc.module_abc import OphysEtlBaseSchema

from ophys_etl.modules.suite2p_wrapper.schemas import Suite2PWrapperSchema
from ophys_etl.schemas.fields import ExistingFile, ExistingH5File


class Suite2PRegistrationInputSchema(OphysEtlBaseSchema):
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

    metadata_field = argschema.fields.String(
            default='motion_corrected_output',
            required=True,
            allow_none=False,
            description=("Field point to file, either JSON or HDF5, "
                         "where metadata gets written"))

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
            self.tmpdir = tempfile.TemporaryDirectory()
            data["suite2p_args"]["output_dir"] = self.tmpdir.name
        if "output_json" not in data["suite2p_args"]:
            Suite2p_output = (Path(data["suite2p_args"]["output_dir"])
                              / "Suite2P_output.json")
            data["suite2p_args"]["output_json"] = str(Suite2p_output)
        # we are not doing registration here, but the wrapper schema wants
        # a value:
        data['suite2p_args']['nbinned'] = 1000
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

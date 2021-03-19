import argschema
import marshmallow as mm

from ophys_etl.modules.suite2p_wrapper.utils import Suite2PWrapperException


class Suite2PWrapperSchema(argschema.ArgSchema):
    """
    s2p parameter names are copied from:
    https://github.com/MouseLand/suite2p/blob/master/suite2p/run_s2p.py
    descriptions here should indicate why certain default setting
    choices have been made for Allen production.
    """

    # s2p IO settings (file paths)
    h5py = argschema.fields.InputFile(
            required=True,
            description=("Path to input video. In Allen production case, "
                         "assumed to be motion-corrected."))
    h5py_key = argschema.fields.Str(
            required=False,
            missing="data",
            default="data",
            description="key in h5py where data array is stored")
    data_path = argschema.fields.List(
            argschema.fields.Str,
            cli_as_single_argument=True,
            required=False,
            default=[],
            description=("Allen production specifies h5py as the source of "
                         "the data, but Suite2P still wants this key in the "
                         "args."))
    # s2p registration settings
    do_registration = argschema.fields.Int(
            default=0,
            description=("0 skips registration (not well-documented). In "
                         "Allen production case, we are providing motion-"
                         "corrected videos, so we skip Suite2P's own "
                         "registration."))
    reg_tif = argschema.fields.Bool(
            default=False,
            description="whether to save registered tiffs")
    maxregshift = argschema.fields.Float(
            default=0.2,
            description=("max allowed registration shift, as a fraction of "
                         "frame max(width and height)"))
    two_step_registration = argschema.fields.Bool(
            default=False,
            description=("whether or not to run registration twice (for low "
                         "SNR data). keep_movie_raw must be True for "
                         "this to work."))
    # s2p cell detection settings
    roidetect = argschema.fields.Bool(
            default=True,
            description=("Whether or not to run ROI extraction. This is the "
                         "main role of Suite2P in Allen production."))
    sparse_mode = argschema.fields.Bool(
            default=True,
            description=("From conversation with authors, True is the "
                         "preferred mode."))
    diameter = argschema.fields.Int(
            default=12,
            description=("If not sparse_mode, use diameter (presumably "
                         "in pixels) for filtering and extracting."))
    spatial_scale = argschema.fields.Int(
            default=0,
            description=("0: multi-scale; 1: 6 pix; 2: 12 pix;"
                         "3: 24 pix; 4: 48 pix. From conversation with "
                         "authors, 0 is the preferred mode."))
    connected = argschema.fields.Bool(
            default=True,
            description=("Whether to keep ROIs fully connected. This is "
                         "Suite2P default."))
    nbinned = argschema.fields.Int(
            required=False,
            description=("Max num of binned frames for cell detection. "
                         "Below are Allen-specific parameters "
                         "`bin_duration + movie_frame_rate` "
                         "from which this setting can be derived, if "
                         "not provided."))
    max_iterations = argschema.fields.Int(
            default=20,
            description="Max num iterations to detect cells. Suite2P default")
    threshold_scaling = argschema.fields.Float(
            default=0.75,
            description=("Adjust automatically determined threshold by this "
                         "scalar multiplier. From an extensive validation "
                         "effort, we found that 0.75 segmented expected cells "
                         "but higher values tended to miss some. A fair "
                         "number of non-cells also are segmented, and need "
                         "to be filtered by a classifier."))
    max_overlap = argschema.fields.Float(
            default=0.75,
            description=("Cells with more overlap than this get removed "
                         "during triage, before refinement. Suite2P default."))
    high_pass = argschema.fields.Int(
            default=100,
            description=("Running mean subtraction with window of "
                         "size 'high_pass'. Suite2P default."))
    smooth_masks = argschema.fields.Bool(
            default=True,
            description=("Whether to smooth masks in final pass of cell "
                         "detection. Especially useful if one is in a high "
                         "noise regime. It appears this is only applicable "
                         "with sparse_mode=False, which is not our default."))
    # s2p ROI extraction parameters
    inner_neuropil_radius = argschema.fields.Int(
            default=2,
            description=("Number of pixels to keep between ROI and neuropil "
                         "donut. "
                         "Allen production will run it's own neuropil "
                         "subtraction routine."))
    min_neuropil_pixels = argschema.fields.Int(
            default=350,
            description=("Minimum number of pixels in the neuropil."
                         "Allen production will run it's own neuropil "
                         "subtraction routine."))
    allow_overlap = argschema.fields.Bool(
            # NOTE: changing this from extraction validation and labeling
            # will be untangled downstream in demixing.
            default=True,
            description=("Pixels that are overlapping are thrown out (False) "
                         "or added to both ROIs (True). Allen production will "
                         "run a demixing step that should allow both ROIs "
                         "to share pixels."))
    # Allen-specific options
    movie_frame_rate = argschema.fields.Float(
            required=False,
            description=("The frame rate (in Hz) of the optical physiology "
                         "movie to be Suite2P segmented. Used in conjunction "
                         "with 'bin_duration' to derive an 'nbinned' "
                         "Suite2P value."))
    bin_duration = argschema.fields.Float(
            required=False,
            default=3.7,
            description=("The duration of time (in seconds) that should be "
                         "considered 1 bin for Suite2P ROI detection "
                         "purposes. Requires a valid value for "
                         "'movie_frame_rate' in order to derive an "
                         "'nbinned' Suite2P value. This allows "
                         "consistent temporal downsampling across movies "
                         "with different lengths and/or frame rates. By "
                         "default, 3.7 seconds."))
    output_dir = argschema.fields.OutputDir(
            required=True,
            description="for minimal and cleaner output, specifies output dir")
    retain_files = argschema.fields.List(
            argschema.fields.Str,
            cli_as_single_argument=True,
            required=True,
            default=['stat.npy', '*.tif'],
            description=("only Suite2P output files with basenames in this "
                         "list will be retained. If 'all', a complete list "
                         "will be retained."))
    timestamp = argschema.fields.Bool(
            default=True,
            description=("if true, will add a timestamp to the output file "
                         "names. "
                         "<basename>.<ext> -> <basename>_<timestamp>.<ext>"))

    @mm.post_load
    def check_args(self, data, **kwargs):
        if ('nbinned' not in data) & ('movie_frame_rate' not in data):
            raise Suite2PWrapperException(
                    "Must provide either `nbinned` or `movie_frame_rate`")
        return data

    @mm.post_load
    def check_retain_files(self, data, **kwargs):
        if 'all' in data['retain_files']:
            data['retain_files'] = [
                    'ops1.npy', 'data.bin', 'Fneu.npy', 'F.npy', 'iscell.npy',
                    'ops.npy', 'spks.npy', 'stat.npy']
        return data


class ExistingFile(argschema.fields.InputFile):
    pass


class Suite2PWrapperOutputSchema(argschema.schemas.DefaultSchema):
    output_files = argschema.fields.Dict(
        keys=argschema.fields.Str,
        values=argschema.fields.List(ExistingFile),
        required=True,
        description="retained output files from Suite2P")

import argschema
import suite2p
import pathlib
import marshmallow as mm
import h5py
import tempfile
from typing import List, Optional
import datetime
import shutil


class Suite2PWrapperException(Exception):
    pass


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
    # s2p registration settings
    do_registration = argschema.fields.Int(
            default=0,
            description=("0 skips registration (not well-documented). In "
                         "Allen production case, we are providing motion-"
                         "corrected videos, so we skip Suite2P's own "
                         "registration."))
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
            description=("Max num of binned frames for cell detection. Below "
                         "is an Allen-specific parameter `bin_size` from "
                         "which this setting is derived, if not provided."))
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
    bin_size = argschema.fields.Int(
            required=False,
            description=("If nbinned not provided, will calculate nbinned as "
                         "nframes / bin_size. This allows consistent temporal "
                         "downsampling across movies of different lengths."))
    output_dir = argschema.fields.OutputDir(
            required=True,
            description="for minimal and cleaner output, specifies output dir")
    retain_files = argschema.fields.List(
            argschema.fields.Str,
            cli_as_single_argument=True,
            required=True,
            default=['stat.npy'],
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
        if ('nbinned' not in data) & ('bin_size' not in data):
            raise Suite2PWrapperException(
                    "must provide either `nbinned` or `bin_size`")
        return data

    @mm.post_load
    def check_retain_files(self, data, **kwargs):
        if 'all' in data['retain_files']:
            data['retain_files'] = [
                    'ops1.npy', 'data.bin', 'Fneu.npy', 'F.npy', 'iscell.npy',
                    'ops.npy', 'spks.npy', 'stat.npy']
        return data


class Suite2PWrapperOutputSchema(argschema.schemas.DefaultSchema):
    output_files = argschema.fields.Dict(
        keys=argschema.fields.Str,
        values=argschema.fields.InputFile,
        required=True,
        description="retained output files from Suite2P")


def copy_and_add_uid(
        srcdir: pathlib.Path, dstdir: pathlib.Path,
        basenames: List[str], uid: Optional[str] = None) -> List[str]:
    """copy files matching basenames from a tree search of srcdir to
    dstdir with an optional unique id inserted into the basename.

    Parameters
    ----------
    srcdir : pathlib.Path
       source directory
    dstdir : pathlib.Path
       destination directory
    basenames : list
        list of basenames to copy
    uid : str
        uid to insert into basename (example a timestamp string)

    Returns
    -------
    copied_files : dict
        keys are basenames and vaues are output paths as strings

    """

    copied_files = {}

    for basename in basenames:
        result = list(srcdir.rglob(basename))
        if len(result) != 1:
            raise ValueError(f"{len(result)} matches found in {srcdir} "
                             f"for {basename}. Expected 1 match.")
        dstbasename = result[0].name
        if uid is not None:
            dstbasename = result[0].stem + f"_{uid}" + result[0].suffix

        dstfile = dstdir / dstbasename

        shutil.copyfile(result[0], dstfile)

        copied_files[basename] = str(dstfile)

    return copied_files


class Suite2PWrapper(argschema.ArgSchemaParser):
    default_schema = Suite2PWrapperSchema
    default_output_schema = Suite2PWrapperOutputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # determine nbinned from bin_size
        if 'nbinned' not in self.args:
            with h5py.File(self.args['h5py'], 'r') as f:
                nframes = f['data'].shape[0]
            self.args['nbinned'] = int(nframes / self.args['bin_size'])
            self.logger.info(f"movie has {nframes} frames. Setting nbinned "
                             f"to {self.args['nbinned']}")

        # make a tempdir for Suite2P's output
        with tempfile.TemporaryDirectory() as tdir:
            self.args['save_path0'] = tdir
            self.logger.info(f"Running Suite2P with output going to {tdir}")
            suite2p.run_s2p.run_s2p(self.args)

            self.logger.info(f"Suite2P complete. Copying output from {tdir} "
                             f"to {self.args['output_dir']}")
            # copy over specified retained files to the output dir
            odir = pathlib.Path(self.args['output_dir'])
            odir.mkdir(parents=True, exist_ok=True)
            self.now = None
            if self.args['timestamp']:
                self.now = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            output_files = copy_and_add_uid(
                    pathlib.Path(tdir),
                    odir,
                    self.args['retain_files'],
                    self.now)
            for k, v in output_files.items():
                self.logger.info(f"wrote {k} to {v}")

        outdict = {
                'output_files': output_files
                }
        self.output(outdict, indent=2)


if __name__ == "__main__":  # pragma: no cover
    s2pw = Suite2PWrapper()
    s2pw.run()

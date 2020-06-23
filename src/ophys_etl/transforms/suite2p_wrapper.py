import argschema
import suite2p
import pathlib
import marshmallow as mm
import h5py
import tempfile
from typing import List
import datetime
import shutil


class Suite2PWrapperException(Exception):
    pass


class Suite2PWrapperSchema(argschema.ArgSchema):
    # Options for running s2p. One list can be found at:
    # https://github.com/MouseLand/suite2p/blob/master/suite2p/run_s2p.py

    # s2p IO settings (file paths)
    h5py = argschema.fields.InputFile(
            required=True,
            description="Path to 2P motion corrected hdf5 data path.")
    h5py_key = argschema.fields.Str(
            required=False,
            missing="data",
            default="data",
            description="hdf5 file key for motion corrected dataset.")
    # s2p registration settings
    do_registration = argschema.fields.Int(
            default=0,
            description="0 skips registration, as we are doing our own.")
    # s2p cell detection settings
    roidetect = argschema.fields.Bool(
            default=True,
            description="Whether or not to run ROI extraction.")
    sparse_mode = argschema.fields.Bool(
            default=True,
            description="Whether or not to run 'sparse mode'.")
    diameter = argschema.fields.Int(
            default=12,
            description=("If not sparse_mode, use diameter (presumably "
                         "in pixels) for filtering and extracting."))
    spatial_scale = argschema.fields.Int(
            default=0,
            description=("0: multi-scale; 1: 6 pix; 2: 12 pix;"
                         "3: 24 pix; 4: 48 pix"))
    connected = argschema.fields.Bool(
            default=True,
            description="Whether to keep ROIs fully connected.")
    nbinned = argschema.fields.Int(
            required=False,
            description="Max num of binned frames for cell detection.")
    max_iterations = argschema.fields.Int(
            default=20,
            description="Max num iterations to detect cells.")
    threshold_scaling = argschema.fields.Float(
            default=0.75,
            description=("Adjust automatically determined threshold by this "
                         "scalar multiplier"))
    max_overlap = argschema.fields.Float(
            default=0.75,
            description=("Cells with more overlap than this get removed "
                         "during triage, before refinement"))
    high_pass = argschema.fields.Int(
            default=100,
            description=("Running mean subtraction with window of "
                         "size 'high_pass'"))
    smooth_masks = argschema.fields.Bool(
            default=True,
            description=("Whether to smooth masks in final pass of cell "
                         "detection. Especially useful if one is in a high "
                         "noise regime."))
    # s2p ROI extraction parameters
    inner_neuropil_radius = argschema.fields.Int(
            default=2,
            description=("Number of pixels to keep between ROI and neuropil "
                         "donut."))
    min_neuropil_pixels = argschema.fields.Int(
            default=350,
            description=("Minimum number of pixels in the neuropil."))
    allow_overlap = argschema.fields.Bool(
            # NOTE: changing this from extraction validation and labeling
            # will be untangled downstream in demixing.
            default=True,
            description=("Pixels that are overlapping are thrown out (False) "
                         "or added to both ROIs (True)"))
    # Allen-specific options
    bin_size = argschema.fields.Int(
            required=False,
            description=("If nbinned not provided, will calculate nbinned as "
                         "nframes / bin_size"))
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
                         "names"))

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
        values=argschema.fields.OutputFile,
        required=True,
        description="retained output files from Suite2P")


def copy_and_add_uid(srcdir: pathlib.Path, dstdir: pathlib.Path,
                     basenames: List[str], uid: str = None) -> List[str]:
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
            suite2p.run_s2p.run_s2p(self.args)

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

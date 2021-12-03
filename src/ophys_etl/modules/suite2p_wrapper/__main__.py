import datetime
import os
import pathlib
import tempfile

import argschema
import h5py
import suite2p

from ophys_etl.modules.suite2p_wrapper import utils
from ophys_etl.modules.suite2p_wrapper import schemas


class Suite2PWrapper(argschema.ArgSchemaParser):
    default_schema = schemas.Suite2PWrapperSchema
    default_output_schema = schemas.Suite2PWrapperOutputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))

        # Should always exist as either a valid SHA or "unknown build"
        # if running in docker container.
        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA",
                                              "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")

        # determine nbinned from bin_duration and movie_frame_rate
        if 'nbinned' not in self.args:
            with h5py.File(self.args['h5py'], 'r') as f:
                nframes = f['data'].shape[0]
            bin_size = (self.args['bin_duration']
                        * self.args['movie_frame_rate'])

            if bin_size > nframes:
                raise utils.Suite2PWrapperException(
                    f"The desired frame bin duration "
                    f"({self.args['bin_duration']} "
                    f"seconds) and movie frame rate "
                    f"({self.args['movie_frame_rate']} Hz) results in a bin "
                    f"size ({bin_size} frames) larger than the number of "
                    f"actual frames in the movie ({nframes})!")

            self.args['nbinned'] = int(nframes / bin_size)
            self.logger.info(f"Movie has {nframes} frames collected at "
                             f"{self.args['movie_frame_rate']} Hz. To get a "
                             f"bin duration of {self.args['bin_duration']} "
                             f"seconds, setting nbinned to "
                             f"{self.args['nbinned']}.")

        # make a tempdir for Suite2P's output
        with tempfile.TemporaryDirectory(prefix=self.args['tmp_dir']) as tdir:
            self.args['save_path0'] = tdir
            self.logger.info(f"Running Suite2P with output going to {tdir}")
            suite2p.run_s2p(self.args)

            self.logger.info(f"Suite2P complete. Copying output from {tdir} "
                             f"to {self.args['output_dir']}")
            # copy over specified retained files to the output dir
            odir = pathlib.Path(self.args['output_dir'])
            odir.mkdir(parents=True, exist_ok=True)
            self.now = None
            if self.args['timestamp']:
                self.now = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            output_files = utils.copy_and_add_uid(
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

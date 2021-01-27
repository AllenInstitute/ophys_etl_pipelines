import h5py
import argschema
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from ophys_etl.schemas.fields import H5InputFile

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402


class RegistrationQCException(Exception):
    pass


class RegistrationQCInputSchema(argschema.ArgSchema):
    motion_corrected_path = H5InputFile(
        required=False,
        description=("path to motion corrected movie."))
    motion_diagnostics_output = argschema.fields.InputFile(
        required=True,
        description=("Saved path for *.csv file containing motion "
                     "correction offset data")
    )
    max_projection_path = argschema.fields.InputFile(
        required=True,
        description=("Saved path for *.png of the max projection of the "
                     "motion corrected video."))
    avg_projection_path = argschema.fields.InputFile(
        required=True,
        description=("Saved path for *.png of the avg projection of the "
                     "motion corrected video."))
    png_output_path = argschema.fields.OutputFile(
        required=True,
        description=("Desired path for *.png summary plot."))


def make_png(max_proj_path: Path, avg_proj_path: Path,
             summary_df: pd.DataFrame, dst_path: Path):
    """
    """
    xo = np.abs(summary_df['x']).max()
    yo = np.abs(summary_df['y']).max()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 4)
    mx_ax = fig.add_subplot(gs[0:2, 0:2])
    av_ax = fig.add_subplot(gs[0:2, 2:4])
    xyax = fig.add_subplot(gs[2, :])
    corrax = fig.add_subplot(gs[3, :])

    for ax, im_path in zip([mx_ax, av_ax], [max_proj_path, avg_proj_path]):
        with Image.open(im_path) as im:
            ax.imshow(im, cmap='gray')
            sz = im.size
        ax.axvline(xo, color='r', linestyle='--')
        ax.axvline(sz[1] - xo, color='r', linestyle='--')
        ax.axhline(yo, color='g', linestyle='--')
        ax.axhline(sz[0] - yo, color='g', linestyle='--')
        ax.set_title(f"{im_path.parent}\n{im_path.name}", fontsize=8)

    xyax.plot(summary_df['x'], linewidth=0.5, color='r', label='xoff')
    xyax.axhline(xo, color='r', linestyle='--')
    xyax.axhline(-xo, color='r', linestyle='--')
    xyax.plot(summary_df['y'], color='g', linewidth=0.5, alpha=0.5,
              label='yoff')
    xyax.axhline(yo, color='g', linestyle='--')
    xyax.axhline(-yo, color='g', linestyle='--')
    xyax.legend(loc=0)
    xyax.set_ylabel("correction offset [pixels]")

    corrax.plot(summary_df['correlation'], color='k', linewidth=0.5,
                label='corrXY')
    corrax.set_xlabel("frame index")
    corrax.set_ylabel("correlation peak value")
    corrax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(dst_path)

    return dst_path


class RegistrationQC(argschema.ArgSchemaParser):
    default_schema = RegistrationQCInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args['log_level'])

        motion_offset_df = pd.read_csv(self.args['motion_diagnostics_output'])
        png_out_path = make_png(Path(self.args['max_projection_path']),
                                Path(self.args['avg_projection_path']),
                                motion_offset_df,
                                Path(self.args['png_output_path']))
        self.logger.info(f"wrote {png_out_path}")

        return


if __name__ == "__main__":
    rqc = RegistrationQC()
    rqc.run()

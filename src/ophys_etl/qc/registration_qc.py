import argschema
import pandas as pd
import numpy as np
from functools import partial
from PIL import Image
from pathlib import Path
from ophys_etl.schemas.fields import H5InputFile
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.utils.video_utils import downsample_h5_video, encode_video

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402


class RegistrationQCException(Exception):
    pass


class RegistrationQCInputSchema(argschema.ArgSchema):
    movie_frame_rate_hz = argschema.fields.Float(
        required=True,
        description="frame rate of movie, usually 31Hz or 11Hz")
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
    uncorrected_path = H5InputFile(
        required=True,
        description=("path to uncorrected original movie."))
    motion_corrected_output = H5InputFile(
        required=True,
        description=("path to motion corrected movie."))
    motion_diagnostics_output = argschema.fields.InputFile(
        required=True,
        description=("Saved path for *.csv file containing motion "
                     "correction offset data")
    )
    max_projection_output = argschema.fields.InputFile(
        required=True,
        description=("Saved path for *.png of the max projection of the "
                     "motion corrected video."))
    avg_projection_output = argschema.fields.InputFile(
        required=True,
        description=("Saved path for *.png of the avg projection of the "
                     "motion corrected video."))
    registration_summary_output = argschema.fields.OutputFile(
        required=True,
        description=("Desired path for *.png summary plot."))
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
        ax.axvline(sz[0] - xo, color='r', linestyle='--')
        ax.axhline(yo, color='g', linestyle='--')
        ax.axhline(sz[1] - yo, color='g', linestyle='--')
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


def downsample_normalize(movie_path: Path, frame_rate: float,
                         bin_size: float, lower_quantile: float,
                         upper_quantile: float) -> np.ndarray:
    """reads in a movie (nframes x nrows x ncols), downsamples,
    creates an average projection, and normalizes according to
    quantiles in that projection.

    Parameters
    ----------
    movie_path: Path
        path to an h5 file, containing an (nframes x nrows x ncol) dataset
        named 'data'
    frame_rate: float
        frame rate of the movie specified by 'movie_path'
    bin_size: float
        desired duration in seconds of a downsampled bin, i.e. the reciprocal
        of the desired downsampled frame rate.
    lower_quantile: float
        arg supplied to `np.quantile()` to determine lower cutoff value from
        avg projection for normalization.
    upper_quantile: float
        arg supplied to `np.quantile()` to determine upper cutoff value from
        avg projection for normalization.

    Returns
    -------
    ds: np.ndarray
        a downsampled and normalized array

    Notes
    -----
    This strategy was satisfactory in the labeling app for maintaining
    consistent visibility.

    """
    ds = downsample_h5_video(
            movie_path,
            input_fps=frame_rate,
            output_fps=1.0 / bin_size)
    avg_projection = ds.mean(axis=0)
    lower_cutoff, upper_cutoff = np.quantile(
                avg_projection.flatten(), (lower_quantile, upper_quantile))
    ds = normalize_array(ds,
                         lower_cutoff=lower_cutoff,
                         upper_cutoff=upper_cutoff)
    return ds


class RegistrationQC(argschema.ArgSchemaParser):
    default_schema = RegistrationQCInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args['log_level'])

        # create and write the summary png
        motion_offset_df = pd.read_csv(self.args['motion_diagnostics_output'])
        png_out_path = make_png(Path(self.args['max_projection_output']),
                                Path(self.args['avg_projection_output']),
                                motion_offset_df,
                                Path(self.args['registration_summary_output']))
        self.logger.info(f"wrote {png_out_path}")

        # downsample and normalize the input movies
        ds_partial = partial(downsample_normalize,
                             frame_rate=self.args['movie_frame_rate_hz'],
                             bin_size=self.args['preview_frame_bin_seconds'],
                             lower_quantile=self.args['movie_lower_quantile'],
                             upper_quantile=self.args['movie_upper_quantile'])
        processed_vids = [ds_partial(i)
                          for i in [
                              Path(self.args['uncorrected_path']),
                              Path(self.args['motion_corrected_output'])]]
        self.logger.info("finished downsampling motion corrected "
                         "and non-motion corrected movies")

        # tile into 1 movie, raw on left, motion corrected on right
        tiled_vids = np.block(processed_vids)

        # make into a viewable artifact
        playback_fps = (self.args['preview_playback_factor'] /
                        self.args['preview_frame_bin_seconds'])
        encode_video(tiled_vids,
                     self.args['motion_correction_preview_output'],
                     playback_fps)
        self.logger.info("wrote "
                         f"{self.args['motion_correction_preview_output']}")

        return


if __name__ == "__main__":
    rqc = RegistrationQC()
    rqc.run()

import json
import os
import tempfile
from pathlib import Path
from typing import Tuple, List

import argschema
import h5py
import marshmallow
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage.filters import median_filter

from ophys_etl.schemas.fields import ExistingFile, ExistingH5File
from ophys_etl.qc.registration_qc import RegistrationQC
from ophys_etl.transforms.suite2p_wrapper import (Suite2PWrapper,
                                                  Suite2PWrapperSchema)
from suite2p.registration.rigid import shift_frame


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
                     "with a median filter of this duration [seconds]"))
    outlier_maxregshift = argschema.fields.Float(
        required=False,
        default=0.1,
        description=("units [fraction FOV dim]. After median-filter "
                     "detrending, outliers more than this value are "
                     "clipped to this value in x and y offset, independently."
                     "This is similar to Suite2P's internal maxregshift, but"
                     "allows for low-frequency drift."))

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


def projection_process(data: np.ndarray,
                       projection: str = "max") -> np.ndarray:
    """

    Parameters
    ----------
    data: np.ndarray
        nframes x nrows x ncols, uint16
    projection: str
        "max" or "avg"

    Returns
    -------
    proj: np.ndarray
        nrows x ncols, uint8

    """
    if projection == "max":
        proj = np.max(data, axis=0)
    elif projection == "avg":
        proj = np.mean(data, axis=0)
    else:
        raise ValueError("projection can be \"max\" or \"avg\" not "
                         f"{projection}")
    proj = np.uint8(proj * 255.0 / proj.max())
    return proj


def identify_and_clip_outliers(data: np.ndarray,
                               med_filter_size: int,
                               thresh: int) -> Tuple[np.ndarray, List]:
    """given data, identify the indices of outliers
    based on median filter detrending, and a threshold

    Parameters
    ----------
    data: np.ndarray
        1D array of samples
    med_filter_size: int
        the number of samples for 'size' in
        scipy.ndimage.filters.median_filter
    thresh: int
        multipled by the noise estimate to establish a threshold, above
        which, samples will be marked as outliers.

    Returns
    -------
    data: np.ndarry
        1D array of samples, clipped to threshold around detrended data
    indices: np.ndarray
        the indices where clipping took place

    """
    detrended = data - median_filter(data,
                                     med_filter_size,
                                     mode='nearest')
    higher = np.argwhere(detrended > thresh).flatten()
    lower = np.argwhere(detrended < -thresh).flatten()
    indices = list(set(higher).union(set(lower)))
    print(higher,lower, indices)
    if higher.size > 0:
        print(data[higher], detrended[higher])
        data[higher] = detrended[higher] + thresh
        print(data[higher], detrended[higher])
    if lower.size > 0:
        data[lower] = detrended[lower] - thresh
    return data, indices


class Suite2PRegistration(argschema.ArgSchemaParser):
    default_schema = Suite2PRegistrationInputSchema
    default_output_schema = Suite2PRegistrationOutputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args['log_level'])
        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA",
                                              "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")

        # register with Suite2P
        suite2p_args = self.args['suite2p_args']
        self.logger.info("attempting to motion correct "
                         f"{suite2p_args['h5py']}")
        register = Suite2PWrapper(input_data=suite2p_args, args=[])
        register.run()

        # why does this logger assume the Suite2PWrapper name? reset
        self.logger.name = type(self).__name__

        # get paths to Suite2P outputs
        with open(suite2p_args["output_json"], "r") as f:
            outj = json.load(f)
        tif_paths = [Path(i) for i in outj['output_files']["*.tif"]]
        ops_path = Path(outj['output_files']['ops.npy'][0])

        # Suite2P ops file contains at least the following keys:
        # ["Lx", "Ly", "nframes", "xrange", "yrange", "xoff", "yoff",
        #  "corrXY", "meanImg"]
        ops = np.load(ops_path, allow_pickle=True)

        # identify and clip offset outliers
        detrend_size = int(self.args['movie_frame_rate_hz'] *
                           self.args['outlier_detrend_window'])
        xlimit = int(ops.item()['Lx'] * self.args['outlier_maxregshift'])
        ylimit = int(ops.item()['Ly'] * self.args['outlier_maxregshift'])
        self.logger.info("checking whether to clip where median de-trended "
                         "offsets exceed (x,y) limits of "
                         f"({xlimit},{ylimit}) [pixels]")
        delta_x, x_clipped = identify_and_clip_outliers(
                np.array(ops.item()["xoff"]), detrend_size, xlimit)
        delta_y, y_clipped = identify_and_clip_outliers(
                np.array(ops.item()["yoff"]), detrend_size, ylimit)
        clipped_indices = list(set(x_clipped).union(set(y_clipped)))
        self.logger.info(f"{len(x_clipped)} frames clipped in x")
        self.logger.info(f"{len(y_clipped)} frames clipped in y")
        self.logger.info(f"{len(clipped_indices)} frames will be adjusted "
                         "for clipping")

        # accumulate data from Suite2P's tiffs
        data = []
        for fname in tif_paths:
            with tifffile.TiffFile(fname) as f:
                nframes = len(f.pages)
                for i, page in enumerate(f.pages):
                    arr = page.asarray()
                    if i == 0:
                        data.append(
                                np.zeros((nframes, *arr.shape), dtype='int16'))
                    data[-1][i] = arr
        data = np.concatenate(data, axis=0)
        data[data < 0] = 0
        data = np.uint16(data)

        # anywhere we've clipped the offset, translate the raw frame
        # with the new clipped value and subsitute into data
        with h5py.File(self.args['suite2p_args']['h5py'], "r") as f:
            for frame_index in clipped_indices:
                raw_frame = f['data'][frame_index]
                data[frame_index] = shift_frame(raw_frame,
                                                delta_y[frame_index],
                                                delta_x[frame_index])


        # write the hdf5
        with h5py.File(self.args['motion_corrected_output'], "w") as f:
            f.create_dataset("data", data=data, chunks=(1, *data.shape[1:]))
        self.logger.info("concatenated Suite2P tiff output to "
                         f"{self.args['motion_corrected_output']}")

        # make projections
        mx_proj = projection_process(data, projection="max")
        av_proj = projection_process(data, projection="avg")
        # TODO: normalize here, if desired
        # save projections
        for im, dst_path in zip(
                [mx_proj, av_proj],
                [self.args['max_projection_output'],
                    self.args['avg_projection_output']]):
            with Image.fromarray(im) as pilim:
                pilim.save(dst_path)
            self.logger.info(f"wrote {dst_path}")


        # Save motion offset data to a csv file
        # TODO: This *.csv file is being created to maintain compatability
        # with current ophys processing pipeline. In the future this output
        # should be removed and a better data storage format used.
        # 01/25/2021 - NJM
        x_is_clipped = np.zeros(delta_x.size).astype(bool)
        x_is_clipped[x_clipped] = True
        y_is_clipped = np.zeros(delta_y.size).astype(bool)
        y_is_clipped[y_clipped] = True
        motion_offset_df = pd.DataFrame({
            "framenumber": list(range(ops.item()["nframes"])),
            "x": delta_x,
            "y": delta_y,
            "x_clipped": x_is_clipped,
            "y_clipped": y_is_clipped,
            "correlation": ops.item()["corrXY"]
        })
        motion_offset_df.to_csv(
            path_or_buf=self.args['motion_diagnostics_output'],
            index=False)
        self.logger.info(
            f"Writing the LIMS expected 'OphysMotionXyOffsetData' "
            f"csv file to: {self.args['motion_diagnostics_output']}")
        if len(clipped_indices) != 0:
            self.logger.warn(
                    "some offsets have been clipped and the values "
                    "for 'correlation' in "
                    "{self.args['motion_diagnostics_output']} "
                    "where (x_clipped OR y_clipped) = True are not valid")

        qc_args = {k: self.args[k]
                   for k in ['movie_frame_rate_hz',
                             'max_projection_output',
                             'avg_projection_output',
                             'motion_diagnostics_output',
                             'motion_corrected_output',
                             'motion_correction_preview_output',
                             'registration_summary_output',
                             'log_level']}
        qc_args.update({
                'uncorrected_path': self.args['suite2p_args']['h5py']})
        rqc = RegistrationQC(input_data=qc_args, args=[])
        rqc.run()

        # Clean up temporary directories and/or files created during
        # Schema invocation
        if self.schema.tmpdir is not None:
            self.schema.tmpdir.cleanup()

        outj = {k: self.args[k]
                for k in ['motion_corrected_output',
                          'motion_diagnostics_output',
                          'max_projection_output',
                          'avg_projection_output',
                          'registration_summary_output',
                          'motion_correction_preview_output'
                          ]}
        self.output(outj, indent=2)


if __name__ == "__main__":  # pragma: nocover
    s2preg = Suite2PRegistration()
    s2preg.run()

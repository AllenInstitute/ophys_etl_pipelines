import logging
import os
import shutil

import h5py
import numpy as np
from argschema import ArgSchemaParser

import ophys_etl.modules.demix.demixer as demixer
from ophys_etl.modules.demix.schemas import (
    DemixJobOutputSchema,
    DemixJobSchema,
)


class DemixJob(ArgSchemaParser):

    default_schema = DemixJobSchema
    default_output_schema = DemixJobOutputSchema

    def __assert_exists(self, file_name):
        if not os.path.exists(file_name):
            raise IOError("file does not exist: %s" % file_name)

    def __get_path(self, obj, key, check_exists):
        try:
            path = obj[key]
        except KeyError:
            raise KeyError("required input field '%s' does not exist" % key)

        if check_exists:
            self.__assert_exists(path)

        return path

    def __parse_input(self, args, exclude_labels):
        movie_h5 = self.__get_path(args, "movie_h5", True)
        traces_h5 = self.__get_path(args, "traces_h5", True)
        output_h5 = self.__get_path(args, "output_file", False)

        with h5py.File(movie_h5, "r") as f:
            movie_shape = f["data"].shape[1:]

        with h5py.File(traces_h5, "r") as f:
            traces = f["data"][()]
            trace_ids = [int(rid) for rid in f["roi_names"][()]]

        rois = self.__get_path(args, "roi_masks", False)
        masks = np.zeros(
            (len(rois), movie_shape[0], movie_shape[1]), dtype=bool
        )
        valid = np.ones(len(rois), dtype=bool)

        for roi in rois:
            mask = np.zeros(movie_shape, dtype=bool)
            mask_matrix = np.array(roi["mask"], dtype=bool)
            mask[
                roi["y"] : roi["y"] + roi["height"],
                roi["x"] : roi["x"] + roi["width"],
            ] = mask_matrix

            rid = int(roi["id"])
            try:
                ridx = trace_ids.index(rid)
            except ValueError as e:
                raise ValueError(
                    "Could not find cell roi id %d in roi traces file" % rid, e
                )
            masks[ridx, :, :] = mask
            exclude_match = (
                len(set(exclude_labels) & set(roi.get("exclusion_labels", [])))
                > 0
            )
            if exclude_match:
                valid[ridx] = False

        return traces, masks, valid, np.array(trace_ids), movie_h5, output_h5

    def run(self):
        logging.debug("reading input")

        (
            traces,
            masks,
            valid,
            trace_ids,
            movie_h5,
            output_h5,
        ) = self.__parse_input(self.args, self.args["exclude_labels"])
        logging.debug(
            "excluded masks: %s",
            str(zip(np.where(~valid)[0], trace_ids[~valid])),
        )
        output_dir = os.path.dirname(output_h5)
        plot_dir = os.path.join(output_dir, "demix_plots")
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
        os.mkdir(plot_dir)

        logging.debug("reading movie")
        with h5py.File(movie_h5, "r") as f:
            movie = f["data"][()]

        # only demix non-union, non-duplicate ROIs
        demix_traces = traces[valid]
        demix_masks = masks[valid]

        logging.debug("demixing")
        demixed_traces, drop_frames = demixer.demix_time_dep_masks(
            demix_traces, movie, demix_masks
        )

        nt_inds = demixer.plot_negative_transients(
            demix_traces,
            demixed_traces,
            demix_masks,
            trace_ids[valid],
            plot_dir,
        )

        logging.debug(
            "rois with negative transients: %s",
            str(trace_ids[valid][nt_inds]),
        )

        nb_inds = demixer.plot_negative_baselines(
            demix_traces,
            demixed_traces,
            demix_masks,
            trace_ids[valid],
            plot_dir,
        )

        # negative baseline rois (and those that overlap with them) become nans
        logging.debug(
            "rois with negative baselines (or overlap with them): %s",
            str(trace_ids[valid][nb_inds]),
        )
        demixed_traces[nb_inds, :] = np.nan

        logging.info("Saving output")
        out_traces = np.zeros(traces.shape, dtype=demix_traces.dtype)
        out_traces[:] = np.nan
        out_traces[valid] = demixed_traces

        with h5py.File(output_h5, "w") as f:
            f.create_dataset("data", data=out_traces, compression="gzip")
            roi_names = np.array([str(rn) for rn in trace_ids]).astype(
                np.string_
            )
            f.create_dataset("roi_names", data=roi_names)

        self.output(
            dict(
                negative_transient_roi_ids=trace_ids[valid][nt_inds],
                negative_baseline_roi_ids=trace_ids[valid][nb_inds],
            )
        )


if __name__ == "__main__":
    demix_job = DemixJob()
    demix_job.run()

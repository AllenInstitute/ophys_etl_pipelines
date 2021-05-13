import argschema
import h5py
import os
import numpy as np

from ophys_etl.modules.event_detection.schemas import EventDetectionInputSchema
from ophys_etl.modules.event_detection import utils


class EventDetection(argschema.ArgSchemaParser):
    default_schema = EventDetectionInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))
        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA",
                                              "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")

        # read in the input file
        with h5py.File(self.args['ophysdfftracefile'], 'r') as f:
            roi_names = f['roi_names'][()].astype('int')
            valid_roi_indices = np.argwhere(
                    np.isin(
                        roi_names,
                        self.args['valid_roi_ids'])).flatten()
            dff = f['data'][valid_roi_indices]
        valid_roi_names = roi_names[valid_roi_indices]

        empty_warning = None
        if valid_roi_names.size == 0:
            events = np.empty_like(dff)
            lambdas = np.empty(0)
            noise_stds = np.empty(0)
            empty_warning = ("No valid ROIs in "
                             f"{self.args['ophysdfftracefile']}. "
                             "datasets in output file will be empty.")
            self.logger.warn(empty_warning)
        else:
            # run FastLZeroSpikeInference
            noise_filter_samples = int(self.args['noise_median_filter'] *
                                       self.args['movie_frame_rate_hz'])
            trace_filter_samples = int(self.args['trace_median_filter'] *
                                       self.args['movie_frame_rate_hz'])
            dff, noise_stds = utils.estimate_noise_detrend(
                    dff,
                    noise_filter_size=noise_filter_samples,
                    trace_filter_size=trace_filter_samples)
            gamma = utils.calculate_gamma(self.args['halflife'],
                                          self.args['movie_frame_rate_hz'])
            events, lambdas = utils.get_events(
                    traces=dff,
                    noise_estimates=noise_stds * self.args['noise_multiplier'],
                    gamma=gamma,
                    ncpu=self.args['n_parallel_workers'])

        with h5py.File(self.args['output_event_file'], "w") as f:
            f.create_dataset("events", data=events)
            f.create_dataset("roi_names", data=valid_roi_names)
            f.create_dataset("noise_stds", data=noise_stds)
            f.create_dataset("lambdas", data=lambdas)
            if empty_warning:
                f.create_dataset("warning", data=empty_warning)
        self.logger.info(f"wrote {self.args['output_event_file']}")


if __name__ == "__main__":  # pragma: no cover
    ed = EventDetection()
    ed.run()

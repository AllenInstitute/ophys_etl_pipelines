import argschema
import warnings
import json
import multiprocessing
import h5py
import numpy as np
import marshmallow as mm

from ophys_etl.modules.event_detection.resources.event_decay_time_lookup \
        import event_decay_lookup_dict as decay_lookup
from ophys_etl.schemas.fields import H5InputFile
from ophys_etl.modules.event_detection.utils import (
        calculate_halflife, EventDetectionException)


class EventDetectionInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(required=False, default="INFO")
    movie_frame_rate_hz = argschema.fields.Float(
        required=True,
        description=("frame rate of the ophys video / trace. "
                     "used to upsample slower rates to 31Hz."))
    full_genotype = argschema.fields.Str(
        required=False,
        description=("full genotype of the specimen. Used to look "
                     "up characteristic decay time from "
                     "ophys_etl.resources.event_decay_time_lookup"))
    decay_time = argschema.fields.Float(
        required=False,
        description="characteristic decay time [seconds]")
    ophysdfftracefile = H5InputFile(
        required=True,
        description=("h5 file containing keys `data` (nROI x nframe) and "
                     "`roi_names` (length = nROI) output by the df/F "
                     "computation."))
    valid_roi_ids = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        cli_as_single_argument=True,
        description=("list of ROI ids that are valid, for which "
                     "event detection will be performed."))
    output_event_file = argschema.fields.OutputFile(
        required=True,
        description="location for output of event detection.")
    n_parallel_workers = argschema.fields.Int(
        required=False,
        default=1,
        description=("number of parallel workers. If set to -1 "
                     "is set to multiprocessing.cpu_count()."))
    noise_median_filter = argschema.fields.Float(
        required=False,
        default=1.0,
        description=("median filter length used to detrend data "
                     "during noise estimation [seconds]. Typically "
                     "shorter than 'trace_median_filter'."))
    trace_median_filter = argschema.fields.Float(
        required=False,
        default=3.2,
        description=("median filter length used to detrend data "
                     "before passing to FastLZero. [seconds]. Typically "
                     "longer than 'noise_median_filter'."))
    noise_multiplier = argschema.fields.Float(
        required=False,
        description=("manual specification of noise multiplier. If not "
                     "provided, will be defaulted by `get_noise_multiplier` "
                     "post_load below."))

    @mm.post_load
    def check_dff_h5_keys(self, data, **kwargs):
        with h5py.File(data['ophysdfftracefile'], 'r') as f:
            if 'roi_names' not in list(f.keys()):
                raise EventDetectionException(
                        f"DFF trace file {data['ophysdfftracefile']} "
                        "does not have the key 'roi_names', which indicates "
                        "it has come from an old version of creating "
                        "DFF traces < April, 2019. Consider recreating the "
                        "DFF file with a current version of that module.")
        return data

    @mm.post_load
    def get_noise_multiplier(self, data, **kwargs):
        if 'noise_multiplier' in data:
            return data
        # NOTE 11Hz value found empirically to behave well compared
        # to 31Hz data sub-sampled to 11Hz.
        val_11hz = 2.6
        val_31hz = 2.0
        if np.round(data['movie_frame_rate_hz'] / 11.0) == 1:
            data['noise_multiplier'] = val_11hz
        elif np.round(data['movie_frame_rate_hz'] / 31.0) == 1:
            data['noise_multiplier'] = val_31hz
        else:
            raise EventDetectionException(
                "You did not specify 'noise_multiplier'. In that case, the "
                "multiplier is selected by 'movie_frame_rate_hz' but only "
                "freqeuncies around 11Hz and 31Hz have specified values: "
                f"({val_11hz}, {val_31hz}) respectively. You specified a "
                f"'movie_frame_rate' of {data['movie_frame_rate_hz']}")
        return data

    @mm.post_load
    def cpu_to_the_max(self, data, **kwargs):
        if data['n_parallel_workers'] == -1:
            data['n_parallel_workers'] = multiprocessing.cpu_count()
        return data

    @mm.post_load
    def get_decay_time(self, data, **kwargs):
        if ('full_genotype' not in data) & ('decay_time' not in data):
            raise EventDetectionException(
                    "Must provide either `decay_time` or `full_genotype` "
                    "for decay time lookup. "
                    "Available lookup values are "
                    f"\n{json.dumps(decay_lookup, indent=2)}")
        if 'full_genotype' in data:
            if data['full_genotype'] not in decay_lookup:
                raise EventDetectionException(
                        f"{data['full_genotype']} not available. "
                        "Available lookup values are "
                        f"\n{json.dumps(decay_lookup, indent=2)}")
            lookup = decay_lookup[data['full_genotype']]
            if 'decay_time' in data:
                warnings.warn("You specified a decay_time of "
                              f"{data['decay_time']} but that is being "
                              "overridden by a lookup by genotype to give "
                              f"a decay time of {lookup}.")
            data['decay_time'] = lookup
        data['halflife'] = calculate_halflife(data['decay_time'])
        return data

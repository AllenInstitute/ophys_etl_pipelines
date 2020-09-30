import h5py
import argschema
import marshmallow as mm

from ophys_etl.schemas.fields import H5InputFile


class ClassifierSchema(argschema.ArgSchema):
    neuropil_traces_path = H5InputFile(
        required=True,
        description=("Path to neuropil traces from an experiment."))
    neuropil_traces_data_key = argschema.fields.Str(
        required=False,
        missing="data",
        description=("Key for trace data in `neuropil_traces_path`"))
    neuropil_trace_names_key = argschema.fields.Str(
        required=False,
        missing="roi_names",
        description=("Key for roi names/ids in `neuropil_traces_path`"))
    traces_path = H5InputFile(
        required=True,
        description=("Path to ROI traces from an experiment."))
    traces_data_key = argschema.fields.Str(
        required=False,
        missing="data",
        description=("Key for roi names/ids in `traces_path`"))
    trace_names_key = argschema.fields.Str(
        required=False,
        missing="roi_names",
        description=("Key for roi names/ids in `traces_path`"))
    roi_masks_path = argschema.fields.InputFile(
        required=True,
        description=("Path to json file of segmented ROI masks. Records "
                     "are expected to conform to `DenseROISchema`"))
    rig = argschema.fields.Str(
        required=True,
        description=("Name of the ophys rig used for the experiment."))
    depth = argschema.fields.Int(
        required=True,
        description=("Imaging depth for the experiment."))
    full_genotype = argschema.fields.Str(
        required=True,
        description=("Genotype of the experimental subject."))
    targeted_structure = argschema.fields.Str(
        required=True,
        description=("Name of the brain structure targeted by imaging."))
    trace_sampling_rate = argschema.fields.Int(
        required=False,
        missing=31,
        description=("Sampling rate of trace (frames per second)"))
    desired_trace_sampling_rate = argschema.fields.Int(
        required=False,
        missing=4,
        validate=lambda x: x > 0,
        description=("Target rate for trace downsampling (frames per second) "
                     "Will use average bin values for downsampling."))

    @mm.post_load
    def check_keys_exist(self, data: dict, **kwargs) -> dict:
        """ For h5 files, check that the passed key exists in the data. """
        pairs = [("neuropil_traces_path", "neuropil_traces_data_key"),
                 ("traces_path", "traces_data_key")]
        for h5file, key in pairs:
            with h5py.File(data[h5file], "r") as f:
                if not data[key] in f.keys():
                    raise mm.ValidationError(
                        f"Key '{data[key]}' ({key}) was missing in h5 file "
                        f"{data[h5file]} ({h5file}.")
        return data

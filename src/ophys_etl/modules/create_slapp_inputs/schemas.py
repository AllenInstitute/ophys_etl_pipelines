import argschema
import marshmallow as mm
from marshmallow import ValidationError

from ophys_etl.schemas.fields import H5InputFile


class ExperimentSchema(argschema.schemas.DefaultSchema):
    experiment_id = argschema.fields.Int(
        required=True,
        description="experiment id")
    local_ids = argschema.fields.List(
        argschema.fields.Int,
        required=False,
        cli_as_single_argument=True,
        description=("which ROIs to select from this experiment. The "
                     "'local_ids' refer to the unique ids assigned to ROIs "
                     "by ophys_etl.convert_rois and are the 'id' field for "
                     "every ROI listed in binarized_rois_path"))
    binarized_rois_path = argschema.fields.InputFile(
        required=False,
        description="path to output from ophys_etl.convert_rois")
    traces_h5_path = H5InputFile(
        required=False,
        description="path to traces file output from extract_traces")
    movie_path = H5InputFile(
        required=True,
        description="path to mtion corrected movie")


class SlappTransformInputSchema(argschema.ArgSchema):
    """This module transforms per ophys experiment paths from typical outputs
    from ophys_etl.pipelines.segment_and_binarize (binarize_output.json)
    and the AllenSDK extract_traces (roi_traces.h5) into a format prepared for
    the segmentation-labeling-app (slapp) transform pipeline when using the
    `xform_from_prod_manifest` method and the
    `ProdSegmentationRunManifestSchema`. It also allows for sub-selecting a
    number of ROIs either through direct specification per-experiment, or a
    random sub-selection across experiments.
    """
    global_id_offset = argschema.fields.Int(
        required=True,
        description=("global ROI ids will start at this value. The "
                     "segmentation-labeling-app (slapp) needs a unique "
                     "id for every ROI. The ROIs between experiments "
                     "will have id collisions without a global re-labeling."))
    experiments = argschema.fields.Nested(ExperimentSchema, many=True)
    random_seed = argschema.fields.Int(
        required=True,
        default=42,
        description=("seed for random number generator to select ROIs. "
                     "ignored if n_roi_total is not set."))
    n_roi_total = argschema.fields.Int(
        required=False,
        description=("if provided, experiments.local_ids will be ignored "
                     "and n_roi_total will be randomly sampled from all "
                     "valid rois in the experiments. A use-case is that "
                     "one has specified a number of experiments and wants "
                     "to randomly select some number of ROIs from those "
                     "experiments to present to annotators. I.e. here are "
                     "my special 100 experiments. I want annotations on 300 "
                     "randomly-selected ROIs from these experiments."))
    output_dir = argschema.fields.OutputDir(
        required=True,
        description=("one file per experiment "
                     "{output_dir}/{experiment_id}_slapp_tform_input.json "
                     "will be created"))
    input_rootdir = argschema.fields.InputDir(
        required=False,
        description=("if given, binarized rois and traces will be sought "
                     "as <input_rootdir>/<experiment_id>/<filename> ."))
    binarized_filename = argschema.fields.Str(
        required=True,
        default="binarize_output.json",
        description="filename of convert_rois output")
    trace_filename = argschema.fields.Str(
        required=True,
        default="roi_traces.h5",
        description="filename of convert_rois output")

    @mm.post_load
    def check_experiment_paths(self, data, **kwargs):
        missing = False
        for experiment in data['experiments']:
            if (('binarized_rois_path' not in experiment) |
               ('traces_h5_path' not in experiment)):
                missing = True
        if missing & ('input_rootdir' not in data):
            raise ValidationError("either specify 'binarized_rois_path' and "
                                  "'traces_h5_path' for all experiments or "
                                  "specify 'input_rootdir'")
        return data

    @mm.post_load
    def check_local_ids(self, data, **kwargs):
        missing = False
        for experiment in data['experiments']:
            if ('local_ids' not in experiment):
                missing = True
        if missing & ('n_roi_total' not in data):
            raise ValidationError("either specify 'local_ids' for all "
                                  "experiments or specify 'n_roi_total' "
                                  "to trigger random selection")
        return data


class SlappTransformOutputSchema(argschema.schemas.DefaultSchema):
    outputs = argschema.fields.List(argschema.fields.InputFile)


class ExperimentOutputSchema(argschema.schemas.DefaultSchema):
    experiment_id = argschema.fields.Int(
        required=True,
        description="experiment id")
    local_to_global_roi_id_map = argschema.fields.Dict(
        keys=argschema.fields.Int(),
        values=argschema.fields.Int(),
        required=True,
        description="{local: global} map")
    binarized_rois_path = argschema.fields.InputFile(
        required=True,
        description="path to output from ophys_etl.convert_rois")
    traces_h5_path = H5InputFile(
        required=True,
        description="path to traces file output from extract_traces")
    movie_path = H5InputFile(
        required=True,
        description="path to mtion corrected movie")

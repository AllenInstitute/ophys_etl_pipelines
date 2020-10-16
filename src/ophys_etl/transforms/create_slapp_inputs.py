import argschema
import json
import numpy as np
from typing import List, Any
from pathlib import Path
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


def select_rois(experiments: List[Any], n_roi_total: int,
                random_seed: int) -> List[Any]:
    """given a list of experiments, subselect a number of non-excluded ROIs
    and return experiments with selected ROIs with those ids (local) listed in
    the 'local_ids' field.

    Parameters
    ----------
    experiments: list
        each element satsifies ExperimentSchema
    n_roi_total: int
        total number of ROIs to select
    random_seed: int
        seed for random number generator

    Returns
    -------
    selected_experiments: list
        each element satsifies ExperimentSchema. 'local_ids' field is populated

    Raises
    ------
    ValueError if `n_roi_total` exceeds the total number of ROIs in the
    `experiments` data

    """
    # list all non-excluded ROIs
    roi_list = []
    for experiment in experiments:
        with open(experiment['binarized_rois_path'], "r") as f:
            j = json.load(f)
        for roi in j:
            if not roi['exclusion_labels']:
                roi_list.append((experiment['experiment_id'], roi['id']))

    # randomly select some of them
    # compatible with numpy 1.16, forced by Suite2P dependencies
    rng = np.random.RandomState(seed=random_seed)
    indices = np.arange(len(roi_list))
    try:
        # in numpy 1.16, choice arg must be 1D
        subinds = rng.choice(indices, n_roi_total, replace=False)
        subset = [roi_list[i] for i in indices if i in subinds]
    except ValueError as ve:
        ve.args += ("perhaps you requested more ROIs than are available. "
                    f"{len(roi_list)} available {n_roi_total} requested", )
        raise

    # group subset by experiment
    groups = {}
    for eid, roi_id in subset:
        if eid not in groups:
            groups[eid] = []
        groups[eid].append(roi_id)

    # add local id lists back in to experiments
    selected_experiments = []
    for experiment in experiments:
        if experiment['experiment_id'] in groups:
            selected_experiments.append(experiment)
            selected_experiments[-1]['local_ids'] = \
                groups[experiment['experiment_id']]

    return selected_experiments


def populate_experiments_rglob(experiments: List[Any], rootdir: Path,
                               binarized_filename: str, trace_filename: str):
    """seeks filename matches and populates experiment dictionary with
    found paths

    Parameters
    ----------
    experiments: list
        each element satsifies ExperimentSchema
    rootdir: Path
        rootdir for search. File names for each experiment will be searched
        in <rootdir>/<experiment_id>
    binarized_filename: str
        filename of the convert_rois output
    trace_filename: str
        filename of the roi traces h5

    Returns
    -------
    experiments: list
        each element satsifies ExperimentSchema, with paths populated by
        this search

    """

    for experiment in experiments:
        edir = rootdir / f"{experiment['experiment_id']}"
        fnames = []
        for filename in [binarized_filename, trace_filename]:
            try:
                fname = next(edir.rglob(filename))
            except StopIteration as si:
                si.args += (f"could not find {filename} in {edir}", )
                raise
            fnames.append(str(fname))
        experiment['binarized_rois_path'] = fnames[0]
        experiment['traces_h5_path'] = fnames[1]

    return experiments


class SlappTransformInput(argschema.ArgSchemaParser):
    default_schema = SlappTransformInputSchema
    default_output_schema = SlappTransformOutputSchema

    def run(self):
        experiments = self.args['experiments']

        # this arg triggers a search for 2 input files
        if 'input_rootdir' in self.args:
            self.logger.info("finding paths for ROIs and traces in "
                             f"{self.args['input_rootdir']}")
            experiments = populate_experiments_rglob(
                    experiments,
                    rootdir=Path(self.args['input_rootdir']),
                    binarized_filename=self.args['binarized_filename'],
                    trace_filename=self.args['trace_filename'])

        # this arg triggers a listing of all included ROIs
        # and a random subselection of those
        if 'n_roi_total' in self.args:
            self.logger.info(f"randomly selecting {self.args['n_roi_total']} "
                             "ROIs from non-excluded ROIs")
            experiments = select_rois(experiments, self.args['n_roi_total'],
                                      self.args['random_seed'])

        # apply global label to each ROI and output
        global_counter = self.args['global_id_offset']
        outdir = Path(self.args['output_dir'])
        outjpaths = []
        for experiment in experiments:
            if 'local_ids' in experiment:
                local_to_global = {}
                for local in experiment['local_ids']:
                    local_to_global[local] = global_counter
                    local_to_global += 1
                experiment['local_to_global_roi_id_map'] = local_to_global
                experiment.pop('local_ids')

                # validate
                ExperimentOutputSchema().load(experiment)

                # output data
                outjpaths.append(
                        str(outdir / f"{experiment['experiment_id']}.json"))
                with open(outjpaths[-1], "w") as f:
                    json.dump(experiment, f, indent=2)
                self.logger.info(f"wrote {outjpaths[-1]}")

        self.output({"outputs": outjpaths})
        self.logger.info(f"wrote {self.args['output_json']}")


if __name__ == "__main__":  # pragma: nocover
    sti = SlappTransformInput()
    sti.run()

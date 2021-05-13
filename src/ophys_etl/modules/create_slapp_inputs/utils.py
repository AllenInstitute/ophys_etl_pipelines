import json
import numpy as np
from typing import List, Any
from pathlib import Path


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

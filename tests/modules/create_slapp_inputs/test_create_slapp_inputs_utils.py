import pytest
import json
from pathlib import Path

from ophys_etl.modules.create_slapp_inputs import utils
from ophys_etl.modules.create_slapp_inputs import schemas


@pytest.mark.parametrize("n_roi_total", [7, 8])
@pytest.mark.parametrize(
        "experiments_fixture",
        [
            {'experiments': [
                {
                    'experiment_id': 1001,
                    'rois': [
                        {'id': 0, 'exclusion_labels': []},
                        {'id': 1, 'exclusion_labels': []},
                        {'id': 2, 'exclusion_labels': ['not empty']},
                        {'id': 3, 'exclusion_labels': []},
                        {'id': 4, 'exclusion_labels': []}]},
                {
                    'experiment_id': 1002,
                    'rois': [
                        {'id': 0, 'exclusion_labels': ['not empty']},
                        {'id': 1, 'exclusion_labels': []},
                        {'id': 2, 'exclusion_labels': []},
                        {'id': 3, 'exclusion_labels': []},
                        {'id': 4, 'exclusion_labels': []}]},
            ]},
            ], indirect=['experiments_fixture'])
def test_select_rois(experiments_fixture, n_roi_total):
    print(json.dumps(experiments_fixture, indent=2))
    selected = utils.select_rois(experiments_fixture, n_roi_total, 42)
    n = len([i for j in selected for i in j['local_ids']])
    assert n == n_roi_total


@pytest.mark.parametrize(
        "experiments_fixture",
        [
            {
                'experiments': [
                    {
                        'experiment_id': 1001,
                        'rois': [{'id': 0, 'exclusion_labels': []}],
                        },
                    {
                        'experiment_id': 1002,
                        'rois': [{'id': 0, 'exclusion_labels': []}],
                        },
                ],
                }], indirect=['experiments_fixture'])
def test_populate_experiments_rglob(experiments_fixture):
    # get the rootdir from the fixture
    exp = experiments_fixture[0]
    rootdir = Path(exp['binarized_rois_path']).parent.parent

    populated = utils.populate_experiments_rglob(
            experiments_fixture,
            rootdir=rootdir,
            binarized_filename="binarize_output.json",
            trace_filename="roi_traces.h5")

    # validate
    eschema = schemas.ExperimentSchema()
    for p in populated:
        eschema.load(p)

    # raises exception when file not found
    with pytest.raises(StopIteration,
                       match=".*could not find misspelled.json.*"):
        utils.populate_experiments_rglob(
                experiments_fixture,
                rootdir=rootdir,
                binarized_filename="misspelled.json",
                trace_filename="roi_traces.h5")

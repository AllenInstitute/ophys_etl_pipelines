import json
import h5py
import pytest
from pathlib import Path

import ophys_etl.transforms.create_slapp_inputs as csli


def experiment(args, tmp_path):
    eid = args.get("experiment_id")
    edir = tmp_path / f"{eid}"
    edir.mkdir()
    bpath = edir / "binarize_output.json"
    with open(bpath, "w") as f:
        json.dump(args.get("rois"), f)
    tpath = edir / "roi_traces.h5"
    with h5py.File(tpath, "w") as f:
        f.create_dataset("data", data=[])
    mpath = edir / "movie.h5"
    with h5py.File(mpath, "w") as f:
        f.create_dataset("data", data=[])

    experiment_dict = {
            'experiment_id': eid,
            'binarized_rois_path': str(bpath),
            'traces_h5_path': str(tpath),
            'movie_path': str(mpath)}
    if 'local_ids' in args:
        experiment_dict['local_ids'] = args.get('local_ids')

    return experiment_dict


@pytest.fixture
def experiments_fixture(request, tmp_path):
    experiments = list([experiment(i, tmp_path)
                        for i in request.param.get('experiments')])
    yield experiments


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
    selected = csli.select_rois(experiments_fixture, n_roi_total, 42)
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

    populated = csli.populate_experiments_rglob(
            experiments_fixture,
            rootdir=rootdir,
            binarized_filename="binarize_output.json",
            trace_filename="roi_traces.h5")

    # validate
    eschema = csli.ExperimentSchema()
    for p in populated:
        eschema.load(p)

    # raises exception when file not found
    with pytest.raises(StopIteration,
                       match=".*could not find misspelled.json.*"):
        csli.populate_experiments_rglob(
                experiments_fixture,
                rootdir=rootdir,
                binarized_filename="misspelled.json",
                trace_filename="roi_traces.h5")


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
                ]
                }], indirect=['experiments_fixture'])
def test_SlappTransformInput(experiments_fixture, tmp_path):
    exp = experiments_fixture[0]
    input_rootdir = Path(exp['binarized_rois_path']).parent.parent
    outdir = str(tmp_path)
    output_json_path = tmp_path / "output.json"
    args = {
            'global_id_offset': 20000,
            'experiments': experiments_fixture,
            'random_seed': 42,
            'n_roi_total': 2,
            'output_dir': outdir,
            'input_rootdir': str(input_rootdir),
            'output_json': str(output_json_path)
            }

    sti = csli.SlappTransformInput(input_data=args, args=[])
    sti.run()

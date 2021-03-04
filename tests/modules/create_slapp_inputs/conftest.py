import pytest
import h5py
import json


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

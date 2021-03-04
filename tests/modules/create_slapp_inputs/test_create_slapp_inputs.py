import pytest
from pathlib import Path

import ophys_etl.modules.create_slapp_inputs.__main__ as csli


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

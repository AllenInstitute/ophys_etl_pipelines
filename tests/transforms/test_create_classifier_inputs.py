import json
import os

import h5py
import pytest
from pathlib import Path

import ophys_etl.transforms.create_classification_inputs as cci


@pytest.fixture
def manifest_fixture(request, tmp_path):
    resources_path = Path(__file__).parent / 'resources'
    manifest = {
        'experiment_id': request.param.get('experiment')['experiment_id'],
        'movie_path': str(resources_path / 'motion_corrected_video_short.h5'),
        'binarized_rois_path': str(resources_path / 'binarize_output.json')
    }

    with open(Path(tmp_path) / 'manifest.json', 'w') as f:
        f.write(json.dumps(manifest))


@pytest.mark.parametrize(
    "manifest_fixture",
    [
        {
            'experiment':
                {
                    'experiment_id': 0
                }
        }
    ], indirect=['manifest_fixture']
)
def test_transform_pipeline(manifest_fixture, tmp_path):
    args = {
        "prod_segmentation_run_manifest": str(Path(tmp_path) / 'manifest.json'),
        "input_fps": 4.0,
        "output_fps": 2.0,
        "artifact_basedir": str(Path(tmp_path) / 'artifacts'),
        "output_manifest": str(Path(tmp_path) / 'manifest.jsonl'),
        "cropped_shape": [32, 32]
    }

    tp = cci.TransformPipeline(input_data=args, args=[])
    tp.run()

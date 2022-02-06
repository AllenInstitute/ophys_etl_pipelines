import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import sagemaker

from ophys_etl.modules.denoising.cloud.cli.__main__ import \
    CloudDenoisingTrainerModule
from ophys_etl.modules.denoising.cloud.ecr import ECRUploader


@patch("boto3.session")
@patch('docker.APIClient')
@patch.object(sagemaker.estimator.Estimator, 'fit')
@patch.object(ECRUploader, '_docker_login', return_value=('', ''))
@pytest.mark.parametrize('local_mode', [True, False])
def test_cli(_, __, ___, ____, local_mode):
    """Smoke tests the CLI"""
    sagemaker_execution_role_name = 'mock_sm_execution_role_name'

    if local_mode:
        instance_type = None
    else:
        instance_type = 'mock_instance_type'

    # Need to update the paths in input json to absolute paths
    with open((Path(__file__).parent / 'test_data' / 'input.json')) as f:
        input_json = json.load(f)

    input_json['finetuning_params']['model_source']['local_path'] = \
        str(Path(__file__).parent / 'test_data' /
            Path(input_json['finetuning_params']['model_source']
                 ['local_path']).name)

    for p in ('generator_params', 'test_generator_params'):
        cur_path = input_json[p]['train_path']
        new_path = str(Path(__file__).parent / 'test_data' /
                       Path(cur_path).name)
        input_json[p]['train_path'] = new_path

    with tempfile.NamedTemporaryFile() as input_json_fd:
        # Write out the input json with updated paths
        with open(input_json_fd.name, 'w') as f:
            f.write(json.dumps(input_json))

        with tempfile.NamedTemporaryFile() as pt_model_fd:
            input_data = {
                'local_mode': local_mode,
                'input_json_path': input_json_fd.name,
                'pretrained_model_path': pt_model_fd.name,
                'instance_type': instance_type,
                'sagemaker_execution_role': sagemaker_execution_role_name
            }
            train_mod = CloudDenoisingTrainerModule(
                input_data=input_data, args=[])
            train_mod.run()

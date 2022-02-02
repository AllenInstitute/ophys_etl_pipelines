import tempfile
from unittest.mock import patch

import pytest
import sagemaker

from ophys_etl.modules.denoising.cloud.cli.__main__ import \
    CloudDenoisingTrainerModule
from ophys_etl.modules.denoising.cloud.ecr import ECRUploader


@patch("boto3.session")
@patch('docker.APIClient')
@patch.object(sagemaker.estimator.Estimator, 'fit')
@patch.object(ECRUploader, '_get_ecr_credentials', return_value=('', ''))
@pytest.mark.parametrize('local_mode', [True, False])
def test_cli(_, __, ___, ____, local_mode):
    """Smoke tests the CLI"""
    if local_mode:
        instance_type = None
        sagemaker_execution_role_name = None
    else:
        instance_type = 'mock_instance_type'
        sagemaker_execution_role_name = 'mock_sm_execution_role_name'
    with tempfile.TemporaryDirectory() as d:
        input_data = {
            'local_mode': local_mode,
            'local_data_dir': d,
            'instance_type': instance_type,
            'sagemaker_execution_role': sagemaker_execution_role_name
        }
        train_mod = CloudDenoisingTrainerModule(input_data=input_data, args=[])
        train_mod.run()

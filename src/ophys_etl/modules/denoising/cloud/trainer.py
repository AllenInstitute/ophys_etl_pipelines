import json
import logging
from pathlib import Path
from typing import Optional

import boto3.session
import sagemaker
from sagemaker.estimator import Estimator


class Trainer:
    """
    A wrapper on sagemaker Estimator. Starts a training job using the docker
    image given by image_uri
    """
    def __init__(self,
                 input_json_path: Path,
                 image_uri: str,
                 bucket_name: str,
                 profile_name='default',
                 region_name='us-west-2',
                 local_mode=False,
                 instance_type: Optional[str] = None,
                 instance_count=1):
        """
        Parameters
        ----------
        input_json_path
            Path to input json. Must contain paths to training and validation
            data metadata
        image_uri
            The container image to run
        bucket_name
            The bucket to upload data to
        profile_name
            AWS profile name to use
        region_name
            AWS region to use
        local_mode
            Whether running locally.
        instance_type
            Instance type to use
        instance_count
            Instance count to use
        """
        if not local_mode:
            if instance_type is None:
                raise ValueError('Must provide instance type if not using '
                                 'local mode')

        self._input_json_path = input_json_path
        self._image_uri = image_uri
        self._local_mode = local_mode
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._profile_name = profile_name
        self._bucket_name = bucket_name
        self._logger = logging.getLogger(__name__)

        boto_session = boto3.session.Session(profile_name=profile_name,
                                             region_name=region_name)
        self._sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, default_bucket=bucket_name)

    def run(self):
        instance_type = 'local' if self._local_mode else self._instance_type
        sagemaker_session = None if self._local_mode else \
            self._sagemaker_session
        sagemaker_role_arn = self._get_sagemaker_execution_role_arn()
        output_path = self._get_output_path()

        estimator = Estimator(
            sagemaker_session=sagemaker_session,
            role=sagemaker_role_arn,
            instance_count=self._instance_count,
            instance_type=instance_type,
            image_uri=self._image_uri,
            hyperparameters={},
            output_path=output_path
        )

        local_input_data_dir = self._get_data_directory()

        if self._local_mode:
            data_path = f'file://{local_input_data_dir}'
        else:
            self._logger.info('Uploading input data to S3')
            data_path = self._sagemaker_session.upload_data(
                path=str(local_input_data_dir),
                key_prefix='input_data',
                bucket=self._bucket_name)
        estimator.fit(data_path)

    def _get_sagemaker_execution_role_arn(self) -> str:
        """
        Gets the sagemaker execution role arn

        Returns
        -------
        The sagemaker execution role arn

        Raises
        -------
        RuntimeError if the role cannot be found
        """
        iam = self._sagemaker_session.boto_session.client('iam')
        roles = iam.list_roles(PathPrefix='/service-role/')
        sm_roles = [x for x in roles['Roles'] if
                    x['RoleName'].startswith('AmazonSageMaker-ExecutionRole')]
        if sm_roles:
            sm_role = sm_roles[0]
        else:
            raise RuntimeError('Could not find the sagemaker execution role. '
                               'It should have already been created in AWS')
        return sm_role['Arn']

    def _get_data_directory(self) -> Path:
        """Validates that all data is in the same directory.
        If successful, returns the directory in which data is stored"""
        dirs = set()
        with open(self._input_json_path) as f:
            input_json = json.load(f)
        for p in ('generator_params', 'test_generator_params'):
            with open(input_json[p]['train_path']) as f:
                data_metadata = json.load(f)
            for exp_id in data_metadata:
                dirs.add(Path(data_metadata[exp_id]['path']).parent)
        if len(dirs) != 1:
            raise RuntimeError('Expected all training data to be in the same '
                               'directory')
        return list(dirs)[0]

    def _get_output_path(self) -> Optional[str]:
        """
        Directory to output the tarball of output. Only used in local mode.
        In non-local mode, this tarball is saved to the default location
        in s3
        Returns
        -------
        Output dir if local mode, else None
        """
        if self._local_mode:
            with open(self._input_json_path) as f:
                input_json = json.load(f)
            path = None
            for v in input_json.values():
                if 'output_dir' in v:
                    path = v['output_dir']
                    break
            if path is None:
                raise RuntimeError('Could not get output_dir from input_json')
            path = f'file://{path}'
        else:
            path = None
        return path

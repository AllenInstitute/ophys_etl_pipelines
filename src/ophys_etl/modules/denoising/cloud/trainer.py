from pathlib import Path
from typing import Optional

import boto3.session
import sagemaker
from sagemaker.estimator import Estimator

from ophys_etl.modules.denoising.cloud.aws_utils import \
    get_account_id


class Trainer:
    def __init__(self,
                 image_uri: str,
                 local_input_data_dir: Path,
                 bucket_name: str,
                 profile_name='default',
                 region_name='us-west-2',
                 local_mode=False,
                 instance_type: Optional[str] = None,
                 instance_count=1,
                 sagemaker_execution_role: Optional[str] = None):
        if not local_mode:
            if instance_type is None:
                raise ValueError('Must provide instance type if not using '
                                 'local mode')
            if sagemaker_execution_role is None:
                raise ValueError('Must provide sagemaker execution role name')

        self._sagemaker_execution_role = sagemaker_execution_role
        self._image_uri = image_uri
        self._local_mode = local_mode
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._profile_name = profile_name
        self._local_input_data_dir = local_input_data_dir
        self._bucket_name = bucket_name

        boto_session = boto3.session.Session(profile_name=profile_name,
                                             region_name=region_name)
        self._sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, default_bucket=bucket_name)

    def run(self):
        instance_type = 'local' if self._local_mode else self._instance_type
        sagemaker_session = None if self._local_mode else \
            self._sagemaker_session
        sagemaker_role_arn = '' if self._local_mode else \
            self._get_sagemaker_role_arn()

        estimator = Estimator(
            sagemaker_session=sagemaker_session,
            role=sagemaker_role_arn,
            instance_count=self._instance_count,
            instance_type=instance_type,
            image_uri=self._image_uri,
            hyperparameters={}
        )

        if self._local_mode:
            data_path = f'file://{self._local_input_data_dir}'
        else:
            data_path = self._sagemaker_session.upload_data(
                path=str(self._local_input_data_dir),
                key_prefix='input_data',
                bucket=self._bucket_name)
        estimator.fit(data_path)

    def _get_sagemaker_role_arn(self):
        account_id = get_account_id(profile_name=self._profile_name)
        role_id = self._sagemaker_execution_role
        return f'arn:aws:iam::{account_id}:role/service-role/{role_id}'

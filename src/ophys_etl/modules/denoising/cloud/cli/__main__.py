import logging
import os
from pathlib import Path

import argschema

from ophys_etl.modules.denoising.cloud.ecr import ECRUploader
from ophys_etl.modules.denoising.cloud.trainer import Trainer
from ophys_etl.modules.denoising.cloud.cli.schemas import \
    CloudDenoisingTrainerSchema


class CloudDenoisingTrainerModule(argschema.ArgSchemaParser):
    default_schema = CloudDenoisingTrainerSchema
    _logger = logging.getLogger(__name__)
    _container_path = Path(__file__).parent.parent / 'container'

    def run(self):
        print(os.listdir(self._container_path.parent))
        repository_name = self.args['docker_params']['repository_name']
        image_tag = self.args['docker_params']['image_tag']

        ecr_uploader = ECRUploader(
            repository_name=repository_name,
            image_tag=image_tag,
            profile_name=self.args['profile_name']
        )
        ecr_uploader.build_and_push_container(
            input_json_path=self.args['input_json_path'],
            pretrained_model_path=self.args['pretrained_model_path'],
            dockerfile_path=self._container_path / 'Dockerfile',
            entrypoint_script_path=self._container_path / 'run.py'
        )

        trainer = Trainer(
            input_json_path=self.args['input_json_path'],
            sagemaker_execution_role=self.args['sagemaker_execution_role'],
            bucket_name=self.args['s3_params']['bucket_name'],
            image_uri=ecr_uploader.image_uri,
            profile_name=self.args['profile_name'],
            local_mode=self.args['local_mode'],
            instance_type=self.args['instance_type'],
            instance_count=self.args['instance_count'],
        )
        trainer.run()


if __name__ == "__main__":
    train_mod = CloudDenoisingTrainerModule()
    train_mod.run()

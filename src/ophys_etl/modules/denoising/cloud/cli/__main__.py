from pathlib import Path

import argschema

from ophys_etl.modules.denoising.cloud.ecr import ECRUploader
from ophys_etl.modules.denoising.cloud.trainer import Trainer
from ophys_etl.modules.denoising.cloud.cli.schemas import \
    CloudDenoisingTrainerSchema


class CloudDenoisingTrainerModule(argschema.ArgSchemaParser):
    default_schema = CloudDenoisingTrainerSchema

    def run(self):
        repository_name = self.args['docker_params']['repository_name']
        image_tag = self.args['docker_params']['image_tag']

        ecr_uploader = ECRUploader(
            repository_name=repository_name,
            image_tag=image_tag,
            profile_name=self.args['profile_name']
        )
        ecr_uploader.build_and_push_container(
            dockerfile_dir=(Path(__file__).parent / 'container')
        )

        trainer = Trainer(
            sagemaker_execution_role=self.args['sagemaker_execution_role'],
            bucket_name=self.args['s3_params']['bucket_name'],
            image_uri=f'{repository_name}:{image_tag}',
            profile_name=self.args['profile_name'],
            local_mode=self.args['local_mode'],
            instance_type=self.args['instance_type'],
            instance_count=self.args['instance_count'],
            local_input_data_dir=self.args['local_data_dir']
        )
        trainer.run()


if __name__ == "__main__":
    train_mod = CloudDenoisingTrainerModule(input_data={'local_mode': True})
    train_mod.run()

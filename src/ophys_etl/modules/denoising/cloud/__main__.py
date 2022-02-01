from pathlib import Path

import argschema

from ophys_etl.modules.denoising.cloud.container.ecr import ECRUploader


class CloudDenoisingTrainerModule(argschema.ArgSchemaParser):
    def run(self):
        ecr_uploader = ECRUploader(
            repository_name='train_deepinterpolation',
            image_tag='latest',
            profile_name='sandbox'
        )
        ecr_uploader.build_and_push_container(
            dockerfile_dir=(Path(__file__).parent / 'container')
        )


if __name__ == "__main__":
    train_mod = CloudDenoisingTrainerModule()
    train_mod.run()

from pathlib import Path

import argschema

from ophys_etl.modules.denoising.cloud.container.ecr_utils import \
    build_and_push_container


class CloudDenoisingTrainerModule(argschema.ArgSchemaParser):
    def run(self):

        # Note that if nothing changed to the Dockerfile compared with the
        # same tag on ECR, then nothing is pushed
        build_and_push_container(
            repository_name='train_deepinterpolation',
            dockerfile_dir=(Path(__file__).parent / 'container'),
            image_tag='latest',
            profile_name='sandbox'
        )


if __name__ == "__main__":
    train_mod = CloudDenoisingTrainerModule()
    train_mod.run()

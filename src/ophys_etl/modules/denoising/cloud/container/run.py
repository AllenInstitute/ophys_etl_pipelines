import json
import logging
import os
import subprocess
import sys
import argparse
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    _update_paths_to_container_paths()

    cmd = 'python -m deepinterpolation.cli.fine_tuning --input_json ' \
          '/opt/ml/input.json'
    cmd = cmd.split(' ')
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               env=os.environ)
    stdout, stderr = process.communicate()
    logger.info(stdout.decode('utf-8'))

    if process.returncode != 0:
        stderr = stderr.decode('utf-8')
        with open(os.path.join('/opt/ml/model', 'failure'), 'w') as s:
            s.write(str(stderr))

        # Also log
        logger.error(stderr)

        # Cause the training job to fail
        sys.exit(255)


def _update_paths_to_container_paths():
    """Update paths from local paths to container paths"""

    def update_input_json():
        data_metadata_paths = []
        with open('/opt/ml/input.json') as f:
            input_json = json.load(f)

        # Update pretrained model path
        input_json['finetuning_params']['model_source']['local_path'] = \
            '/opt/ml/pretrained_model.h5'

        # Update output path
        input_json['finetuning_params']['output_dir'] = '/opt/ml/model'

        # Update data metadata paths
        for p in ('generator_params', 'test_generator_params'):
            cur_path = input_json[p]['train_path']
            new_path = f'/opt/ml/{Path(cur_path).name}'
            input_json[p]['train_path'] = new_path
            data_metadata_paths.append(new_path)

        # Write out updated input json
        with open('/opt/ml/input.json', 'w') as f:
            f.write(json.dumps(input_json))
        return data_metadata_paths

    def update_data_paths(data_metadata_paths: List[str]):
        """
        Parameters
        ----------
        data_metadata_paths
            Paths to data metadata input jsons. There should be 1 file for
            train and a separate file for validation
        Returns
        -------
        None, writes file
        """
        for p in data_metadata_paths:
            with open(p) as f:
                input_json = json.load(f)
            for exp_id in input_json:
                cur_path = input_json[exp_id]['path']
                new_path = f'/opt/ml/input/data/training/' \
                           f'{Path(cur_path).name}'
                input_json[exp_id]['path'] = new_path
            with open(p, 'w') as f:
                f.write(json.dumps(input_json))

    data_metadata_paths = update_input_json()
    update_data_paths(data_metadata_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train',
        help='This is a dummy argument that doesn\'t do '
        'anything. It is required due to the way '
        'sagemaker runs the container. Sagemaker always passes an argument of '
        '"train". See https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html')   # noqa E501
    args = parser.parse_args()

    main()

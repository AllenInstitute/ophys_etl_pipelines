import os
import shutil
from pathlib import Path

import tempfile

import boto3
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.pipeline_module import OutputFile
from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .mlflow_utils \
    import MLFlowRun


def download_trained_model(
    mlflow_run_name: str,
    model_dest: OutputFile
):
    """
    Downloads trained model from s3 using mlflow_run_name to look up the
    associated sagemaker training jobs

    Parameters
    ----------
    mlflow_run_name
        MLFlow run name
    model_dest
        Where to save the model.

    Returns
    -------
    None, just saves model to disk
    """
    s3 = boto3.client('s3')

    mlflow_parent_run = MLFlowRun(
        mlflow_experiment_name=(
            app_config.pipeline_steps.roi_classification.tracking.
            mlflow_experiment_name),
        run_name=mlflow_run_name
    )

    for run in mlflow_parent_run.child_runs:
        model_s3_bucket, model_s3_key = run.s3_model_save_path
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Download compressed file to tmp dir
            tmp_dest = str(Path(temp_dir) / Path(model_s3_key).name)
            s3.download_file(
                Bucket=model_s3_bucket,
                Key=model_s3_key,
                Filename=tmp_dest
            )

            # 2. Unpack compressed file
            shutil.unpack_archive(filename=tmp_dest,
                                  extract_dir=temp_dir)

            # 3. Move expected output files to permanent location
            os.makedirs(model_dest.path, exist_ok=True)
            shutil.move(
                src=Path(temp_dir) / run.fold / f'{run.fold}.pt',
                dst=model_dest.path)

    assert len(os.listdir(model_dest.path)) == \
           app_config.pipeline_steps.roi_classification.training.n_folds, \
           'We expect there to be a model per fold'

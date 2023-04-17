import os
import shutil
from pathlib import Path

import boto3
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.output_file import OutputFile
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
            app_config.pipeline_steps.roi_classification.training.tracking.
            mlflow_experiment_name),
        run_name=mlflow_run_name
    )

    os.makedirs(model_dest.path, exist_ok=True)

    for run in mlflow_parent_run.child_runs:
        model_s3_bucket, model_s3_key = run.s3_model_save_path
        s3.download_file(
            Bucket=model_s3_bucket,
            Key=model_s3_key,
            # this is a compressed file ending in .tar.gz
            Filename=model_dest.path / Path(model_s3_key).name
        )

        # this will create a directory with the name of the fold with a file
        # <fold>.pt beneath it
        shutil.unpack_archive(
            filename=model_dest.path / Path(model_s3_key).name,
            extract_dir=model_dest.path)

        # move the model file to model_dest.path
        shutil.move(str(model_dest.path / run.fold / f'{run.fold}.pt'),
                    str(model_dest.path))

        # cleanup
        os.remove(str(model_dest.path / Path(model_s3_key).name))
        shutil.rmtree(str(model_dest.path / run.fold))

    assert len(os.listdir(model_dest.path)) == \
           app_config.pipeline_steps.roi_classification.training.n_folds, \
           'We expect there to be a model per fold'

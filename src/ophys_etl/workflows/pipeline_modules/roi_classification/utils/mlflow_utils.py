"""Utils for interacting with mlflow"""
import re
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import boto3
import mlflow
from mlflow.entities import Run

from ophys_etl.workflows.app_config.app_config import app_config


class MLFlowRun:
    """A single MLFlow run"""
    def __init__(
        self,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        """

        Parameters
        ----------
        mlflow_experiment_name
            MLFlow experiment name. Provide this or mlflow_experiment_id
        mlflow_experiment_id
            MLFlow experiment id. Provide this or mlflow_experiment_name
        run_name
            MLFlow run name. Provide this or run_id
        run_id
            MLFlow run id. Provide this or run_name
        """
        mlflow.set_tracking_uri(
            app_config.pipeline_steps.roi_classification.training.tracking.
            mlflow_server_uri)
        if mlflow_experiment_name is not None:
            self._experiment_id = self._get_experiment_id(
                mlflow_experiment_name=mlflow_experiment_name
            )
        elif mlflow_experiment_id is not None:
            self._experiment_id = mlflow_experiment_id
        else:
            raise ValueError('provide either mlflow_experiment_id or '
                             'mlflow_experiment_name')

        if run_name is not None:
            run = self._get_run(
                mlflow_experiment_id=self._experiment_id,
                run_name=run_name
            )
        elif run_id is not None:
            run = mlflow.get_run(
                run_id=run_id
            )
        else:
            raise ValueError('provide either run_name or run_id')
            
        self._run = run

    @staticmethod
    def _get_experiment_id(
        mlflow_experiment_name: str
    ) -> str:
        """
        Gets the mlflow experiment id given name

        Parameters
        ----------
        mlflow_experiment_name

        Returns
        -------
        mlflow experiment id
        """
        experiment = mlflow.get_experiment_by_name(
            name=mlflow_experiment_name
        )
        return experiment.experiment_id

    @staticmethod
    def _get_run(
        mlflow_experiment_id: str,
        run_name: str
    ) -> Run:
        """
        Gets the mlfow run_id given `run_name`

        Returns
        -------
        mlflow Run

        Raises
        ------
        ValueError
            If no run_id can be found for run_name
        """
        runs: List[Run] = mlflow.search_runs(
            experiment_ids=[mlflow_experiment_id],
            output_format='list'
        )

        for run in runs:
            if run.data.tags['mlflow.runName'] == run_name:
                return run

        raise ValueError(
            f'Could not find an mlflow run {run_name}')

    @property
    def run(self) -> Run:
        """

        Returns
        -------
        MLFlow run
        """
        return self._run
    
    @property
    def child_runs(self) -> List["MLFlowRun"]:
        """

        Returns
        -------
        List of child `Run` under self._run
        """
        runs: List[Run] = mlflow.search_runs(
            experiment_ids=[self._experiment_id],
            output_format='list'
        )
        child_runs = []
        for run in runs:
            if 'mlflow.parentRunId' in run.data.tags and \
                    run.data.tags['mlflow.parentRunId'] == \
                    self._run.info.run_id:
                run = MLFlowRun(
                    mlflow_experiment_id=self._experiment_id,
                    run_name=run.data.tags['mlflow.runName']
                )
                child_runs.append(run)
        return child_runs

    @property
    def sagemaker_job_id(self) -> str:
        """

        Returns
        -------
        sagemaker job id

        Raises
        ------
        ValueError
            if can't get it
        """
        sagemaker_job = self._run.data.tags.get('sagemaker_job', None)
        if sagemaker_job is None:
            raise ValueError(f'Could not find sagemaker job for '
                             f'{self._run.info.run_id}')
        return sagemaker_job

    @property
    def s3_model_save_path(self) -> Tuple[str, str]:
        """
        Gets where model is saved on s3 for this run

        Returns
        -------
        Tuple
            bucket, key
        """
        sagemaker = boto3.client('sagemaker')
        training_job_meta = sagemaker.describe_training_job(
            TrainingJobName=self.sagemaker_job_id
        )
        model_path = \
            training_job_meta['ModelArtifacts']['S3ModelArtifacts']
        model_path = urlparse(url=model_path)
        key = Path(model_path.path.lstrip('/'))
        return model_path.netloc, str(key)
    
    @property
    def fold(self) -> str:
        """Return fold this model run was trained on
        Assumes run name formatted like fold-<fold>

        Raises
        ------
        ValueError
            if cannot parse fold
        """
        run_name = self._run.data.tags['mlflow.runName']
        fold = re.findall(r'fold-(\d)', run_name)
        if len(fold) == 0 or len(fold) > 1:
            raise ValueError(f'Could not get fold name from {run_name}')
        return fold[0]

"""Utils for interacting with mlflow"""
from typing import List, Dict

import requests

from ophys_etl.workflows.app_config.app_config import app_config


class MLFlowRun:
    def __init__(self):
        self._run_id = self._get_run_id()

    @staticmethod
    def _get_run_id():
        """
        Gets the mlfow run_id given `run_name`

        Returns
        -------
        mlflow run_id

        Raises
        ------
        ValueError
            If no run_id can be found for run_name
        """
        settings = app_config.pipeline_steps.roi_classification.inference

        r = requests.post(
            url=f'http://{settings.tracking_uri}:80/api/2.0/mlflow/runs/'
                f'search',
            json={'experiment_ids': [settings.mlflow_experiment_id]})

        res = r.json()
        runs = res['runs']

        for run in runs:
            tags = run['data']['tags']
            for tag in tags:
                if tag['key'] == 'mlflow.runName':
                    if tag['value'] == settings.mlflow_run_name:
                        return run['info']['run_id']

        raise ValueError(
            f'Could not find an mlflow run {settings.mlflow_run_name}')

    @property
    def run_params(self) -> List[Dict]:
        """

        Returns
        -------
        Params used for `self._run_id`
        """
        settings = app_config.pipeline_steps.roi_classification.inference

        r = requests.get(
            f'http://{settings.mlflow_tracking_server_uri}:80/'
            f'api/2.0/mlflow/runs/get?run_id={self._run_id}')
        run = r.json()
        return run['data']['params']

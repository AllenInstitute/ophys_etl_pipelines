import pickle
from pathlib import Path
from unittest.mock import patch, create_autospec

import botocore
import pytest

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent.parent.parent / 'resources' /
            'config.yml'),
    test_di_base_model_path=(Path(__file__).parent.parent.parent.parent /
                             'resources' / 'di_model.h5')
)


from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .mlflow_utils import \
    MLFlowRun


class TestMLFlowUtils:
    @classmethod
    def setup_class(cls):
        with open(Path(__file__).parent / 'resources' /
                  'mlflow_search_runs.pkl', 'rb') as f:
            # created using  mlflow.search_runs(..., output_format='list')
            cls._dummy_mlflow_search_runs_res = pickle.load(f)

        with open(Path(__file__).parent / 'resources' /
                  'sagemaker_describe_training_job.pkl', 'rb') as f:
            # created using  sagemaker.describe_training_job(
            # TrainingJobName=...)
            cls._sagemaker_describe_training_job_res = pickle.load(f)

    @pytest.mark.parametrize('run_name', ('CV-1678301354', 'bad'))
    @patch('mlflow.search_runs')
    def test_get_run(self, mock_search_runs, run_name):
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res
        if run_name == 'bad':
            with pytest.raises(ValueError):
                MLFlowRun._get_run(
                    mlflow_experiment_id='foo',
                    run_name=run_name
                )
        else:
            run = MLFlowRun._get_run(
                mlflow_experiment_id='foo',
                run_name=run_name
            )
            assert run.data.tags['mlflow.runName'] == run_name

    @pytest.mark.parametrize('run_name', ('CV-1678301354', 'fold-0'))
    @patch('mlflow.search_runs')
    def test_child_runs(self, mock_search_runs, run_name):
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res

        run = MLFlowRun(
            mlflow_experiment_id='foo',
            run_name=run_name
        )

        if run.run.data.tags.get('is_parent', False):
            assert len(run.child_runs) == 5
        else:
            assert len(run.child_runs) == 0

    @patch('mlflow.search_runs')
    def test_sagemaker_job_id(self, mock_search_runs):
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res

        run = MLFlowRun(
            mlflow_experiment_id='foo',
            run_name='fold-0'
        )
        assert run.sagemaker_job_id == 'deepcell-train-fold-0-1678301355'

    @patch('boto3.client')
    @patch('mlflow.search_runs')
    def test_s3_model_save_path(self, mock_search_runs, mock_sagemaker):
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res

        sagemaker_describe_training_job_res = \
            self._sagemaker_describe_training_job_res

        class DummySagemaker:
            @staticmethod
            def describe_training_job(TrainingJobName):
                return sagemaker_describe_training_job_res

        mock_sagemaker.return_value = DummySagemaker

        run = MLFlowRun(
            mlflow_experiment_id='foo',
            run_name='fold-3'
        )
        assert run.s3_model_save_path == (
            'dev.deepcell.alleninstitute.org',
            'deepcell-train-fold-3-1678301372/output/model.tar.gz')

    @pytest.mark.parametrize('run_name', ('CV-1678301354', 'fold-0'))
    @patch('mlflow.search_runs')
    def test_fold(self, mock_search_runs, run_name):
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res

        run = MLFlowRun(
            mlflow_experiment_id='foo',
            run_name=run_name
        )

        if run.run.data.tags.get('is_parent', False):
            with pytest.raises(ValueError):
                run.fold
        else:
            assert run.fold == run_name[-1]

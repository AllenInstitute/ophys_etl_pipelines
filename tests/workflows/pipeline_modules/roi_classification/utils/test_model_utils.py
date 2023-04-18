import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch


from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent.parent.parent / 'resources' /
            'config.yml'),
    test_di_base_model_path=(Path(__file__).parent.parent.parent.parent /
                             'resources' / 'di_model.h5')
)

from ophys_etl.workflows.app_config.app_config import app_config    # noqa E402

from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .mlflow_utils import \
    MLFlowRun   # noqa E402
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum # noqa E402

from ophys_etl.workflows.output_file import OutputFile  # noqa E402

from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .model_utils import \
    download_trained_model  # noqa E402


class TestModelUtils:
    @classmethod
    def setup_class(cls):
        with open(Path(__file__).parent.parent / 'resources' /
                  'mlflow_search_runs.pkl', 'rb') as f:
            # created using  mlflow.search_runs(..., output_format='list')
            cls._dummy_mlflow_search_runs_res = pickle.load(f)

        with open(Path(__file__).parent.parent / 'resources' /
                  'sagemaker_describe_training_job.pkl', 'rb') as f:
            # created using  sagemaker.describe_training_job(
            # TrainingJobName=...)
            cls._sagemaker_describe_training_job_res = pickle.load(f)

    @patch('boto3.client')
    @patch('mlflow.search_runs')
    @patch.object(MLFlowRun, '_get_experiment_id',
                  wraps=lambda mlflow_experiment_name: 'foo')
    def test_download_trained_model(
            self,
            __,
            mock_search_runs,
            mock_boto3_client
    ):
        """Tests that download_trained_model works as expected.
        We have to mock the downloaded tar.gz file"""
        sagemaker_describe_training_job_res = \
            self._sagemaker_describe_training_job_res

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dest = OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL),
                path=Path(tmp_dir) / 'model'
            )

            class DummyBoto3Client:
                @staticmethod
                def download_file(
                    Bucket,
                    Key,
                    Filename
                ):
                    with tempfile.TemporaryDirectory() as tmp_dir2:
                        fold = re.findall(r'fold-(\d)', Key)[0]
                        dst = (Path(tmp_dir2) / fold)
                        os.makedirs(dst)
                        with open(dst / f'{fold}.pt', 'w') as f:
                            f.write('')
                        shutil.make_archive(
                            base_name=str(
                                model_dest.path / model_dest.path.name),
                            format='gztar',
                            root_dir=Path(tmp_dir2))

                @staticmethod
                def describe_training_job(TrainingJobName):
                    # we are using a single dummy describe training job res
                    # mock the fold by replacing the key with the expected fold
                    fold = re.findall(r'fold-(\d)', TrainingJobName)[0]
                    res = sagemaker_describe_training_job_res
                    res['ModelArtifacts']['S3ModelArtifacts'] = \
                        re.sub(r'fold-\d', f'fold-{fold}',
                               res['ModelArtifacts']['S3ModelArtifacts'])
                    return res

            mock_search_runs.return_value = self._dummy_mlflow_search_runs_res
            mock_boto3_client.return_value = DummyBoto3Client

            download_trained_model(
                mlflow_run_name='CV-1678301354',
                model_dest=model_dest
            )

            n_folds = app_config.pipeline_steps.roi_classification.training.\
                n_folds
            assert len(os.listdir(model_dest.path)) == n_folds
            assert set(os.listdir(model_dest.path)) == \
                   set([f'{i}.pt' for i in range(n_folds)])

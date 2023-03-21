import datetime
import os
import pickle
import shutil
from pathlib import Path

import tempfile
from unittest.mock import patch

import json

import pandas as pd
import random

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent.parent / 'resources' / 'config.yml'),
    test_di_base_model_path=Path(__file__).parent.parent.parent / 'resources' /
    'di_model.h5',
)

from ophys_etl.workflows.pipeline_modules.roi_classification.inference import \
    InferenceModule # noqa E402
from ophys_etl.workflows.pipeline_modules.roi_classification import \
    TrainingModule  # noqa E402
from ophys_etl.workflows.pipeline_modules.segmentation import \
    SegmentationModule  # noqa E402
from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .mlflow_utils import \
    MLFlowRun   # noqa E402
from ophys_etl.workflows.app_config.app_config import app_config    # noqa E402
from ophys_etl.workflows.workflow_names import WorkflowName # noqa E402
from ophys_etl.workflows.pipeline_module import OutputFile  # noqa E402

from ophys_etl.workflows.db.db_utils import save_job_run_to_db  # noqa E402
from sqlmodel import create_engine, Session, select # noqa E402

from ophys_etl.workflows.db.initialize_db import IntializeDBRunner  # noqa E402
from ophys_etl.workflows.well_known_file_types import WellKnownFileType # noqa E402
from ophys_etl.workflows.workflow_steps import WorkflowStep # noqa E402


class TestInference:
    @classmethod
    def setup_class(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / 'app.db'
        os.environ['AIRFLOW_CONN_OPHYS_WORKFLOW_DB'] = str(cls._db_path)
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f'sqlite:///{cls._db_path}'
        cls._engine = IntializeDBRunner(
            input_data={
                'db_url': db_url
            },
            args=[]).run()

        # initialize model files
        model_path = Path(cls._tmp_dir) / 'model'
        os.makedirs(model_path)

        cls._mlflow_parent_run_name = 'CV-1678301354'

        cls._n_folds = \
            app_config.pipeline_steps.roi_classification.training.n_folds
        for fold in range(cls._n_folds):
            with open(model_path / f'{fold}.pt', 'w') as f:
                f.write('')
        cls._model_path = model_path

        with open(Path(__file__).parent / 'resources' /
                  'mlflow_search_runs.pkl', 'rb') as f:
            # created using  mlflow.search_runs(..., output_format='list')
            cls._dummy_mlflow_search_runs_res = pickle.load(f)

        cls._rois = cls._insert_rois()
        cls._ophys_experiment_id = '1'
        cls._preds_path = \
            Path(cls._tmp_dir) / f'{cls._ophys_experiment_id}_inference.csv'
        pd.DataFrame({
            'roi-id': roi['id'],
            'y_score': random.random()
        } for roi in cls._rois).to_csv(cls._preds_path, index=False)
        cls._insert_model()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmp_dir)

    @patch('mlflow.search_runs')
    @patch.object(MLFlowRun, '_get_experiment_id',
                  wraps=lambda mlflow_experiment_name: 'foo')
    def test_save_predictions_to_db(self, __, mock_search_runs):
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res

        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStep.ROI_CLASSIFICATION_INFERENCE,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[OutputFile(
                        well_known_file_type=(
                            WellKnownFileType.
                            ROI_CLASSIFICATION_EXPERIMENT_PREDICTIONS),
                        path=self._preds_path
                    )
                ],
                ophys_experiment_id='1',
                sqlalchemy_session=session,
                storage_directory='/foo',
                additional_steps=InferenceModule.save_predictions_to_db,
                additional_steps_kwargs={
                    # only 1 inserted, so we can assume id is 1
                    'ensemble_id': 1
                },
                workflow_name=WorkflowName.OPHYS_PROCESSING
            )

    @classmethod
    def _insert_rois(cls):
        rois_path = Path(__file__).parent.parent / 'resources' / 'rois.json'
        with open(rois_path) as f:
            rois = json.load(f)

        with Session(cls._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStep.SEGMENTATION,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[OutputFile(
                        well_known_file_type=(
                            WellKnownFileType.OPHYS_ROIS),
                        path=rois_path
                    )
                ],
                ophys_experiment_id='1',
                sqlalchemy_session=session,
                storage_directory='/foo',
                additional_steps=SegmentationModule.save_rois_to_db,
                workflow_name=WorkflowName.OPHYS_PROCESSING
            )
        return rois

    @classmethod
    @patch('mlflow.search_runs')
    @patch.object(MLFlowRun, '_get_experiment_id',
                  wraps=lambda mlflow_experiment_name: 'foo')
    def _insert_model(cls, __, mock_search_runs):
        mock_search_runs.return_value = cls._dummy_mlflow_search_runs_res

        with Session(cls._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStep.ROI_CLASSIFICATION_TRAINING,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[OutputFile(
                        well_known_file_type=(
                            WellKnownFileType.
                            ROI_CLASSIFICATION_TRAINED_MODEL),
                        path=cls._model_path
                    )
                ],
                ophys_experiment_id='1',
                sqlalchemy_session=session,
                storage_directory='/foo',
                additional_steps=TrainingModule.save_trained_model_to_db,
                additional_steps_kwargs={
                  'mlflow_parent_run_name': cls._mlflow_parent_run_name
                },
                workflow_name=WorkflowName.ROI_CLASSIFIER_TRAINING
            )

import datetime
import json
import os
import pickle
import random
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule

from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession, Specimen, OphysContainer

from ophys_etl.test_utils.workflow_utils import setup_app_config
from tests.workflows.conftest import MockSQLiteDB

setup_app_config(
    ophys_workflow_app_config_path=(
        Path(__file__).parent.parent.parent / "resources" / "config.yml"
    ),
    test_di_base_model_path=Path(__file__).parent.parent.parent
    / "resources"
    / "di_model.h5",
)

from sqlmodel import Session

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.roi_classification import (
    TrainingModule,
)
from ophys_etl.workflows.pipeline_modules.roi_classification.inference import (
    InferenceModule,
)
from ophys_etl.workflows.pipeline_modules.roi_classification.utils.mlflow_utils import ( # noqa E501
    MLFlowRun,
)
from ophys_etl.workflows.pipeline_modules.segmentation import (
    SegmentationModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class TestInference(MockSQLiteDB):
    def setup(self):
        super().setup()

        # initialize model files
        model_path = Path(self._tmp_dir) / "model"
        os.makedirs(model_path)

        self._mlflow_parent_run_name = "CV-1678301354"

        self._n_folds = (
            app_config.pipeline_steps.roi_classification.training.n_folds
        )
        for fold in range(self._n_folds):
            with open(model_path / f"{fold}.pt", "w") as f:
                f.write("")
        self._model_path = model_path
        self._ophys_experiment_id = "1"

        with open(
            Path(__file__).parent / "resources" / "mlflow_search_runs.pkl",
            "rb",
        ) as f:
            # created using  mlflow.search_runs(..., output_format='list')
            self._dummy_mlflow_search_runs_res = pickle.load(f)

        xy_offset_path = (
            Path(__file__).parent.parent / "resources" /
            "rigid_motion_transform.csv"
        )

        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.
                            MOTION_CORRECTED_IMAGE_STACK
                        ),
                        path=Path(f'{self._ophys_experiment_id}_'
                                  f'motion_correction.h5'),
                    ),
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.
                            MAX_INTENSITY_PROJECTION_IMAGE
                        ),
                        path=Path(f'{self._ophys_experiment_id}_max_proj.png'),
                    ),
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                        ),
                        path=xy_offset_path
                    )
                ],
                ophys_experiment_id=self._ophys_experiment_id,
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                validate_files_exist=False,
                additional_steps=MotionCorrectionModule.save_metadata_to_db
            )

        self._rois = self._insert_rois()
        self._preds_path = (
            Path(self._tmp_dir) / f"{self._ophys_experiment_id}_inference.csv"
        )
        pd.DataFrame(
            {"roi-id": roi["id"], "y_score": random.random()}
            for roi in self._rois
        ).to_csv(self._preds_path, index=False)
        self._insert_model()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmp_dir)

    @patch("mlflow.search_runs")
    @patch.object(
        MLFlowRun,
        "_get_experiment_id",
        wraps=lambda mlflow_experiment_name: "foo",
    )
    def test_save_predictions_to_db(self, __, mock_search_runs):
        """Checks that predictions are saved to the db and that we can
        fetch rois with the expected classification flag"""
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res

        with patch('ophys_etl.workflows.pipeline_modules.roi_classification.'
                   'inference.engine', new=self._engine):
            with Session(self._engine) as session:
                save_job_run_to_db(
                    workflow_step_name=WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE, # noqa E501
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.ROI_CLASSIFICATION_EXPERIMENT_PREDICTIONS # noqa E501
                            ),
                            path=self._preds_path,
                        )
                    ],
                    ophys_experiment_id="1",
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    additional_steps=InferenceModule.save_predictions_to_db,
                    additional_steps_kwargs={
                        # only 1 inserted, so we can assume id is 1
                        "ensemble_id": 1
                    },
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                )

            oe = OphysExperiment(
                id=1,
                movie_frame_rate_hz=11.0,
                raw_movie_filename=Path('foo'),
                session=OphysSession(id=1, specimen=Specimen(id='1')),
                container=OphysContainer(id=1, specimen=Specimen(id='1')),
                specimen=Specimen(id='1'),
                storage_directory=Path('foo'),
                equipment_name='MESO.1',
                full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt"
            )

            with patch('ophys_etl.workflows.ophys_experiment.engine',
                       new=self._engine):
                rois = oe.rois

            preds = pd.read_csv(self._preds_path).set_index('roi-id')
            for roi in rois:
                assert roi._is_cell == (
                    preds.loc[roi.id]['y_score'] >
                    app_config.pipeline_steps.roi_classification.inference.
                    classification_threshold
                    )

    def _insert_rois(self):
        rois_path = Path(__file__).parent.parent / "resources" / "rois.json"
        with open(rois_path) as f:
            rois = json.load(f)

        with patch.object(OphysExperiment, 'from_id') as mock_oe_from_id:
            mock_oe_from_id.return_value = OphysExperiment(
                id=1,
                movie_frame_rate_hz=11.0,
                raw_movie_filename=Path('foo'),
                session=OphysSession(id=1, specimen=Specimen(id='1')),
                container=OphysContainer(id=1, specimen=Specimen(id='1')),
                specimen=Specimen(id='1'),
                storage_directory=Path('foo'),
                equipment_name='MESO.1',
                full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
            )
            with Session(self._engine) as session:
                with patch('ophys_etl.workflows.ophys_experiment.engine',
                           new=self._engine):
                    save_job_run_to_db(
                        workflow_step_name=WorkflowStepEnum.SEGMENTATION,
                        start=datetime.datetime.now(),
                        end=datetime.datetime.now(),
                        module_outputs=[
                            OutputFile(
                                well_known_file_type=(
                                    WellKnownFileTypeEnum.OPHYS_ROIS
                                ),
                                path=rois_path,
                            )
                        ],
                        ophys_experiment_id="1",
                        sqlalchemy_session=session,
                        storage_directory="/foo",
                        log_path="/foo",
                        additional_steps=SegmentationModule.save_rois_to_db,
                        workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    )
        return rois

    @patch("mlflow.search_runs")
    @patch.object(
        MLFlowRun,
        "_get_experiment_id",
        wraps=lambda mlflow_experiment_name: "foo",
    )
    def _insert_model(self, __, mock_search_runs):
        mock_search_runs.return_value = self._dummy_mlflow_search_runs_res

        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING, # noqa E501
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL # noqa E501
                        ),
                        path=self._model_path,
                    )
                ],
                ophys_experiment_id="1",
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                additional_steps=TrainingModule.save_trained_model_to_db,
                additional_steps_kwargs={
                    "mlflow_parent_run_name": self._mlflow_parent_run_name
                },
                workflow_name=WorkflowNameEnum.ROI_CLASSIFIER_TRAINING,
            )

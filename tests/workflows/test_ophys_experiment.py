import os
import shutil
from pathlib import Path

import tempfile

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent / 'resources' / 'config.yml'),
    test_di_base_model_path=Path(__file__).parent / 'resources' / 'di_model.h5'
)

from ophys_etl.workflows.db.initialize_db import InitializeDBRunner # noqa E402
from sqlmodel import Session
from ophys_etl.test_utils.db_base import MockSQLiteDB
from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession, Specimen
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.workflow_step_runs import get_workflow_step_by_name
from ophys_etl.workflows.db.schemas import WorkflowStepRun, MotionCorrectionRun, \
    OphysROI, OphysROIMaskValue


class TestOphysExperiment:
    @classmethod
    def _initializeDB(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / 'app.db'
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f'sqlite:///{cls._db_path}'
        cls._engine = InitializeDBRunner(
            input_data={
                'db_url': db_url
            },
            args=[]).run()

    
    def _create_mock_data(self):
        workflow_name = WorkflowNameEnum.OPHYS_PROCESSING.value
        workflow_step_name = WorkflowStepEnum.SEGMENTATION.value
        ophys_experiment_id="1"
        with Session(self._engine) as session:

            workflow_step = get_workflow_step_by_name(
                session=session,
                workflow=workflow_name,
                name=workflow_step_name
            )
            workflow_step_run = WorkflowStepRun(
                ophys_experiment_id=ophys_experiment_id,
                workflow_step_id=workflow_step.id,
                log_path="log_path",
                storage_directory="storage_directory",
                start="2023-04-01 00:00:00",
                end="2023-04-01 01:00:00",
            )
            session.add(workflow_step_run)
            session.flush()
            motion_correction = MotionCorrectionRun(
                workflow_step_run_id=workflow_step_run.id,
                max_correction_left=10,
                max_correction_right=20,
                max_correction_up=30,
                max_correction_down=40,
            )
            session.add(motion_correction)

            ophys_roi = OphysROI(
                workflow_step_run_id=1,
                x=10,
                y=20,
                width=30,
                height=40,
            )
            session.add(ophys_roi)
            session.flush()

            ophys_roi_mask_value = OphysROIMaskValue(
                ophys_roi_id=1,
                row_index=5,
                col_index=6,
            )
            session.add(ophys_roi_mask_value)

            ophys_roi_mask_value = OphysROIMaskValue(
                ophys_roi_id=1,
                row_index=6,
                col_index=8,
            )
            session.add(ophys_roi_mask_value)
            session.commit()
    
    def setup(self):
        self._initializeDB()
        self._create_mock_data()
        self.ophys_experiment = OphysExperiment(
                    id='1',
                    session=OphysSession(id='2'),
                    specimen=Specimen(id='3'),
                    storage_directory=Path('/storage_dir'),
                    raw_movie_filename=Path('mov.h5'),
                    movie_frame_rate_hz=11.0
                )

    def teardown_method(cls):
        shutil.rmtree(cls._tmp_dir)


    def test__get_ophys_experiment_roi_metadata(self):
        with Session(self._engine) as session:
            roi_metadata = self.ophys_experiment.get_ophys_experiment_roi_metadata(
                session=session,
            )
        assert len(roi_metadata) == 1
        assert roi_metadata[0]["x"] == 10
        assert roi_metadata[0]["y"] == 20
        assert roi_metadata[0]["width"] == 30
        assert roi_metadata[0]["height"] == 40
        assert roi_metadata[0]["mask"][5][6] == 1
        assert len(roi_metadata[0]["mask"]) == 40
        assert len(roi_metadata[0]["mask"][0]) == 30
    
    def test_get_ophys_experiment_motion_border(self):
        with Session(self._engine) as session:
            motion_border = self.ophys_experiment.get_ophys_experiment_motion_border(
                session=session
            )
        assert motion_border['x0'] == 10
        assert motion_border['x1'] == 20
        assert motion_border['y0'] == 30
        assert motion_border['y1'] == 40
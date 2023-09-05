import datetime
import h5py
import shutil
import tempfile
import numpy as np
import os
import pytest
from pathlib import Path
from sqlmodel import Session
from unittest.mock import patch

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
        Path(__file__).parent / "resources" / "config.yml"
    ),
    test_di_base_model_path=(
        Path(__file__).parent / "resources" / "di_model.h5"
    ),
)

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.db.initialize_db import InitializeDBRunner
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,
    OphysSession,
    Specimen, OphysContainer)
from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue, MotionCorrectionRun
from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule

class MockSQLiteDB:
    @classmethod
    def _initializeDB(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / "app.db"
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f"sqlite:///{cls._db_path}"
        cls._engine = InitializeDBRunner(
            input_data={"db_url": db_url}, args=[]
        ).run()

    def setup(self):
        self._initializeDB()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmp_dir)

@pytest.fixture
def experiment_ids():
    return [1, 2]

@pytest.fixture
def workflow_step_run_id():
    return 1

@pytest.fixture
def temp_dir():
    yield Path(tempfile.TemporaryDirectory().name)

@pytest.fixture
def motion_corrected_ophys_movie_path():
    _, motion_corrected_ophys_movie_path = tempfile.mkstemp(suffix='.h5')
    motion_corrected_ophys_movie_path = Path(motion_corrected_ophys_movie_path)
    with h5py.File(motion_corrected_ophys_movie_path, 'w') as f:
        f.create_dataset("dummy_data", data=[1.0, 2.0, 3.0])
    yield motion_corrected_ophys_movie_path

@pytest.fixture
def trace_path():
    _, trace_path = tempfile.mkstemp(suffix='.h5')
    trace_path = Path(trace_path)
    with h5py.File(trace_path, 'w') as h5:
        h5.create_dataset('roi_names', np.arange(10))
    yield trace_path

@pytest.fixture
def mock_ophys_session():
    return OphysSession(
                id=1,
                specimen=Specimen(id='specimen_1')
            )

@pytest.fixture
def mock_ophys_experiment(experiment_ids):
    return OphysExperiment(
        id=experiment_ids[0],
        movie_frame_rate_hz=31.0,
        raw_movie_filename=Path('foo'),
        session=OphysSession(id=1, specimen=Specimen(id='1')),
        container=OphysContainer(id=1, specimen=Specimen(id='1')),
        specimen=Specimen(id='1'),
        storage_directory=Path('foo'),
        equipment_name='DEEPSCOPE',
        full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
    )

@pytest.fixture
def mock_rois():
    mock_roi = OphysROI(
        id=1,
        x=0,
        y=0,
        width=2,
        height=1,
        is_in_motion_border=False
    )
    mock_roi._mask_values = [
        OphysROIMaskValue(
            id=1,
            ophys_roi_id=1,
            row_index=0,
            col_index=0
        )
    ]
    return [mock_roi]

@pytest.fixture
def mock_motion_border_run(workflow_step_run_id):
    MotionCorrectionRun(
                workflow_step_run_id=workflow_step_run_id,
                max_correction_up=0.1,
                max_correction_down=0.1,
                max_correction_left=0.1,
                max_correction_right=0.1,
            )

@pytest.fixture
def xy_offset_path():
    return (
        Path(__file__).parent / "resources" / "rigid_motion_transform.csv"
    )

@pytest.fixture
def rois_path():
    return Path(__file__).parent / "resources" / "rois.json"
class BaseTestPipelineModule(MockSQLiteDB):

    def setup(self, experiment_ids,
              rois_path):
        super().setup()

        with Session(self._engine) as session:
            
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.DEMIX_TRACES,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.DEMIXED_TRACES
                        ),
                        path=trace_path,
                    )
                ],
                ophys_experiment_id=str(experiment_ids[0]),
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                validate_files_exist=False
            )

            for oe_id in experiment_ids:
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
                            path=(self._tmp_dir /
                                  f'{oe_id}_motion_correction.h5'),
                        ),
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.
                                MAX_INTENSITY_PROJECTION_IMAGE
                            ),
                            path=self._tmp_dir / f'{oe_id}_max_proj.png',
                        ),
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                            ),
                            path=xy_offset_path
                        )
                    ],
                    ophys_experiment_id=oe_id,
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    validate_files_exist=False,
                    additional_steps=MotionCorrectionModule.save_metadata_to_db
                )
                
                with patch(
                        'ophys_etl.workflows.ophys_experiment.engine',
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
                                path=rois_path
                            )
                        ],
                        ophys_experiment_id=oe_id,
                        sqlalchemy_session=session,
                        storage_directory="/foo",
                        log_path="/foo",
                        additional_steps=SegmentationModule.save_rois_to_db,
                        workflow_name=WorkflowNameEnum.OPHYS_PROCESSING
                    )

                save_job_run_to_db(
                    workflow_step_name=WorkflowStepEnum.TRACE_EXTRACTION,
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.ROI_TRACE
                            ),
                            path=Path(f'{oe_id}_roi_traces.h5'),
                        )
                    ],
                    ophys_experiment_id=oe_id,
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    validate_files_exist=False
                )

                with open(self._tmp_dir / f'{oe_id}_max_proj.png', 'w') as f:
                    f.write('')
                with open(self._tmp_dir / f'{oe_id}_motion_correction.h5',
                          'w') as f:
                    f.write('')

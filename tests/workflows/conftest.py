import datetime
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from sqlmodel import Session

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
from ophys_etl.workflows.db.schemas import (
    MotionCorrectionRun,
    OphysROI,
    OphysROIMaskValue,
)
from ophys_etl.workflows.ophys_experiment import (
    OphysContainer,
    OphysExperiment,
    OphysSession,
    Specimen,
)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

MOCK_EXPERIMENT_IDS = [1, 2]


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
    return MOCK_EXPERIMENT_IDS


@pytest.fixture
def workflow_step_run_id():
    return 1


@pytest.fixture
def temp_dir():
    yield Path(tempfile.TemporaryDirectory().name)


@pytest.fixture
def motion_corrected_ophys_movie_path():
    _, motion_corrected_ophys_movie_path = tempfile.mkstemp(suffix=".h5")
    motion_corrected_ophys_movie_path = Path(motion_corrected_ophys_movie_path)
    with h5py.File(motion_corrected_ophys_movie_path, "w") as f:
        f.create_dataset("dummy_data", data=[1.0, 2.0, 3.0])
    yield motion_corrected_ophys_movie_path


@pytest.fixture
def trace_path():
    _, trace_path = tempfile.mkstemp(suffix=".h5")
    trace_path = Path(trace_path)
    with h5py.File(trace_path, "w") as h5:
        h5.create_dataset("roi_names", np.arange(10))
    yield trace_path


@pytest.fixture
def mock_ophys_session():
    return OphysSession(id=1, specimen=Specimen(id="specimen_1"))


@pytest.fixture
def mock_ophys_experiment(experiment_ids):
    return OphysExperiment(
        id=experiment_ids[0],
        movie_frame_rate_hz=31.0,
        raw_movie_filename=Path("foo"),
        session=OphysSession(id=1, specimen=Specimen(id="1")),
        container=OphysContainer(id=1, specimen=Specimen(id="1")),
        specimen=Specimen(id="1"),
        storage_directory=Path("foo"),
        equipment_name="DEEPSCOPE",
        full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
    )


@pytest.fixture
def mock_rois():
    mock_roi = OphysROI(
        id=1, x=0, y=0, width=2, height=1, is_in_motion_border=False
    )
    mock_roi._mask_values = [
        OphysROIMaskValue(id=1, ophys_roi_id=1, row_index=0, col_index=0)
    ]
    return [mock_roi]


@pytest.fixture
def mock_thumbnails_dir(temp_dir):
    return {
        "1": temp_dir,
    }


@pytest.fixture
def mock_motion_border_run(workflow_step_run_id):
    return MotionCorrectionRun(
        workflow_step_run_id=workflow_step_run_id,
        max_correction_up=0.1,
        max_correction_down=0.1,
        max_correction_left=0.1,
        max_correction_right=0.1,
    )


@pytest.fixture
def xy_offset_path():
    return Path(__file__).parent / "resources" / "rigid_motion_transform.csv"


@pytest.fixture
def rois_path():
    return Path(__file__).parent / "resources" / "rois.json"


class BaseTestPipelineModule(MockSQLiteDB):
    @property
    def experiment_ids(self):
        return MOCK_EXPERIMENT_IDS

    def setup(self):
        super().setup()
        self.db_setup()

    def db_setup(self):
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
                ophys_experiment_id=str(self.experiment_ids[0]),
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                validate_files_exist=False,
            )

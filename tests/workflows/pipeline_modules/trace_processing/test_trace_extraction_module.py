import datetime
from pathlib import Path
from unittest.mock import patch, PropertyMock
import tempfile
import h5py
import numpy as np

from tests.workflows.conftest import MockSQLiteDB
from sqlmodel import Session

from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,
    OphysSession,
    Specimen, OphysContainer)
from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue, MotionCorrectionRun
from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.pipeline_modules.trace_processing.trace_extraction import TraceExtractionModule  # noqa E501

class TestTraceExtractionModule(MockSQLiteDB):

    def setup_class(cls):
        cls._experiment_id = 1
        cls._workflow_step_run_id = 1

    def setup(self):
        super().setup()

        _, motion_corrected_movie_path = tempfile.mkstemp(suffix='.h5')
        self.motion_corrected_movie_path = Path(motion_corrected_movie_path)
        self.temp_dir = self.motion_corrected_movie_path

        with h5py.File(self.motion_corrected_movie_path, 'w') as f:
            f.create_dataset("dummy_data", data=[1.0, 2.0, 3.0])

        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                        ),
                        path=self.motion_corrected_movie_path,
                    )
                ],
                ophys_experiment_id=str(self._experiment_id),
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                validate_files_exist=False
            )

    def teardown(self):
        self.motion_corrected_movie_path.unlink()

    @patch.object(OphysExperiment, 'motion_border',
                  new_callable=PropertyMock)    
    @patch.object(OphysExperiment, 'rois',
                  new_callable=PropertyMock)
    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(TraceExtractionModule, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir,
                    mock_rois,
                    mock_motion_border):
        """Test that inputs are correctly formatted for input into the module.
        """
        mock_motion_border.return_value = MotionCorrectionRun(
                workflow_step_run_id=self._workflow_step_run_id,
                max_correction_up=0.1,
                max_correction_down=0.1,
                max_correction_left=0.1,
                max_correction_right=0.1,
            )
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
        mock_rois.return_value = [mock_roi]
        mock_output_path.return_value = self._tmp_dir
        mock_output_dir.return_value = self._tmp_dir
        ophys_experiment = OphysExperiment(
            id=self._experiment_id,
            movie_frame_rate_hz=31.0,
            raw_movie_filename=self.motion_corrected_movie_path,
            session=OphysSession(id=1, specimen=Specimen(id='1')),
            container=OphysContainer(id=1, specimen=Specimen(id='1')),
            specimen=Specimen(id='1'),
            storage_directory=str(self._tmp_dir),
            equipment_name='DEEPSCOPE',
            full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
        )
        
        with patch(
                'ophys_etl.workflows.ophys_experiment.engine',
                new=self._engine):
            mod = TraceExtractionModule(
                docker_tag='main',
                ophys_experiment=ophys_experiment,
                motion_corrected_ophys_movie_file=OutputFile(
                    well_known_file_type=(
                        WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                    ),
                    path=self.motion_corrected_movie_path,
                )
            )

        mod.inputs


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
from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue
from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.pipeline_modules.trace_processing.event_detection import EventDetectionModule  # noqa E501


class TestEventDetectionModule(MockSQLiteDB):

    def setup_class(cls):
        cls._experiment_id = 1

    def setup(self):
        super().setup()

        _, dff_trace_path = tempfile.mkstemp(suffix='.h5')
        self.dff_trace_path = Path(dff_trace_path)
        self.temp_dir = self.dff_trace_path.parent

        with h5py.File(self.dff_trace_path, 'w') as h5:
            h5.create_dataset('roi_names', np.arange(10))

        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.DFF,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.DFF_TRACES
                        ),
                        path=self.dff_trace_path,
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
        self.dff_trace_path.unlink()

    @patch.object(OphysExperiment, 'rois',
                  new_callable=PropertyMock)
    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(EventDetectionModule, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir,
                    mock_rois):
        """Test that inputs are correctly formated for input into the module.
        """
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
        mock_output_path.return_value = self.temp_dir
        mock_output_dir.return_value = self.temp_dir
        ophys_experiment = OphysExperiment(
            id=self._experiment_id,
            movie_frame_rate_hz=31.0,
            raw_movie_filename=Path('foo'),
            session=OphysSession(id=1, specimen=Specimen(id='1')),
            container=OphysContainer(id=1, specimen=Specimen(id='1')),
            specimen=Specimen(id='1'),
            storage_directory=Path('foo'),
            equipment_name='DEEPSCOPE',
            full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
        )

        mod = EventDetectionModule(
            docker_tag='main',
            ophys_experiment=ophys_experiment,
            dff_traces=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DFF_TRACES
                ),
                path=self.dff_trace_path,
            )
        )

        mod.inputs

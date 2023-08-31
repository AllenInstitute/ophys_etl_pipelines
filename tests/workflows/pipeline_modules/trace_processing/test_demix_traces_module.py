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
from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.pipeline_modules.trace_processing.demix_traces import DemixTracesModule  # noqa E501


class TestDemixTracesModule(MockSQLiteDB):

    def setup_class(cls):
        cls._experiment_id = 1

    def setup(self):
        super().setup()

        _, motion_corrected_ophys_movie_path = tempfile.mkstemp(suffix='.h5')
        self.motion_corrected_ophys_movie_path = Path(motion_corrected_ophys_movie_path)

        with h5py.File(self.motion_corrected_ophys_movie_path, 'w') as f:
            f.create_dataset("dummy_data", data=[1.0, 2.0, 3.0])

        _, roi_trace_path = tempfile.mkstemp(suffix='.h5')
        self.roi_trace_path = Path(roi_trace_path)
        self.temp_dir = self.roi_trace_path.parent

        with h5py.File(self.roi_trace_path, 'w') as h5:
            h5.create_dataset('roi_names', np.arange(10))


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
                        path=self.roi_trace_path,
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
        self.roi_trace_path.unlink()
        self.motion_corrected_ophys_movie_path.unlink()

    @patch.object(OphysExperiment, 'rois',
                  new_callable=PropertyMock)
    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(DemixTracesModule, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir,
                    mock_rois):
        """Test that inputs are correctly formatted for input into the module.
        """

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

        mod = DemixTracesModule(
            docker_tag='main',
            ophys_experiment=ophys_experiment,
            motion_corrected_ophys_movie_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
                path=self.motion_corrected_ophys_movie_path,
            ),
            roi_traces_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_TRACE
                ),
                path=self.roi_trace_path,
            )
        )

        mod.inputs


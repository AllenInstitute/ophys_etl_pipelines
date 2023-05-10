import datetime
from pathlib import Path
from unittest.mock import patch, PropertyMock

from ophys_etl.workflows.output_file import OutputFile

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from sqlmodel import Session

from ophys_etl.workflows.ophys_experiment import OphysSession, Specimen, \
    OphysExperiment, ImagingPlaneGroup

from ophys_etl.workflows.pipeline_modules.decrosstalk import DecrosstalkModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestDecrosstalk(MockSQLiteDB):
    @classmethod
    def setup_class(cls):
        cls._experiment_ids = ['oe_1', 'oe_2']

    def setup(self):
        super().setup()

        with Session(self._engine) as session:
            for oe_id in self._experiment_ids:
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
                            path=Path(f'{oe_id}_motion_correction.h5'),
                        ),
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.
                                MAX_INTENSITY_PROJECTION_IMAGE
                            ),
                            path=Path(f'{oe_id}_max_proj.png'),
                        )
                    ],
                    ophys_experiment_id=oe_id,
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    validate_files_exist=False
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

    @patch.object(OphysExperiment, 'from_id')
    @patch.object(OphysSession, 'ophys_experiment_ids',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_ophys_session_oe_ids,
                    mock_ophys_experiment_from_id
                    ):
        ophys_session = OphysSession(
                id='session_1',
                specimen=Specimen(id='specimen_1')
            )

        mock_ophys_session_oe_ids.return_value = self._experiment_ids
        mock_ophys_experiment_from_id.side_effect = \
            lambda id: OphysExperiment(
                id=id,
                movie_frame_rate_hz=1,
                raw_movie_filename=Path('foo'),
                session=ophys_session,
                specimen=ophys_session.specimen,
                storage_directory=Path('foo'),
                imaging_plane_group=ImagingPlaneGroup(
                    id=0 if id == 'oe_1' else 1,
                    group_order=0 if id == 'oe_1' else 1
                )
            )

        mod = DecrosstalkModule(
            docker_tag='main',
            ophys_session=ophys_session
        )

        with patch('ophys_etl.workflows.pipeline_modules.decrosstalk.engine',
                   new=self._engine):
            obtained_inputs = mod.inputs

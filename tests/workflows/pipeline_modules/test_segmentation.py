import datetime
from pathlib import Path
from unittest.mock import patch

from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule

from ophys_etl.workflows.ophys_experiment import OphysExperiment, Specimen, \
    OphysSession
from sqlmodel import Session, select

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.segmentation import (
    SegmentationModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestSegmentation(MockSQLiteDB):
    def setup(self):
        super().setup()

        xy_offset_path = (
            Path(__file__).parent / "resources" / "rigid_motion_transform.csv"
        )
        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                        ),
                        path=xy_offset_path,
                    )
                ],
                ophys_experiment_id="1",
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                additional_steps=MotionCorrectionModule.save_metadata_to_db,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            )

    @patch.object(OphysExperiment, 'from_id')
    def test_save_metadata_to_db(self, mock_ophys_experiment_from_id):
        mock_ophys_experiment_from_id.return_value = \
            OphysExperiment(
                id='1',
                movie_frame_rate_hz=1,
                raw_movie_filename=Path('foo'),
                specimen=Specimen(id='1'),
                session=OphysSession(id='1', specimen=Specimen(id='1')),
                storage_directory=Path('foo'),
                equipment_name='MESO.1'
            )

        _rois_path = Path(__file__).parent / "resources" / "rois.json"
        with patch('ophys_etl.workflows.ophys_experiment.engine',
                   new=self._engine):
            with Session(self._engine) as session:
                save_job_run_to_db(
                    workflow_step_name=WorkflowStepEnum.SEGMENTATION,
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.OPHYS_ROIS
                            ),
                            path=_rois_path,
                        )
                    ],
                    ophys_experiment_id="1",
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    additional_steps=SegmentationModule.save_rois_to_db,
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                )
        with Session(self._engine) as session:
            rois = session.exec(select(OphysROI)).all()
            masks = session.exec(select(OphysROIMaskValue)).all()

        assert len(rois) == 2
        assert rois[0].x == 19
        assert rois[0].y == 326
        assert rois[0].workflow_step_run_id == 2
        assert rois[0].width == 14
        assert rois[0].height == 15
        assert not rois[0].is_in_motion_border
        assert rois[1].is_in_motion_border

        assert len(masks) == 2
        mask1 = [x for x in masks if x.ophys_roi_id == 1]
        mask2 = [x for x in masks if x.ophys_roi_id == 2]
        assert {(x.row_index, x.col_index) for x in mask1} == {(0, 0)}
        assert {(x.row_index, x.col_index) for x in mask2} == {(0, 1)}

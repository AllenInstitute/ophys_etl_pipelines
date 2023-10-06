import datetime
from pathlib import Path
from unittest.mock import PropertyMock, patch

from sqlmodel import Session, select

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.db.schemas import OphysROI
from ophys_etl.workflows.ophys_experiment import OphysExperiment, OphysSession
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.motion_correction import (
    MotionCorrectionModule,
)
from ophys_etl.workflows.pipeline_modules.segmentation import SegmentationModule # noqa E501
from ophys_etl.workflows.pipeline_modules.trace_processing.trace_extraction import ( # noqa E501
    TraceExtractionModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestTraceExtractionModule(MockSQLiteDB):
    def setup(self):
        super().setup()
        xy_offset_path = (
            Path(__file__).parent.parent / "resources" / "rigid_motion_transform.csv" # noqa E501
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

    @patch.object(OphysExperiment, "motion_border", new_callable=PropertyMock)
    @patch.object(OphysExperiment, "rois", new_callable=PropertyMock)
    @patch.object(OphysSession, "output_dir", new_callable=PropertyMock)
    @patch.object(TraceExtractionModule, "output_path", new_callable=PropertyMock) # noqa E501
    def test_inputs(
        self,
        mock_output_path,
        mock_output_dir,
        mock_oe_rois,
        mock_motion_border,
        temp_dir,
        mock_ophys_experiment,
        mock_motion_border_run,
        motion_corrected_ophys_movie_path,
        mock_rois,
    ):
        """Test that inputs are correctly formatted
        for input into the module."""
        mock_motion_border.return_value = mock_motion_border_run
        mock_oe_rois.return_value = mock_rois
        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir

        mod = TraceExtractionModule(
            docker_tag="main",
            ophys_experiment=mock_ophys_experiment,
            motion_corrected_ophys_movie_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
                path=motion_corrected_ophys_movie_path,
            ),
        )

        mod.inputs

    @patch.object(OphysExperiment, "from_id")
    @patch.object(TraceExtractionModule, "outputs")
    def test_save_metadata_to_db(
        self,
        mock_output,
        mock_ophys_experiment_from_id,
        mock_ophys_experiment,
        trace_path,
    ):
        mock_ophys_experiment_from_id.return_value = mock_ophys_experiment
        output_files = [
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.TRACE_EXTRACTION_EXCLUSION_LABELS, # noqa E501
                path=Path(__file__).parent.parent
                / "resources"
                / "trace_extraction_output.json",
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.ROI_TRACE, path=trace_path # noqa E501
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_TRACE,
                path=trace_path,
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_MASK,
                path=trace_path,
            ),
        ]
        mock_output.return_value = output_files

        _rois_path = Path(__file__).parent.parent / "resources" / "rois.json"
        with patch("ophys_etl.workflows.ophys_experiment.engine", new=self._engine): # noqa E501
            with Session(self._engine) as session:
                save_job_run_to_db(
                    workflow_step_name=WorkflowStepEnum.SEGMENTATION,
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=(WellKnownFileTypeEnum.OPHYS_ROIS), # noqa E501
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
                save_job_run_to_db(
                    workflow_step_name=WorkflowStepEnum.TRACE_EXTRACTION,
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=output_files,
                    ophys_experiment_id="1",
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    additional_steps=TraceExtractionModule.save_exclusion_labels_to_db, # noqa E501
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                )

        with Session(self._engine) as session:
            rois = session.exec(select(OphysROI)).all()

        assert len(rois) == 2
        assert rois[0].id == 1 and rois[1].id == 2
        assert rois[0].empty_roi_mask is True
        assert rois[1].empty_roi_mask is False
        assert rois[0].empty_neuropil_mask is True
        assert rois[1].empty_neuropil_mask is False

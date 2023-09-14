from unittest.mock import PropertyMock, patch

from ophys_etl.workflows.ophys_experiment import OphysExperiment, OphysSession
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.trace_processing.trace_extraction import ( # noqa E501
    TraceExtractionModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from tests.workflows.conftest import MockSQLiteDB


class TestTraceExtractionModule:

    @patch.object(OphysExperiment, "motion_border", new_callable=PropertyMock)
    @patch.object(OphysExperiment, "rois", new_callable=PropertyMock)
    @patch.object(OphysSession, "output_dir", new_callable=PropertyMock)
    @patch.object(
        TraceExtractionModule, "output_path", new_callable=PropertyMock
    )
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

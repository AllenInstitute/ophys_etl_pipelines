from unittest.mock import PropertyMock, patch

from ophys_etl.workflows.ophys_experiment import OphysExperiment, OphysSession
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.trace_processing.demix_traces import (  # noqa E501
    DemixTracesModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from tests.workflows.conftest import MockSQLiteDB


class TestDemixTracesModule:

    @patch.object(OphysExperiment, "rois", new_callable=PropertyMock)
    @patch.object(OphysSession, "output_dir", new_callable=PropertyMock)
    @patch.object(DemixTracesModule, "output_path", new_callable=PropertyMock)
    def test_inputs(
        self,
        mock_output_path,
        mock_output_dir,
        mock_rois,
        temp_dir,
        mock_ophys_experiment,
        motion_corrected_ophys_movie_path,
        trace_path,
    ):
        """Test that inputs are correctly formatted
        for input into the module."""

        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir
        ophys_experiment = mock_ophys_experiment

        mod = DemixTracesModule(
            docker_tag="main",
            ophys_experiment=ophys_experiment,
            motion_corrected_ophys_movie_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
                path=motion_corrected_ophys_movie_path,
            ),
            roi_traces_file=OutputFile(
                well_known_file_type=(WellKnownFileTypeEnum.ROI_TRACE),
                path=trace_path,
            ),
        )

        mod.inputs

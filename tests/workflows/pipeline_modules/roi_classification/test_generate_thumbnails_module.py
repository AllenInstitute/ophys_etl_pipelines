from unittest.mock import PropertyMock, patch

import pytest

from ophys_etl.workflows.ophys_experiment import OphysExperiment, OphysSession
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.roi_classification.generate_thumbnails import ( # noqa E501
    GenerateThumbnailsModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from tests.workflows.conftest import BaseTestPipelineModule


class TestGenerateThumbnailsModule(BaseTestPipelineModule):
    def setup(self):
        super().setup()

    @pytest.mark.parametrize("is_training", [True, False])
    @patch.object(OphysExperiment, "rois", new_callable=PropertyMock)
    @patch.object(OphysSession, "output_dir", new_callable=PropertyMock)
    @patch.object(
        GenerateThumbnailsModule, "output_path", new_callable=PropertyMock
    )
    def test_inputs(
        self,
        mock_output_path,
        mock_output_dir,
        mock_oe_rois,
        is_training,
        mock_rois,
        temp_dir,
        mock_ophys_experiment,
        motion_corrected_ophys_movie_path,
        trace_path,
    ):
        """Test that inputs are correctly formatted for
        input into the module."""

        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir
        mock_oe_rois.return_value = mock_rois

        mod = GenerateThumbnailsModule(
            docker_tag="main",
            ophys_experiment=mock_ophys_experiment,
            denoised_ophys_movie_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DEEPINTERPOLATION_DENOISED_MOVIE
                ),
                path=motion_corrected_ophys_movie_path,
            ),
            rois=[x.to_dict() for x in mock_rois],
            correlation_projection_graph_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH # noqa E501
                ),
                path=trace_path,
            ),
            is_training=is_training,
            motion_correction_shifts_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                ),
                path=trace_path,
            ),
        )
        mod.inputs

from unittest.mock import patch, PropertyMock

from tests.workflows.conftest import *

from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,
    OphysSession)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.pipeline_modules.denoising.denoising_inference import DenoisingInferenceModule  # noqa E501


class TestDenoisingFinetuningModule(BaseTestPipelineModule):

    def setup(self):
        super().setup_method()

    @patch.object(OphysExperiment, 'rois',
                  new_callable=PropertyMock)
    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(DenoisingInferenceModule, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir,
                    mock_rois,
                    temp_dir, mock_ophys_experiment,
                    motion_corrected_ophys_movie_path,
                    trace_path):
        """Test that inputs are correctly formatted for input into the module.
        """

        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir


        mod = DenoisingInferenceModule(
            docker_tag='main',
            ophys_experiment=mock_ophys_experiment,
            motion_corrected_ophys_movie_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
                path=motion_corrected_ophys_movie_path,
            ),
            roi_traces_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_TRACE
                ),
                path=trace_path,
            )
        )

        mod.inputs


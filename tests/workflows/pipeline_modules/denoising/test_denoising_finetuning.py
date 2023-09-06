from unittest.mock import patch, PropertyMock

from tests.workflows.conftest import *

from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,
    OphysSession)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.pipeline_modules.denoising.denoising_finetuning import DenoisingFinetuningModule  # noqa E501


class TestDenoisingFinetuningModule(BaseTestPipelineModule):

    def setup(self):
        super().setup()

    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(DenoisingFinetuningModule, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir,
                    temp_dir, mock_ophys_experiment,
                    motion_corrected_ophys_movie_path):
        """Test that inputs are correctly formatted for input into the module.
        """

        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir


        mod = DenoisingFinetuningModule(
            docker_tag='main',
            ophys_experiment=mock_ophys_experiment,
            motion_corrected_ophys_movie_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
                path=motion_corrected_ophys_movie_path,
            )
        )

        mod.inputs


import pytest
from unittest.mock import PropertyMock, patch

from ophys_etl.workflows.ophys_experiment import OphysSession
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
try:
    from ophys_etl.workflows.pipeline_modules.denoising.denoising_inference import ( # noqa E501
        DenoisingInferenceModule,
    )

    class TestDenoisingInferencegModule:

        @pytest.mark.deepinterpolation_only
        @patch.object(OphysSession, "output_dir", new_callable=PropertyMock)
        @patch.object(
            DenoisingInferenceModule, "output_path", new_callable=PropertyMock
        )
        def test_inputs(
            self,
            mock_output_path,
            mock_output_dir,
            temp_dir,
            mock_ophys_experiment,
            motion_corrected_ophys_movie_path,
            trace_path,
        ):
            """Test that inputs are correctly formatted for input
            into the module. The inputs are validated during object
            instantiation and will fail if the format is not
            correctly formatted.
            """

            mock_output_path.return_value = temp_dir
            mock_output_dir.return_value = temp_dir

            mod = DenoisingInferenceModule(
                docker_tag="main",
                ophys_experiment=mock_ophys_experiment,
                motion_corrected_ophys_movie_file=OutputFile(
                    well_known_file_type=(
                        WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                    ),
                    path=motion_corrected_ophys_movie_path,
                ),
                trained_denoising_model_file=OutputFile(
                    well_known_file_type=(
                        WellKnownFileTypeEnum.DEEPINTERPOLATION_FINETUNED_MODEL
                    ),
                    path=trace_path,
                ),
            )
            mod.inputs

except ModuleNotFoundError:
    # even though we might skip tests, pytest tries these imports
    pass

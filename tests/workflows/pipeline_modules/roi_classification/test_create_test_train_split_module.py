# TODO: This test needs to be completed when the CreateTestTrainSplitModule is completed. See ticket PSB-192 for more details.
# from unittest.mock import patch, PropertyMock

# from tests.workflows.conftest import *

# from ophys_etl.workflows.ophys_experiment import (
#     OphysExperiment,
#     OphysSession)
# from ophys_etl.workflows.output_file import OutputFile
# from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
# from ophys_etl.workflows.pipeline_modules.roi_classification.create_train_test_split import CreateTrainTestSplitModule  # noqa E501


# class TestCreateTrainTestSplitModule(BaseTestPipelineModule):

#     def setup(self):
#         super().setup()

#     @patch.object(OphysSession, 'output_dir',
#                   new_callable=PropertyMock)
#     @patch.object(CreateTrainTestSplitModule, 'output_path',
#                   new_callable=PropertyMock)
#     def test_inputs(self,
#                     mock_output_path,
#                     mock_output_dir,
#                     mock_thumbnails_dir,
#                     temp_dir):
#         """Test that inputs are correctly formatted for input into the module.
#         """

#         mock_output_path.return_value = temp_dir
#         mock_output_dir.return_value = temp_dir


#         mod = CreateTrainTestSplitModule(
#             docker_tag='main',
#             thumbnail_dirs=mock_thumbnails_dir
#         )

#         mod.inputs


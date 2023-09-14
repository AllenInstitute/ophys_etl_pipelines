from unittest.mock import PropertyMock, patch

import pytest

from ophys_etl.workflows.ophys_experiment import OphysSession
from ophys_etl.workflows.pipeline_modules.roi_classification.create_train_test_split import ( # noqa E501
    CreateTrainTestSplitModule,
)


class TestCreateTrainTestSplitModule:

    @pytest.mark.skip(
        reason="this test needs to be completed when the"
        "CreateTestTrainSplitModule is completed. See ticket"
        "PSB-192 for more details."
    )
    @patch.object(OphysSession, "output_dir", new_callable=PropertyMock)
    @patch.object(
        CreateTrainTestSplitModule, "output_path", new_callable=PropertyMock
    )
    def test_inputs(
        self, mock_output_path, mock_output_dir, mock_thumbnails_dir, temp_dir
    ):
        """Test that inputs are correctly formatted
        for input into the module.
        """

        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir

        mod = CreateTrainTestSplitModule(
            docker_tag="main", thumbnail_dirs=mock_thumbnails_dir
        )

        mod.inputs

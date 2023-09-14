import pytest
from unittest.mock import patch, PropertyMock

from tests.workflows.conftest import MockSQLiteDB

from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,
    OphysSession)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.pipeline_modules.trace_processing.event_detection import EventDetectionModule  # noqa E501


class TestEventDetectionModule:

    @pytest.mark.event_detect_only
    @patch.object(OphysExperiment, 'rois',
                  new_callable=PropertyMock)
    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(EventDetectionModule, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir,
                    mock_oe_rois,
                    mock_rois,
                    temp_dir,
                    mock_ophys_experiment,
                    trace_path):
        """Test that inputs are correctly formated for input into the module.
        """

        mock_oe_rois.return_value = mock_rois
        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir

        mod = EventDetectionModule(
            docker_tag='main',
            ophys_experiment=mock_ophys_experiment,
            dff_traces=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DFF_TRACES
                ),
                path=trace_path,
            )
        )

        mod.inputs

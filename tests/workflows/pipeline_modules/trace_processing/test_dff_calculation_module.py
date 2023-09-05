from unittest.mock import patch, PropertyMock

from tests.workflows.conftest import MockSQLiteDB

from ophys_etl.workflows.ophys_experiment import OphysSession

from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.pipeline_modules.trace_processing.dff_calculation import DffCalculationModule  # noqa E501


class TestDffCalculationModule(MockSQLiteDB):

    def setup(self):
        super().setup()

    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(DffCalculationModule, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir,
                    temp_dir,
                    mock_ophys_experiment,
                    trace_path
                    ):
        """Test that inputs are correctly formatted for input into the module.
        """
        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir

        mod = DffCalculationModule(
            docker_tag='main',
            ophys_experiment=mock_ophys_experiment,
            neuropil_corrected_traces=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.NEUROPIL_CORRECTED_TRACES
                ),
                path=trace_path,
            )
        )

        mod.inputs

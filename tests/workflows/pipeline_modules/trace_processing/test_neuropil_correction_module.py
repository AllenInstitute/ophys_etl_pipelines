from pathlib import Path
from unittest.mock import patch, PropertyMock
import tempfile
import h5py
import numpy as np

from tests.workflows.conftest import MockSQLiteDB

from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,
    OphysSession,
    Specimen, OphysContainer)
from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.pipeline_modules.trace_processing.neuropil_correction import NeuropilCorrection  # noqa E501


class TestNeuropilCorrection(MockSQLiteDB):

    def setup_class(cls):
        cls._experiment_id = 1

    def setup(self):
        super().setup()

        _, demixed_roi_traces_path = tempfile.mkstemp(suffix='.h5')
        _, neuropil_traces_path = tempfile.mkstemp(suffix='.h5')
        self.demixed_roi_traces_path = Path(demixed_roi_traces_path)
        self.neuropil_traces_path = Path(neuropil_traces_path)
        self.temp_dir = self.demixed_roi_traces_path.parent

        with h5py.File(self.demixed_roi_traces_path, 'w') as h5:
            h5.create_dataset('roi_names', np.arange(10))

        with h5py.File(self.neuropil_traces_path, 'w') as h5:
            h5.create_dataset('neuropil_names', np.arange(10))

    def teardown(self):
        self.demixed_roi_traces_path.unlink()
        self.neuropil_traces_path.unlink()

    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(NeuropilCorrection, 'output_path',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_output_path,
                    mock_output_dir):
        """Test that inputs are correctly formatted for input into the module.
        """
        mock_output_path.return_value = self.temp_dir
        mock_output_dir.return_value = self.temp_dir
        ophys_experiment = OphysExperiment(
            id=self._experiment_id,
            movie_frame_rate_hz=31.0,
            raw_movie_filename=Path('foo'),
            session=OphysSession(id=1, specimen=Specimen(id='1')),
            container=OphysContainer(id=1, specimen=Specimen(id='1')),
            specimen=Specimen(id='1'),
            storage_directory=Path('foo'),
            equipment_name='DEEPSCOPE',
            full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
        )

        mod = NeuropilCorrection(
            docker_tag='main',
            ophys_experiment=ophys_experiment,
            demixed_roi_traces_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DEMIXED_TRACES
                ),
                path=self.demixed_roi_traces_path,
            ),
            neuropil_traces_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.NEUROPIL_TRACE
                ),
                path=self.neuropil_traces_path,
            )
        )

        assert mod.inputs == {
            "roi_trace_file": str(self.demixed_roi_traces_path),
            "storage_directory": str(self.temp_dir),
            "neuropil_trace_file": str(self.neuropil_traces_path)
        }

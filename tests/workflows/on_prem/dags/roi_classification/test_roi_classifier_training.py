from pathlib import Path
from unittest.mock import patch

from ophys_etl.workflows.on_prem.dags.cell_classification.\
    roi_classifier_training import _get_roi_meta_for_experiment_ids
from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysContainer, Specimen, OphysSession
from tests.workflows.conftest import MockSQLiteDB
from tests.workflows.test_ophys_experiment import TestOphysExperiment


class TestROIClassifierTraining(MockSQLiteDB):
    def setup(self):
        super().setup()
        TestOphysExperiment.create_mock_data(engine=self._engine)

    @patch.object(OphysExperiment, 'from_id')
    def test__get_roi_meta_for_experiment_ids(self, mock_oe):
        mock_oe.return_value = OphysExperiment(
            id=1,
            container=OphysContainer(id=1, specimen=Specimen(id='1')),
            session=OphysSession(id=1, specimen=Specimen(id='1')),
            equipment_name='',
            full_genotype='',
            movie_frame_rate_hz=1,
            raw_movie_filename=Path(''),
            specimen=Specimen(id='1'),
            storage_directory=Path('')
        )

        with patch(
                "ophys_etl.workflows.ophys_experiment.engine", self._engine
        ):
            result = _get_roi_meta_for_experiment_ids(
                experiment_ids=[1]
            )
        assert result == {
            '1': {
                '1': {
                    'is_inside_motion_border': True
                },
                '2': {
                    'is_inside_motion_border': True
                }
            }
        }

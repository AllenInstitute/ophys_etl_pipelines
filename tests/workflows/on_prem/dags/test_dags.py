from pathlib import Path
from airflow.models import DagBag

from ophys_etl.workflows import on_prem


class TestDags:
    @classmethod
    def setup_class(cls):
        cls._dag_bag = DagBag(
            dag_folder=str(Path(on_prem.__file__).parent / 'dags'),
            include_examples=False
        )

    def test_dags_load(self):
        """Test that all dags load successfully"""
        for dag_path, traceback in self._dag_bag.import_errors.items():
            raise Exception(f'{dag_path} failed to load\n {traceback}')

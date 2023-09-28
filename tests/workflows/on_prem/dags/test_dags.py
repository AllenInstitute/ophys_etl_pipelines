import os
from pathlib import Path

from ophys_etl.workflows import on_prem
os.environ['AIRFLOW_HOME'] = str(Path(on_prem.__file__).parent)

from airflow.models import DagBag


class TestDags:
    @classmethod
    def setup_class(cls):
        cls._dag_bag = DagBag(
            dag_folder=str(Path(os.environ['AIRFLOW_HOME']) / 'dags'),
            include_examples=False
        )

    def test_dags_load(self):
        """Test that all dags load successfully"""
        for dag_path, traceback in self._dag_bag.import_errors.items():
            raise Exception(f'{dag_path} failed to load\n {traceback}')

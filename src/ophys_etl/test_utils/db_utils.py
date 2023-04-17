import os
import shutil
from pathlib import Path

import tempfile

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent / 'resources' / 'config.yml'),
    test_di_base_model_path=Path(__file__).parent.parent / 'resources' /
    'di_model.h5'
)

from ophys_etl.workflows.db.db_utils import save_job_run_to_db # noqa E402
from ophys_etl.workflows.db.initialize_db import InitializeDBRunner # noqa E402


class TestDBUtils:
    @classmethod
    def _initialize_db(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / 'app.db'
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f'sqlite:///{cls._db_path}'
        cls._engine = InitializeDBRunner(
            input_data={
                'db_url': db_url
            },
            args=[]).run()

    def setup(self):
        self._initialize_db()

    def teardown_method(self):
        shutil.rmtree(self._tmp_dir)

class MockSQLiteDB:
    @classmethod
    def setup_class(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / 'app.db'
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f'sqlite:///{cls._db_path}'
        cls._engine = InitializeDBRunner(
            input_data={
                'db_url': db_url
            },
            args=[]).run()
        cls._xy_offset_path = \
            Path(__file__).parent / 'resources' / 'rigid_motion_transform.csv'
        cls._rois_path = \
            Path(__file__).parent / 'resources' / 'rois.json'

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmp_dir)


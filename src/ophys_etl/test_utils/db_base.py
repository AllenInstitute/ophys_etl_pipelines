import os
import shutil
from pathlib import Path

import tempfile

from ophys_etl.test_utils.workflow_utils import setup_app_config

pkg_root = Path(__file__).parent.parent.parent.parent
setup_app_config(
    ophys_workflow_app_config_path=(
            pkg_root / 'tests' / 'workflows' / 'resources' / 'config.yml'),
    test_di_base_model_path = pkg_root / 'tests' /  'workflows' / 'resources' /
    'di_model.h5'
)

from ophys_etl.workflows.db.initialize_db import InitializeDBRunner # noqa E402


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

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmp_dir)


import os
import shutil
import tempfile
from pathlib import Path


from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
        Path(__file__).parent / "resources" / "config.yml"
    ),
    test_di_base_model_path=(
        Path(__file__).parent / "resources" / "di_model.h5"
    ),
)

from ophys_etl.workflows.db.initialize_db import (
    InitializeDBRunner,
)  # noqa E402


class MockSQLiteDB:
    @classmethod
    def _initializeDB(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / "app.db"
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f"sqlite:///{cls._db_path}"
        cls._engine = InitializeDBRunner(
            input_data={"db_url": db_url}, args=[]
        ).run()

    def setup(self):
        self._initializeDB()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmp_dir)
import os
import tempfile


class AirflowTest:
    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()

        # IMPORTANT -- make sure all airflow imports happen after setup_class
        # gets called
        os.environ['AIRFLOW_HOME'] = cls._tmpdir.name

    @classmethod
    def teardown_class(cls):
        cls._tmpdir.cleanup()

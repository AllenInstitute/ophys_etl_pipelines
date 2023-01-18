import datetime
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from ophys_etl.workflows.db.schemas import WorkflowStepRun, WellKnownFile, \
    WellKnownFileType

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent / 'resources' / 'config.yml'),
    test_di_base_model_path=(Path(__file__).parent.parent / 'resources' /
                             'di_model.h5')
)

from ophys_etl.workflows.pipeline_module import OutputFile  # noqa #402
from ophys_etl.workflows.db.db_utils import _get_workflow_step_by_name, \
    _get_well_known_file_type, save_job_run_to_db, \
    ModuleOutputFileDoesNotExistException   # noqa #402
from sqlalchemy import create_engine    # noqa #402
from sqlmodel import Session, select    # noqa #402

from ophys_etl.workflows.db.initialize_db import IntializeDBRunner  # noqa #402


class TestDBUtils:
    @classmethod
    def setup_class(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / 'app.db'
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f'sqlite:///{cls._db_path}'
        IntializeDBRunner(
            input_data={
                'db_url': db_url
            },
            args=[]).run()
        cls._engine = create_engine(db_url)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls._tmp_dir)

    def teardown_method(self):
        for file in os.listdir(self._tmp_dir):
            if Path(file).suffix != '.db':
                os.remove(self._tmp_dir / file)

    def test__get_workflow_step_by_name(self):
        with Session(self._engine) as session:
            step = _get_workflow_step_by_name(
                session=session,
                name='motion_correction'
            )
        assert step.name == 'motion_correction'

    def test__get_well_known_file_type(self):
        with Session(self._engine) as session:
            wkft = _get_well_known_file_type(
                session=session,
                name='MotionCorrectedImageStack'
            )
        assert wkft.name == 'MotionCorrectedImageStack'

    @pytest.mark.parametrize('out_file_exists', (True, False))
    def test__save_job_run_to_db(self, out_file_exists):
        start = datetime.datetime.now()
        end = datetime.datetime.now()

        output_files = [
            OutputFile(
                path=self._tmp_dir / 'out1',
                well_known_file_type='MotionCorrectedImageStack'
            ),
            OutputFile(
                path=self._tmp_dir / 'out2',
                well_known_file_type='OphysMotionXyOffsetData'
            )
        ]

        if out_file_exists:
            for out in output_files:
                with open(out.path, 'w') as f:
                    f.write('foo')

        def save_job_run():
            save_job_run_to_db(
                workflow_step_name='motion_correction',
                start=start,
                end=end,
                module_outputs=output_files,
                ophys_experiment_id='1',
                sqlalchemy_session=session,
                storage_directory='/foo'
            )

        with Session(self._engine) as session:
            if not out_file_exists:
                with pytest.raises(ModuleOutputFileDoesNotExistException):
                    save_job_run()
                return

            save_job_run()
            stmt = select(WorkflowStepRun)
            runs = session.exec(stmt).all()

            # check 1 run inserted
            assert len(runs) == 1

            run_id = runs[0].id

            # check output files inserted
            well_known_files = session.exec(
                select(WellKnownFile)
                .where(WellKnownFile.workflow_step_run_id == run_id)
            ).all()
            assert len(well_known_files) == len(output_files)

            # check output files of correct type
            file_types = session.exec(
                select(WellKnownFileType)
                .where(WellKnownFileType.id.in_(
                    [x.well_known_file_type_id for x in
                     well_known_files]
                    )
                )
            ).all()
            assert set([x.name for x in file_types]) == \
                {x.well_known_file_type for x in output_files}

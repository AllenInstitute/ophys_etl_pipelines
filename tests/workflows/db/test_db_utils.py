import datetime
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
        Path(__file__).parent.parent / "resources" / "config.yml"
    ),
    test_di_base_model_path=(
        Path(__file__).parent.parent / "resources" / "di_model.h5"
    ),
)

from sqlmodel import Session, select  # noqa #402

from ophys_etl.workflows.db.db_utils import (  # noqa #402
    ModuleOutputFileDoesNotExistException,
    _validate_files_exist,
    get_well_known_file_type,
    get_workflow_step_by_name,
    save_job_run_to_db,
)
from ophys_etl.workflows.db.initialize_db import (
    InitializeDBRunner,
)  # noqa #402
from ophys_etl.workflows.db.schemas import (
    WellKnownFile,
    WellKnownFileType,
    WorkflowStepRun,
)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class TestDBUtils:
    @classmethod
    def _initialize_db(cls):
        cls._tmp_dir = Path(tempfile.TemporaryDirectory().name)
        cls._db_path = cls._tmp_dir / "app.db"
        os.makedirs(cls._db_path.parent, exist_ok=True)

        db_url = f"sqlite:///{cls._db_path}"
        cls._engine = InitializeDBRunner(
            input_data={"db_url": db_url}, args=[]
        ).run()

    def setup(self):
        self._initialize_db()

    def teardown_method(self):
        shutil.rmtree(self._tmp_dir)

    def test__get_workflow_step_by_name(self):
        with Session(self._engine) as session:
            step = get_workflow_step_by_name(
                session=session,
                name=WorkflowStepEnum.MOTION_CORRECTION,
                workflow=WorkflowNameEnum.OPHYS_PROCESSING,
            )
        assert step.name == WorkflowStepEnum.MOTION_CORRECTION

    def test__get_well_known_file_type(self):
        with Session(self._engine) as session:
            wkft = get_well_known_file_type(
                session=session,
                name=WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK,
                workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                workflow=WorkflowNameEnum.OPHYS_PROCESSING,
            )
        assert wkft.name == WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK

    @pytest.mark.parametrize("file_type", ("file", "dir"))
    def test__validate_files_exists(self, file_type):
        """test that _validate_files_exists works for both file and dir
        and that error is raised when dir is empty"""
        if file_type == "file":
            path = self._tmp_dir / "foo.txt"
        else:
            path = self._tmp_dir / "foo"
            os.makedirs(path)
        files = [
            OutputFile(
                path=path,
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
            )
        ]
        with pytest.raises(ModuleOutputFileDoesNotExistException):
            _validate_files_exist(output_files=files)

        if file_type == "file":
            with open(path, "w") as f:
                f.write("")
        else:
            with open(path / "foo.txt", "w") as f:
                f.write("")

        _validate_files_exist(output_files=files)

    @pytest.mark.parametrize("out_file_exists", (True, False))
    def test__save_job_run_to_db(self, out_file_exists):
        start = datetime.datetime.now()
        end = datetime.datetime.now()

        output_files = [
            OutputFile(
                path=self._tmp_dir / "out1",
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
            ),
            OutputFile(
                path=self._tmp_dir / "out2",
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                ),
            ),
        ]

        if out_file_exists:
            for out in output_files:
                with open(out.path, "w") as f:
                    f.write("foo")

        def save_job_run():
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                start=start,
                end=end,
                module_outputs=output_files,
                ophys_experiment_id="1",
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
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
                select(WellKnownFile).where(
                    WellKnownFile.workflow_step_run_id == run_id
                )
            ).all()
            assert len(well_known_files) == len(output_files)

            # check output files of correct type
            file_types = session.exec(
                select(WellKnownFileType).where(
                    WellKnownFileType.id.in_(
                        [x.well_known_file_type_id for x in well_known_files]
                    )
                )
            ).all()
            assert set([x.name for x in file_types]) == {
                x.well_known_file_type for x in output_files
            }

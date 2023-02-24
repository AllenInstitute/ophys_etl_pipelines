"""Database interface"""
import datetime
import json
import logging
from pathlib import Path
from typing import List, Union, Optional, Callable

from sqlalchemy import event, create_engine
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session, select

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.db.schemas import WorkflowStepRun, WellKnownFile, \
    WellKnownFileType, WorkflowStep
from ophys_etl.workflows.workflow_steps import WorkflowStep as WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileType as \
    WellKnownFileTypeEnum
from ophys_etl.workflows.pipeline_module import OutputFile

logger = logging.getLogger('airflow.task')


class ModuleOutputFileDoesNotExistException(Exception):
    pass


def fk_pragma_on_connect(dbapi_con, con_record):
    """Needed for sqlite to enforce foreign key constraint"""
    dbapi_con.execute('pragma foreign_keys=ON')


def enable_fk_if_sqlite():
    """Needed for sqlite to enforce foreign key constraint"""
    db_conn = app_config.app_db.conn_string
    db_conn = json.loads(db_conn)

    if db_conn['conn_type'] == 'sqlite':
        # enable foreign key constraint
        engine = create_engine(f'sqlite:///{db_conn["host"]}')
        event.listen(engine, 'connect', fk_pragma_on_connect)


def save_job_run_to_db(
        workflow_step_name: WorkflowStepEnum,
        start: datetime.datetime,
        end: datetime.datetime,
        ophys_experiment_id: str,
        module_outputs: List[OutputFile],
        sqlalchemy_session: Session,
        storage_directory: Union[Path, str],
        validate_files_exist: bool = True,
        additional_steps: Optional[Callable] = None
):
    """
    Inserts job run in db

    Parameters
    ----------
    workflow_step_name
        Name of the workflow step to log data for
    start
        Start datetime of workflow step run
    end
        End datetime of workflow step run
    ophys_experiment_id
        Identifier for experiment associated with this workflow step run
    module_outputs
        What files are output by this workflow step run
    validate_files_exist
        Validate whether `module_outputs` exist. Default is true
    sqlalchemy_session
        Session to use for inserting into DB
    storage_directory
        Where `ophys_experiment_id` is saving data to
    additional_steps
        A function which inserts additional data into the database
        Needs to have signature:
            - session: sqlalchemy Session,
            - output_files: dict mapping well known file type to OutputFile
            - run_id: workflow step run id, int
    """

    if validate_files_exist:
        for out in module_outputs:
            if not Path(out.path).exists():
                raise ModuleOutputFileDoesNotExistException(
                    f'Expected {out.well_known_file_type} to '
                    f'exist at {out.path} but it did not')
    logger.info(f'Logging output data to database for workflow step '
                f'{workflow_step_name}')

    # 1. get the workflow step
    workflow_step = _get_workflow_step_by_name(
        session=sqlalchemy_session,
        name=workflow_step_name
    )

    # 2. add a run for this workflow step
    workflow_step_run = WorkflowStepRun(
        workflow_step_id=workflow_step.id,
        storage_directory=str(storage_directory),
        ophys_experiment_id=ophys_experiment_id,
        start=start,
        end=end
    )
    sqlalchemy_session.add(workflow_step_run)

    # 3. add well known files for each output file
    for out in module_outputs:
        wkft = _get_well_known_file_type(
            session=sqlalchemy_session,
            name=out.well_known_file_type
        )
        wkf = WellKnownFile(
            workflow_step_run_id=workflow_step_run.id,
            well_known_file_type_id=wkft.id,
            path=str(out.path)
        )
        sqlalchemy_session.add(wkf)

    if additional_steps is not None:
        additional_steps(
            session=sqlalchemy_session,
            output_files={
                x.well_known_file_type.value: x for x in module_outputs
            },
            run_id=workflow_step_run.id
        )
    sqlalchemy_session.commit()


def _get_workflow_step_by_name(
    session,
    name: WorkflowStepEnum
) -> WorkflowStep:
    statement = (
        select(WorkflowStep)
        .where(WorkflowStep.name == name))
    results = session.exec(statement)
    try:
        workflow_step = results.one()
    except NoResultFound:
        logger.error(f'Workflow step {name} not found')
        raise
    return workflow_step


def _get_well_known_file_type(
    session,
    name: WellKnownFileTypeEnum
) -> WellKnownFileType:
    statement = (
        select(WellKnownFileType)
        .where(WellKnownFileType.name == name))
    results = session.exec(statement)
    try:
        wkft = results.one()
    except NoResultFound:
        logger.error(f'Well-known file type {name} not found')
        raise
    return wkft
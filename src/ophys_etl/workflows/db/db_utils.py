"""Database interface"""
import datetime
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from ophys_etl.workflows.app_config.app_config import app_config
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session, select

from ophys_etl.workflows.db.schemas import (
    WellKnownFile,
    WellKnownFileType,
    Workflow,
    WorkflowStep,
    WorkflowStepRun,
)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

logger = logging.getLogger("airflow.task")


class ModuleOutputFileDoesNotExistException(Exception):
    pass


def save_job_run_to_db(
    workflow_name: WorkflowNameEnum,
    workflow_step_name: WorkflowStepEnum,
    start: datetime.datetime,
    end: datetime.datetime,
    module_outputs: List[OutputFile],
    sqlalchemy_session: Session,
    storage_directory: Union[Path, str],
    log_path: Union[Path, str],
    ophys_experiment_id: Optional[str] = None,
    ophys_session_id: Optional[str] = None,
    ophys_container_id: Optional[str] = None,
    validate_files_exist: bool = True,
    additional_steps: Optional[Callable] = None,
    additional_steps_kwargs: Optional[Dict] = None,
):
    """
    Inserts job run in db

    Parameters
    ----------
    workflow_name
        Name of the workflow
    workflow_step_name
        Name of the workflow step to log data for
    start
        Start datetime of workflow step run
    end
        End datetime of workflow step run
    ophys_experiment_id
        Identifier for experiment associated with this workflow step run.
        None if this workflow step is not associated with a specific
        ophys experiment
    ophys_session_id
        Identifier for session associated with this workflow step run.
        None if this workflow step is not associated with a specific
        ophys session
    ophys_container_id
        Identifier for container associated with this workflow step run.
        None if this workflow step is not associated with a specific
        ophys container
    module_outputs
        What files are output by this workflow step run
    validate_files_exist
        Validate whether `module_outputs` exist. Default is true
    sqlalchemy_session
        Session to use for inserting into DB
    storage_directory
        Where `ophys_experiment_id` is saving data to
    log_path
        Path where logs for this run are saved
    additional_steps
        A function which inserts additional data into the database
        Needs to have signature:
            - session: sqlalchemy Session,
            - output_files: dict mapping well known file type to OutputFile
            - run_id: workflow step run id, int
            - **kwargs
    additional_steps_kwargs
        Kwargs to send to `additional_steps`
    """

    if validate_files_exist:
        _validate_files_exist(output_files=module_outputs)
    logger.info(
        f"Logging output data to database for workflow step "
        f"{workflow_step_name}"
    )

    # 1. get the workflow step
    workflow_step = get_workflow_step_by_name(
        session=sqlalchemy_session,
        workflow=workflow_name,
        name=workflow_step_name,
    )

    # 2. add a run for this workflow step
    workflow_step_run = WorkflowStepRun(
        workflow_step_id=workflow_step.id,
        storage_directory=str(storage_directory),
        log_path=str(log_path),
        ophys_experiment_id=ophys_experiment_id,
        ophys_session_id=ophys_session_id,
        ophys_container_id=ophys_container_id,
        start=start,
        end=end,
    )
    sqlalchemy_session.add(workflow_step_run)

    # 3. add well known files for each output file
    for out in module_outputs:
        wkft = get_well_known_file_type(
            session=sqlalchemy_session,
            name=out.well_known_file_type,
            workflow=workflow_name,
            workflow_step_name=workflow_step_name,
        )
        wkf = WellKnownFile(
            workflow_step_run_id=workflow_step_run.id,
            well_known_file_type_id=wkft.id,
            path=str(out.path),
        )
        sqlalchemy_session.add(wkf)

    if additional_steps is not None:
        if additional_steps_kwargs is not None:
            additional_steps_kwargs['ophys_experiment_id'] = \
                ophys_experiment_id
        else:
            additional_steps_kwargs = {
                'ophys_experiment_id': ophys_experiment_id
            }
        additional_steps(
            session=sqlalchemy_session,
            output_files={
                x.well_known_file_type.value: x for x in module_outputs
            },
            run_id=workflow_step_run.id,
            **additional_steps_kwargs
            if additional_steps_kwargs is not None
            else {},
        )
    sqlalchemy_session.commit()


def get_workflow_step_by_name(
    session, name: WorkflowStepEnum, workflow: WorkflowNameEnum
) -> WorkflowStep:
    """
    Get workflow step by name

    Parameters
    ----------
    session
        sqlalchemy session
    name
        workflow step name
    workflow
        workflow name

    Returns
    -------
    WorkflowStep
    """
    statement = (
        select(WorkflowStep)
        .join(Workflow, onclause=WorkflowStep.workflow_id == Workflow.id)
        .where(WorkflowStep.name == name, Workflow.name == workflow)
    )
    results = session.exec(statement)
    try:
        workflow_step = results.one()
    except NoResultFound:
        logger.error(f"Workflow step {name} not found for workflow {workflow}")
        raise
    return workflow_step


def get_well_known_file_type(
    session,
    name: WellKnownFileTypeEnum,
    workflow_step_name: WorkflowStepEnum,
    workflow: WorkflowNameEnum,
) -> WellKnownFileType:
    """
    Get well known file type by name

    Parameters
    ----------
    session
        sqlalchemy session
    name
        well known file type name
    workflow_step_name
        workflow step name
    workflow
        workflow name

    Returns
    -------
    WellKnownFileType
    """
    workflow_step = get_workflow_step_by_name(
        session=session, name=workflow_step_name, workflow=workflow
    )
    statement = select(WellKnownFileType).where(
        WellKnownFileType.name == name,
        WellKnownFileType.workflow_step_id == workflow_step.id,
    )
    results = session.exec(statement)
    try:
        wkft = results.one()
    except NoResultFound:
        logger.error(
            f"Well-known file type {name} not found for workflow "
            f"step {workflow_step_name}, workflow {workflow}"
        )
        raise
    return wkft


def _validate_files_exist(output_files: List[OutputFile]):
    """Checks that all output files exist.
    If an output file is a directory, makes sure that it is not empty

    Parameters
    ----------
    output_files
        List of output files to check

    Raises
    ------
    `ModuleOutputFileDoesNotExistException`
        If any of the files don't exist or dir is empty
    """
    for out in output_files:
        if not Path(out.path).exists():
            raise ModuleOutputFileDoesNotExistException(
                f"Expected {out.well_known_file_type} to "
                f"exist at {out.path} but it did not"
            )
        if Path(out.path).is_dir():
            if len(os.listdir(out.path)) == 0:
                if app_config.is_debug:
                    # Skipping in debug mode as it's possible for no files to
                    #  get written, which is expected
                    return
                raise ModuleOutputFileDoesNotExistException(
                    f"Directory {out.path} is empty"
                )

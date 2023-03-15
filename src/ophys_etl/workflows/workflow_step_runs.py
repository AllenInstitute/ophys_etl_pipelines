"""Functions relating to workflow step runs"""
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session, select, col

from ophys_etl.workflows.db.db_utils import get_workflow_step_by_name, \
    get_well_known_file_type
from ophys_etl.workflows.db.schemas import WorkflowStepRun, WorkflowStep, \
    WellKnownFile
from ophys_etl.workflows.pipeline_module import logger

from ophys_etl.workflows.well_known_file_types import WellKnownFileType
from ophys_etl.workflows.workflow_names import WorkflowName
from ophys_etl.workflows.workflow_steps import WorkflowStep as WorkflowStepEnum


def get_well_known_file_for_latest_run(
    engine: Engine,
    workflow_step: WorkflowStepEnum,
    workflow_name: WorkflowName,
    well_known_file_type: WellKnownFileType,
    ophys_experiment_id: Optional[str] = None
):
    """
    Gets the latest well_known_file_type path for `ophys_experiment_id`

    Parameters
    ----------
    engine
        sqlalchemy Engine
    workflow_step
        `WorkflowStep` enum
    workflow_name
        `WorkflowName` enum
    ophys_experiment_id
        Optional ophys experiment id
    well_known_file_type
        Well known file type to retrieve path for

    Returns
    -------
    `Path`
        path of well_known_file_type
    """
    with Session(engine) as session:
        workflow_step_run_id = _get_latest_run(
            session=session,
            workflow_step=workflow_step,
            workflow_name=workflow_name,
            ophys_experiment_id=ophys_experiment_id
        )
        well_known_file_type = get_well_known_file_type(
            session=session,
            name=well_known_file_type,
            workflow=workflow_name,
            workflow_step_name=workflow_step
        )
        statement = (
            select(WellKnownFile.path)
            .where(
                WellKnownFile.workflow_step_run_id == workflow_step_run_id,
                WellKnownFile.well_known_file_type_id ==
                well_known_file_type.id)
        )
        res = session.exec(statement)
        try:
            well_known_file_path = res.one()
        except NoResultFound:
            logger.error(
                f'Could not find latest {well_known_file_type} for '
                f'ophys experiment {ophys_experiment_id}: '
                f'{workflow_step}, {workflow_name}')
            raise
    return Path(well_known_file_path)


def _get_latest_run(
    session: Session,
    workflow_step: WorkflowStepEnum,
    workflow_name: WorkflowName,
    ophys_experiment_id: Optional[str] = None
) -> int:
    """
    Gets the latest workflow step run id for `workflow_step` as part of
    `workflow_name` for `ophys_experiment_id`


    Parameters
    ----------
    session
        sqlalchemy session
    workflow_step
        `WorkflowStep` enum
    workflow_name
        `WorkflowName` enum
    ophys_experiment_id
        Optional ophys experiment id

    Returns
    -------
    int
        workflow step run id

    Raises
    -------
    NoResultFound
        If workflow step run cannot be found
    """
    workflow_step = get_workflow_step_by_name(
        session=session,
        name=workflow_step,
        workflow=workflow_name
    )

    statement = select(WorkflowStepRun.id)
    if ophys_experiment_id is not None:
        statement = statement.where(
            WorkflowStepRun.ophys_experiment_id == ophys_experiment_id)

    statement = (
        statement
        .where(WorkflowStep.id == workflow_step.id)
        .order_by(col(WorkflowStepRun.end).desc())
        .limit(1))
    res = session.exec(statement)
    try:
        workflow_step_run_id = res.one()
    except NoResultFound:
        logger.error(
            f'No workflow step run found for '
            f'ophys experiment {ophys_experiment_id}: '
            f'{workflow_step}, {workflow_name}')
        raise
    return workflow_step_run_id

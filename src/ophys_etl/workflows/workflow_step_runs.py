"""Functions relating to workflow step runs"""
import logging
from pathlib import Path
from typing import Optional, List

from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session, col, select

from ophys_etl.workflows.db import engine
from ophys_etl.workflows.db.db_utils import (
    get_well_known_file_type,
    get_workflow_step_by_name,
)
from ophys_etl.workflows.db.schemas import (
    WellKnownFile,
    WorkflowStep,
    WorkflowStepRun
)
from ophys_etl.workflows.utils.ophys_experiment_utils import \
    get_session_experiment_id_map, get_container_experiment_id_map
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

logger = logging.getLogger(__name__)


def get_well_known_file_for_latest_run(
    engine: Engine,
    workflow_step: WorkflowStepEnum,
    workflow_name: WorkflowNameEnum,
    well_known_file_type: WellKnownFileTypeEnum,
    ophys_experiment_id: Optional[int] = None
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
        `WorkflowNameEnum` enum
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
        workflow_step_run_id = get_latest_workflow_step_run(
            session=session,
            workflow_step=workflow_step,
            workflow_name=workflow_name,
            ophys_experiment_id=ophys_experiment_id,
        )
        well_known_file_type = get_well_known_file_type(
            session=session,
            name=well_known_file_type,
            workflow=workflow_name,
            workflow_step_name=workflow_step,
        )
        statement = select(WellKnownFile.path).where(
            WellKnownFile.workflow_step_run_id == workflow_step_run_id,
            WellKnownFile.well_known_file_type_id == well_known_file_type.id,
        )
        res = session.exec(statement)
        try:
            well_known_file_path = res.one()
        except NoResultFound:
            logger.error(
                f"Could not find latest {well_known_file_type} for "
                f"ophys experiment {ophys_experiment_id}: "
                f"{workflow_step}, {workflow_name}"
            )
            raise
    return Path(well_known_file_path)


def get_latest_workflow_step_run(
    session: Session,
    workflow_step: WorkflowStepEnum,
    workflow_name: WorkflowNameEnum,
    ophys_experiment_id: Optional[int] = None,
    ophys_session_id: Optional[int] = None
) -> int:
    """
    Gets the latest workflow step run id for `workflow_step` as part of
    `workflow_name` for `ophys_experiment_id` if associated with an
    ophys_experiment_id, or `ophys_session_id` if associated with an
    ophys_session_id, or neither if not associated with either.


    Parameters
    ----------
    session
        sqlalchemy session
    workflow_step
        `WorkflowStep` enum
    workflow_name
        `WorkflowNameEnum` enum
    ophys_experiment_id
        Optional ophys experiment id
    ophys_session_id
        Optional ophys session id
    Returns
    -------
    int
        workflow step run id

    Raises
    -------
    NoResultFound
        If workflow step run cannot be found
    """
    if ophys_experiment_id is not None and ophys_session_id is not None:
        raise ValueError('Provide either ophys_experiment_id or '
                         'ophys_session_id, not both')
    workflow_step = get_workflow_step_by_name(
        session=session, name=workflow_step, workflow=workflow_name
    )

    statement = select(WorkflowStepRun.id)
    if ophys_experiment_id is not None:
        statement = statement.where(
            WorkflowStepRun.ophys_experiment_id == ophys_experiment_id
        )
    elif ophys_session_id is not None:
        statement = statement.where(
            WorkflowStepRun.ophys_session_id == ophys_session_id
        )

    statement = (
        statement.join(
            WorkflowStep,
            onclause=WorkflowStep.id == WorkflowStepRun.workflow_step_id)
        .where(WorkflowStep.id == workflow_step.id)
        .order_by(col(WorkflowStepRun.end).desc())
        .limit(1)
    )

    res = session.exec(statement)
    try:
        workflow_step_run_id = res.one()
    except NoResultFound:
        msg_level = 'ophys experiment' if ophys_experiment_id is not None \
            else 'ophys session'
        msg_id = ophys_experiment_id if ophys_experiment_id is not None else \
            ophys_session_id
        logger.error(
            f"No {workflow_step.name} run found for "
            f"{msg_level} {msg_id}: "
            f"{workflow_step}, {workflow_name}"
        )
        raise
    return workflow_step_run_id


def is_level_complete(
    ophys_experiment_id: int,
    workflow_step: WorkflowStepEnum,
    level: str = 'ophys_session'
) -> bool:
    """Returns whether all ophys experiments in same `level` as
    `ophys_experiment_id`have completed `workflow_step`

    Parameters
    ----------
    ophys_experiment_id
        Ophys experiment id
    workflow_step
        Workflow step to check for completion
    level
        The level at which to group experiments. Can be one of:
        - ophys_session
        - ophys_container

    Returns
    -------
    bool
        Whether all ophys experiments in same `level` as
        `ophys_experiment_id`have completed `workflow_step`
    """
    if level == 'ophys_session':
        level_exp_map = get_session_experiment_id_map(
            ophys_experiment_ids=[ophys_experiment_id]
        )
    elif level == 'ophys_container':
        level_exp_map = get_container_experiment_id_map(
            ophys_experiment_ids=[ophys_experiment_id]
        )
    else:
        valid_levels = ('ophys_session', 'ophys_container')
        raise ValueError(f'{level} is invalid. Must be on of {valid_levels}')

    with Session(engine) as session:
        step = get_workflow_step_by_name(
            name=workflow_step,
            workflow=WorkflowNameEnum.OPHYS_PROCESSING,
            session=session
        )
        statement = (
            select(WorkflowStepRun.ophys_experiment_id)
            .where(WorkflowStepRun.workflow_step_id == step.id)
        )
        completed_ophys_experiment_ids = session.exec(statement).all()

    all_ophys_experiment_ids = [
        x['ophys_experiment_id'] for x in level_exp_map]
    for ophys_experiment_id in all_ophys_experiment_ids:
        if ophys_experiment_id not in completed_ophys_experiment_ids:
            return False
    return True


def get_most_recent_run(
    workflow_step: WorkflowStepEnum,
    ophys_experiment_ids: List[int]
) -> int:
    """Gets the most recent `ophys_experiment_id` that has completed
    `workflow_step` from the list of `ophys_experiment_ids`

    Parameters
    ----------
    workflow_step
        The workflow step to get most recent run for
    ophys_experiment_ids
        List of ophys experiment ids to filter by

    Returns
    -------
    int
        ophys experiment id that has most recently completed `workflow_step`
        from the list of `ophys_experiment_ids`
    """
    with Session(engine) as session:
        step = get_workflow_step_by_name(
            name=workflow_step,
            workflow=WorkflowNameEnum.OPHYS_PROCESSING,
            session=session
        )
        statement = (
            select(WorkflowStepRun.ophys_experiment_id)
            .where(WorkflowStepRun.workflow_step_id == step.id,
                   col(WorkflowStepRun.ophys_experiment_id)
                   .in_(ophys_experiment_ids))
            .order_by(col(WorkflowStepRun.insertion_time).desc())
            .limit(1)
        )
        return session.exec(statement).one()

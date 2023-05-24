"""Functions relating to workflow step runs"""
import datetime
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd

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
    WorkflowStepRun, Workflow,
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
    ophys_experiment_id: Optional[str] = None,
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
    ophys_experiment_id: Optional[str] = None,
    ophys_session_id: Optional[str] = None
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
        logger.error(
            f"No workflow step run found for "
            f"ophys experiment {ophys_experiment_id}: "
            f"{workflow_step}, {workflow_name}"
        )
        raise
    return workflow_step_run_id


def get_runs_completed_since(
        session: Session,
        since: datetime.datetime,
        workflow_step: WorkflowStepEnum,
        workflow_name: WorkflowNameEnum
) -> List[WorkflowStepRun]:
    """Gets all `WorkflowStepRun` for `WorkflowStep`
    that have completed `since`

    Parameters
    ----------
    session
        sqlalchemy session
    since
        Check since this datetime
    workflow_step
        Check for this WorkflowStep
    workflow_name
        Check for this workflow
    """
    workflow_step = get_workflow_step_by_name(
        session=session, name=workflow_step, workflow=workflow_name
    )
    statement = (
        select(WorkflowStepRun)
        .join(Workflow, onclause=WorkflowStep.workflow_id == Workflow.id)
        .where(WorkflowStepRun.workflow_step_id == workflow_step.id,
               WorkflowStepRun.end >= since)
    )
    res = session.exec(statement)
    return res.all()


def get_completed(
    ophys_experiment_ids: List[str],
    workflow_step: WorkflowStepEnum,
    level: str = 'ophys_session',

) -> pd.DataFrame:
    """Gets `level` from the list of `ophys_experiment_ids`
    that have completed `workflow_step`. i.e. all experiments in `level`
    have completed `workflow_step`

    Parameters
    ----------
    ophys_experiment_ids
        List of ophys experiment ids
    workflow_step
        Workflow step to check for completion
    level
        The level at which to group experiments. Can be one of:
        - ophys_session
        - ophys_container

    Returns
    -------
    pd.DataFrame
        A dataframe containing all experiments in `level` for all `level`
        that have completed

        Columns:
        - `level`_id
        - ophys_experiment_id
    """
    if level == 'ophys_session':
        level_exp_map = get_session_experiment_id_map(
            ophys_experiment_ids=ophys_experiment_ids
        )
    elif level == 'ophys_container':
        level_exp_map = get_container_experiment_id_map(
            ophys_experiment_ids=ophys_experiment_ids
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
        ophys_experiment_ids = session.exec(statement).all()

    level_exp_map = pd.DataFrame(level_exp_map)
    level_exp_map['has_completed'] = \
        level_exp_map['ophys_experiment_id'].apply(
            lambda x: x in ophys_experiment_ids)
    has_completed = \
        level_exp_map.groupby(f'{level}_id')['has_completed']\
        .all()
    completed = has_completed[has_completed].index
    return completed.tolist()

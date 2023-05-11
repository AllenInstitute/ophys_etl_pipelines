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
    get_session_experiment_id_map
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
        workflow_step_run_id = get_latest_run(
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


def get_latest_run(
    session: Session,
    workflow_step: WorkflowStepEnum,
    workflow_name: WorkflowNameEnum,
    ophys_experiment_id: Optional[str] = None,
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
        `WorkflowNameEnum` enum
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
        session=session, name=workflow_step, workflow=workflow_name
    )

    statement = select(WorkflowStepRun.id)
    if ophys_experiment_id is not None:
        statement = statement.where(
            WorkflowStepRun.ophys_experiment_id == ophys_experiment_id
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


def get_completed_ophys_sessions(
    ophys_experiment_ids: List[str],
    workflow_step: WorkflowStepEnum
):
    """Gets ophys sessions from the list of `ophys_experiment_ids`
    that have completed workflow_step

    Parameters
    ----------
    ophys_experiment_ids
        List of ophys experiment ids
    """
    session_exps = get_session_experiment_id_map(
        ophys_experiment_ids=ophys_experiment_ids
    )

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

    session_exps = pd.DataFrame(session_exps)
    session_exps['has_completed'] = \
        session_exps['ophys_experiment_id'].apply(
            lambda x: x in ophys_experiment_ids)
    has_session_completed = \
        session_exps.groupby('ophys_session_id')['has_completed']\
        .all()
    completed_sessions = has_session_completed[has_session_completed].index
    return completed_sessions.tolist()

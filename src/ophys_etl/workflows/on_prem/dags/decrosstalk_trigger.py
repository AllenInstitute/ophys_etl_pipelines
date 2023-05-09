import datetime
from typing import List, Dict

import logging
import pandas as pd
from airflow.decorators import task
from airflow.models.dag import dag
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from ophys_etl.workflows.db.db_utils import get_workflow_step_by_name

from ophys_etl.workflows.db.schemas import WorkflowStepRun
from sqlalchemy import select

from ophys_etl.workflows.utils.lims_utils import LIMSDB

from ophys_etl.workflows.db import engine
from sqlmodel import Session

from ophys_etl.workflows.utils.dag_utils import get_most_recent_run
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_runs_completed_since
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

logger = logging.getLogger('airflow.task')


def _get_multiplane_experiments(ophys_experiment_ids: List[str]) -> List[str]:
    """Returns only those experiments from ophys_experiment_ids which are
    multiplane. Currently only checks if it is mesoscope"""
    if len(ophys_experiment_ids) == 0:
        return []

    lims_db = LIMSDB()

    if len(ophys_experiment_ids) == 0:
        oe_ids_clause = 'false'
    elif len(ophys_experiment_ids) > 1:
        oe_ids_clause = f'oe.id in {tuple(ophys_experiment_ids)}'
    else:
        oe_ids_clause = f'oe.id = {ophys_experiment_ids[0]}'

    query = f'''
        SELECT oe.id as ophys_experiment_id
        FROM  ophys_experiments oe
        JOIN ophys_sessions os ON oe.ophys_session_id = os.id
        JOIN  equipment
            ON equipment.id = os.equipment_id
        WHERE {oe_ids_clause}
        ) AND
            equipment.name in ('MESO.1', 'MESO.2')
    '''
    res = lims_db.query(query=query)
    ophys_experiment_ids = [x['ophys_experiment_id'] for x in res]
    return ophys_experiment_ids


def _get_session_experiment_id_map(
        ophys_experiment_ids: List[str]
) -> List[Dict]:
    """Get full list of experiment ids for each ophys session that each
    ophys_experiment_id belongs to"""
    lims_db = LIMSDB()

    if len(ophys_experiment_ids) == 0:
        oe_ids_clause = 'false'
    elif len(ophys_experiment_ids) > 1:
        oe_ids_clause = f'oe.id in {tuple(ophys_experiment_ids)}'
    else:
        oe_ids_clause = f'oe.id = {ophys_experiment_ids[0]}'

    query = f'''
        SELECT oe.id as ophys_experiment_id, os.id as ophys_session_id
        FROM  ophys_experiments oe
        JOIN ophys_sessions os ON oe.ophys_session_id = os.id
        WHERE os.id = (
            SELECT oe.ophys_session_id
            FROM ophys_experiments oe
            WHERE {oe_ids_clause}
        )
    '''
    res = lims_db.query(query=query)
    return res


def _get_completed_ophys_sessions(completed_ophys_experiment_ids: List[str]):
    """Gets ophys sessions from the list of `ophys_experiment_ids`
    that have completed segmentation

    Parameters
    ----------
    completed_ophys_experiment_ids
        List of ophys experiment ids that have completed segmentation
    """
    session_exps = _get_session_experiment_id_map(
        ophys_experiment_ids=completed_ophys_experiment_ids
    )

    with Session(engine) as session:
        segmentation_step = get_workflow_step_by_name(
            name=WorkflowStepEnum.SEGMENTATION,
            workflow=WorkflowNameEnum.OPHYS_PROCESSING,
            session=session
        )
        statement = (
            select(WorkflowStepRun.ophys_experiment_id)
            .where(WorkflowStepRun.workflow_step_id == segmentation_step.id)
        )
        completed_ophys_experiment_ids = session.exec(statement).all()

    completed_ophys_experiment_ids = \
        [x.ophys_experiment_id for x in completed_ophys_experiment_ids]
    session_exps = pd.DataFrame(session_exps)
    session_exps['has_completed_segmentation'] = \
        session_exps['ophys_experiment_id'].apply(
            lambda x: x in completed_ophys_experiment_ids)
    has_session_completed_segmentation = \
        session_exps.groupby('ophys_session_id')['has_completed_segmentation']\
        .all()
    completed_sessions = has_session_completed_segmentation[
        has_session_completed_segmentation].index
    return completed_sessions.tolist()


@dag(
    dag_id='decrosstalk_trigger',
    schedule='*/5 * * * *',  # every 5 minutes
    catchup=False,
    start_date=datetime.datetime.now()
)
def decrosstalk_trigger():
    """Checks for any ophys experiments that have completed segmentation
    since the last time this ran. If so, and
    1) this is a multiplane experiment and
    2) all other experiments from this session have also completed segmentation
    then we trigger decrosstalk for this session.
    """
    @task
    def trigger():
        last_run_datetime = get_most_recent_run(
            dag_id='decrosstalk_trigger')
        if last_run_datetime is None:
            # this DAG hasn't been successfully run before
            # nothing to do
            return None
        with Session(engine) as session:
            segmentation_runs = get_runs_completed_since(
                session=session,
                since=last_run_datetime,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )
        ophys_experiment_ids = \
            [x.ophys_experiment_id for x in segmentation_runs]

        ophys_experiment_ids = _get_multiplane_experiments(
            ophys_experiment_ids=ophys_experiment_ids
        )
        completed_ophys_sessions = _get_completed_ophys_sessions(
            completed_ophys_experiment_ids=ophys_experiment_ids)
        for ophys_session_id in completed_ophys_sessions:
            logger.info(
                f'Triggering decrosstalk for ophys session {ophys_session_id}')
            TriggerDagRunOperator(
                task_id='trigger_decrosstalk_for_ophys_session',
                trigger_dag_id='decrosstalk',
                conf={
                    'ophys_session_id': ophys_session_id
                }
            ).execute(context=get_current_context())

    trigger()


decrosstalk_trigger()

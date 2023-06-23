import datetime
from typing import List

import logging
from airflow.decorators import task
from airflow.models.dag import dag
from airflow.operators.python import get_current_context

from ophys_etl.workflows.utils.lims_utils import LIMSDB

from ophys_etl.workflows.db import engine
from sqlmodel import Session

from ophys_etl.workflows.utils.dag_utils import get_latest_dag_run, \
    trigger_dag_runs
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_runs_completed_since, \
    get_completed
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

logger = logging.getLogger('airflow.task')


def _get_multiplane_experiments(ophys_experiment_ids: List[int]) -> List[int]:
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
        WHERE {oe_ids_clause} AND
            equipment.name in ('MESO.1', 'MESO.2')
    '''
    res = lims_db.query(query=query)
    ophys_experiment_ids = [x['ophys_experiment_id'] for x in res]
    return ophys_experiment_ids


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
        last_run_datetime = get_latest_dag_run(
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
        completed_ophys_sessions = get_completed(
            ophys_experiment_ids=ophys_experiment_ids,
            workflow_step=WorkflowStepEnum.SEGMENTATION,
            level='ophys_session'
        )
        trigger_dag_runs(
            key_name='ophys_session_id',
            values=completed_ophys_sessions,
            task_id='trigger_decrosstalk_for_ophys_session',
            trigger_dag_id='decrosstalk',
            context=get_current_context()
        )
    trigger()


decrosstalk_trigger()

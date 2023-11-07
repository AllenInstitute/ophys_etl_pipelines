import datetime
import logging
from typing import List

from airflow.decorators import task
from airflow.models.dag import dag
from airflow.operators.python import get_current_context
from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.utils.dag_utils import get_latest_dag_run, \
    trigger_dag_runs

from ophys_etl.workflows.utils.lims_utils import LIMSDB


def _get_all_ophys_experiments_completed_since(
        since: datetime.datetime
) -> List[int]:
    """
    Get all ophys experiments which successfully completed one of the
    `LIMS_OPHYS_PROCESSING_TRIGGER_QUEUES` since `since`

    Parameters
    ----------
    since
        Check for jobs that completed since this time

    Returns
    -------
    List of ophys experiment LIMS ids

    """
    trigger_queues = app_config.ophys_processing_trigger.lims_trigger_queues
    if len(trigger_queues) == 1:
        trigger_queues = trigger_queues[0]
        queue_query = f'jq.name = \'{trigger_queues}\''
    else:
        queue_query = f'jq.name in {trigger_queues}'

    query = f'''
        SELECT oe.id
        FROM jobs
        JOIN job_queues jq on jq.id = jobs.job_queue_id
        JOIN ophys_sessions os on os.id = jobs.enqueued_object_id
        JOIN ophys_experiments oe on oe.ophys_session_id = os.id
        JOIN job_states js on js.id = jobs.job_state_id
        WHERE {queue_query} AND
                js.name = 'SUCCESS' AND
                completed_at >= '{since}'
    '''
    lims_db = LIMSDB()
    res = lims_db.query(query=query)
    ophys_experiment_ids = [x['id'] for x in res]
    return ophys_experiment_ids


logger = logging.getLogger('airflow.task')


@dag(
    dag_id='ophys_processing_trigger',
    schedule='*/60 * * * *',     # every 60 minutes
    catchup=False,
    start_date=datetime.datetime(
        year=2023, month=8, day=7, hour=1, minute=0, second=0),
    max_active_runs=1
)
def ophys_processing_trigger():
    """Triggers ophys_processing DAG by querying LIMS and looking for any
    experiments ready to be processed that have not been processed yet"""
    @task
    def trigger():
        most_recent_dag_run = get_latest_dag_run(
            dag_id='ophys_processing_trigger',
            states=['success']
        )
        last_success_dag_run_datetime = \
            datetime.datetime.now() if most_recent_dag_run is None \
            else most_recent_dag_run['start_date']
        ophys_experiment_ids = _get_all_ophys_experiments_completed_since(
            since=last_success_dag_run_datetime
        )
        if len(ophys_experiment_ids) == 0:
            logger.info('No new experiments')
        else:
            trigger_dag_runs(
                key_name='ophys_experiment_id',
                values=ophys_experiment_ids,
                task_id='trigger_ophys_processing_for_ophys_experiment',
                trigger_dag_id='ophys_processing',
                context=get_current_context()
            )
    trigger()


ophys_processing_trigger()

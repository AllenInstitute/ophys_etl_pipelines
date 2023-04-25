import base64
import datetime
import logging
from typing import List, Optional

import requests
from airflow.decorators import task
from airflow.models.dag import dag
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.utils.lims_utils import LIMSDB

# List of queues in LIMS that should trigger ophys processing
LIMS_OPHYS_PROCESSING_TRIGGER_QUEUES = (
    'MESOSCOPE_FILE_SPLITTING_QUEUE',
    'DEWARPING_QUEUE',
    'DEEPSCOPE_SESSION_UPLOAD_QUEUE',
    'BESSEL_SESSION_UPLOAD_QUEUE'
)


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
    query = f'''
        SELECT oe.id
        FROM jobs
        JOIN job_queues jq on jq.id = jobs.job_queue_id
        JOIN ophys_sessions os on os.id = jobs.enqueued_object_id
        JOIN ophys_experiments oe on oe.ophys_session_id = os.id
        JOIN job_states js on js.id = jobs.job_state_id
        WHERE jq.name in {LIMS_OPHYS_PROCESSING_TRIGGER_QUEUES} AND
                js.name = 'SUCCESS' AND
                completed_at >= '{since}'
    '''
    lims_db = LIMSDB()
    res = lims_db.query(query=query)
    ophys_experiment_ids = [x['id'] for x in res]
    return ophys_experiment_ids


def _get_most_recent_run() -> Optional[datetime.datetime]:
    """Gets the most recent run of this DAG, or None if not run before"""
    rest_api_username = \
        app_config.airflow_rest_api_credentials.username.get_secret_value()
    rest_api_password = \
        app_config.airflow_rest_api_credentials.password.get_secret_value()
    auth = base64.b64encode(
        f'{rest_api_username}:{rest_api_password}'.encode('utf-8'))
    r = requests.get(
        'http://0.0.0.0:8080/api/v1/dags/ophys_processing_trigger/dagRuns?'
        'limit=1&'
        'order_by=-execution_date&'
        'state=success',
        headers={
            'Authorization': f'Basic {auth.decode()}'
        }
    )
    response = r.json()
    if len(response['dag_runs']) == 0:
        last_run_datetime = None
    else:
        last_dag_run = response['dag_runs'][0]
        last_run_datetime = datetime.datetime.strptime(
            last_dag_run['logical_date'], '%Y-%m-%dT%H:%M:%S.%f%z')
    return last_run_datetime


logger = logging.getLogger('airflow.task')


@dag(
    dag_id='ophys_processing_trigger',
    schedule='*/5 * * * *',     # every 5 minutes
    catchup=False,
    start_date=datetime.datetime.now()
)
def ophys_processing_trigger():

    @task
    def trigger():
        last_run_datetime = _get_most_recent_run()
        if last_run_datetime is None:
            # this DAG hasn't been successfully run before
            # nothing to do
            return None
        ophys_experiment_ids = _get_all_ophys_experiments_completed_since(
            since=last_run_datetime
        )
        for ophys_experiment_id in ophys_experiment_ids:
            logger.info(f'Triggering ophys_processing DAG for '
                        f'{ophys_experiment_id}')
            TriggerDagRunOperator(
                task_id='trigger_ophys_processing_for_ophys_experiment',
                trigger_dag_id='ophys_processing',
                conf={
                    'ophys_experiment_id': ophys_experiment_id
                }
            ).execute(context=get_current_context())
    trigger()


ophys_processing_trigger()
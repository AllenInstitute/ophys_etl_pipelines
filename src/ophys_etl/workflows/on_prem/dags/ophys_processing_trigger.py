import base64
import datetime
import logging
from typing import List

import requests
from airflow.decorators import task
from airflow.models.dag import dag
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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
    `LIMS_OPHYS_PROCESSING_TRIGGER_QUEUES`

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
        auth = base64.b64encode(b'rest_api:1234')
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
            # this DAG hasn't been successfully run before
            # nothing to do
            return

        last_dag_run = response['dag_runs'][0]
        last_run_datetime = datetime.datetime.strptime(
            last_dag_run['logical_date'], '%Y-%m-%dT%H:%M:%S.%f%z')
        ophys_experiment_ids = _get_all_ophys_experiments_completed_since(
            since=last_run_datetime
        )
        for ophys_experiment_id in ophys_experiment_ids:
            print(ophys_experiment_id)
            TriggerDagRunOperator(
                trigger_dag_id='ophys_processing',
                conf={
                    'params': {
                        'ophys_experiment_id': ophys_experiment_id
                    }
                }
            )
    trigger()


if __name__ == '__main__':
    ophys_processing_trigger().test()

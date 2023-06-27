import base64
import datetime
import logging
import time
from typing import Optional, Dict, List, Any

import requests
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.utils.airflow_utils import get_rest_api_port, \
    REST_API_RETRY_SECONDS, MAX_REST_API_RETRIES

logger = logging.getLogger(__name__)


def get_latest_dag_run(
        dag_id: str,
        states: Optional[List[str]] = None
) -> Optional[Dict]:
    """Gets the most recent run of this DAG with a state in `states`, or None

    Parameters
    ----------
    dag_id
        Gets most recent run for this dag id
    states
        Filter by dag runs with any of these states

    """
    if states is None:
        states = ['success']
    states_query = '&'.join([f'state={x}' for x in states])

    rest_api_username = \
        app_config.airflow_rest_api_credentials.username.get_secret_value()
    rest_api_password = \
        app_config.airflow_rest_api_credentials.password.get_secret_value()
    auth = base64.b64encode(
        f'{rest_api_username}:{rest_api_password}'.encode('utf-8'))
    rest_api_port = get_rest_api_port()

    num_tries = 0
    url = f'http://0.0.0.0:{rest_api_port}/api/v1/dags/{dag_id}/' \
          f'dagRuns?limit=1&order_by=-execution_date&{states_query}'

    while True:
        try:
            r = requests.get(
                url=url,
                headers={
                    'Authorization': f'Basic {auth.decode()}'
                }
            )
            break
        # This error is sporadically thrown
        except requests.exceptions.ConnectionError as e:
            num_tries += 1
            if num_tries > MAX_REST_API_RETRIES:
                logger.error('Reached max numb of retries')
                raise e
            logger.error(f'Call to {url} failed with {e}. Retrying again in '
                         f'{REST_API_RETRY_SECONDS} seconds')
            time.sleep(REST_API_RETRY_SECONDS)

    response = r.json()
    if len(response['dag_runs']) == 0:
        last_dag_run = None
    else:
        last_dag_run = response['dag_runs'][0]
        try:
            last_dag_run['logical_date'] = datetime.datetime.strptime(
                last_dag_run['logical_date'], '%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            # try without fractional seconds
            last_dag_run['logical_date'] = datetime.datetime.strptime(
                last_dag_run['logical_date'], '%Y-%m-%dT%H:%M:%S%z')
    return last_dag_run


def trigger_dag_run(
    task_id: str,
    trigger_dag_id: str,
    context: Dict,
    conf: Dict
):
    """Triggers dag run for `trigger_dag_id`

    Parameters
    ----------
    task_id
        See `TriggerDagRunOperator`
    trigger_dag_id
        See `TriggerDagRunOperator`
    context
        Context in which trigger is running (as returned by
        `get_current_context`)
    conf
        See `TriggerDagRunOperator`
    """
    now = datetime.datetime.now().astimezone(
        tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%s%z')

    trigger_run_id = f'{now}_triggered_by_' \
                     f'{context["dag_run"].dag_id}_' \
                     f'{context["dag_run"].run_id}'
    logger.info(f'Triggering {trigger_dag_id} for {conf}')
    TriggerDagRunOperator(
        task_id=task_id,
        trigger_dag_id=trigger_dag_id,
        conf=conf,
        trigger_run_id=trigger_run_id
    ).execute(context=context)


def trigger_dag_runs(
    key_name: str,
    values: List[Any],
    task_id: str,
    trigger_dag_id: str,
    context: Dict
):
    """Triggers dag runs for `trigger_dag_id` for all of `values`

    Parameters
    ----------
    task_id
        See `TriggerDagRunOperator`
    trigger_dag_id
        See `TriggerDagRunOperator`
    context
        Context in which trigger is running (as returned by
        `get_current_context`)
    key_name
        The value type (ophys_experiment_id, etc)
    values
        The values to submit dag runs for
    """
    for value in values:
        trigger_dag_run(
            task_id=task_id,
            trigger_dag_id=trigger_dag_id,
            conf={
                key_name: value
            },
            context=context
        )
        # Sleeping so that we get a unique dag run id
        time.sleep(1)

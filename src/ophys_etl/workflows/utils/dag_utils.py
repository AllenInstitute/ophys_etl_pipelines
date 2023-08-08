import datetime
import logging
import time
from typing import Optional, Dict, List, Any

from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.utils.airflow_utils import call_endpoint_with_retries

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

    url = f'http://{app_config.webserver.host_name}:8080/api/v1/dags/' \
          f'{dag_id}/dagRuns?limit=1&order_by=-execution_date&{states_query}'

    response = call_endpoint_with_retries(
        url=url,
        http_method='GET'
    )

    if len(response['dag_runs']) == 0:
        last_dag_run = None
    else:
        last_dag_run = response['dag_runs'][0]
        try:
            last_dag_run['start_date'] = datetime.datetime.strptime(
                last_dag_run['start_date'], '%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            # try without fractional seconds
            last_dag_run['start_date'] = datetime.datetime.strptime(
                last_dag_run['start_date'], '%Y-%m-%dT%H:%M:%S%z')
    return last_dag_run


def trigger_dag_run(
    task_id: str,
    trigger_dag_id: str,
    context: Dict,
    conf: Dict,
    object_type: Optional[str] = None,
    object_id: Optional[int] = None
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
    object_type
        ophys_experiment_id, ophys_session_id or ophys_container_id
    object_id
        identifier for object_type
    """
    if object_type is not None:
        valid_object_types = ('ophys_experiment_id', 'ophys_session_id',
                              'ophys_container_id')
        if object_type not in valid_object_types:
            raise ValueError(f'object type must be one of '
                             f'{valid_object_types}. '
                             f'Gave {object_type}')
        if object_id is None:
            raise ValueError('Must give object_id if object_type given')
    now = datetime.datetime.now().astimezone(
        tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%s%z')

    trigger_run_id = f'{now}_triggered_by_' \
                     f'{context["dag_run"].dag_id}_' \
                     f'{context["dag_run"].run_id}'
    if object_type is not None:
        trigger_run_id = f'{object_type}_{object_id}_{trigger_run_id}'

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
            context=context,
            object_type=key_name,
            object_id=value
        )
        # Sleeping so that we get a unique dag run id
        time.sleep(1)

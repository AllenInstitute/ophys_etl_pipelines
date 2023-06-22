import base64
import datetime
import logging
import time
from typing import Optional

import requests

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.utils.airflow_utils import get_rest_api_port, \
    REST_API_RETRY_SECONDS, MAX_REST_API_RETRIES

logger = logging.getLogger(__name__)

def get_latest_dag_run(
        dag_id: str,
        state: str = 'success'
) -> Optional[datetime.datetime]:
    """Gets the most recent run of this DAG, or None if not run before

    Parameters
    ----------
    dag_id
        Gets most recent run for this dag id
    state
        Filter by dag runs with this state

    """
    rest_api_username = \
        app_config.airflow_rest_api_credentials.username.get_secret_value()
    rest_api_password = \
        app_config.airflow_rest_api_credentials.password.get_secret_value()
    auth = base64.b64encode(
        f'{rest_api_username}:{rest_api_password}'.encode('utf-8'))
    rest_api_port = get_rest_api_port()

    num_tries = 0
    url = f'http://0.0.0.0:{rest_api_port}/api/v1/dags/{dag_id}/' \
          f'dagRuns?limit=1&order_by=-execution_date&state={state}'

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
        last_run_datetime = None
    else:
        last_dag_run = response['dag_runs'][0]
        try:
            last_run_datetime = datetime.datetime.strptime(
                last_dag_run['logical_date'], '%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            # try without fractional seconds
            last_run_datetime = datetime.datetime.strptime(
                last_dag_run['logical_date'], '%Y-%m-%dT%H:%M:%S%z')
    return last_run_datetime

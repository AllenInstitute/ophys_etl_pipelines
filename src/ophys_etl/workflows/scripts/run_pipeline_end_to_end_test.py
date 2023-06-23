"""This script triggers ophys processing for all experiments within
the same session and container. This should then trigger other downstream
processing once all have finished"""

import datetime

import base64

import json
import time

import requests

from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.utils.airflow_utils import get_rest_api_port

from ophys_etl.workflows.utils.ophys_experiment_utils import \
    get_session_experiment_id_map, get_container_experiment_id_map


def main():
    ophys_experiment_id = 1274961379
    oe_session_map = get_session_experiment_id_map(
        ophys_experiment_ids=[ophys_experiment_id]
    )
    oe_container_map = get_container_experiment_id_map(
        ophys_experiment_ids=[x['ophys_experiment_id'] for x in oe_session_map]
    )

    # get oe_session_map again to get all sessions for all containers
    oe_session_map = get_session_experiment_id_map(
        ophys_experiment_ids=[x['ophys_experiment_id'] for x in oe_container_map]
    )

    # getting all ophys experiments either in session with
    # `ophys_experiment_ids` or container
    ophys_experiments = list(set(
        [x['ophys_experiment_id'] for x in oe_session_map] +
        [x['ophys_experiment_id'] for x in oe_container_map]
    ))

    print(f'Total number of ophys processing jobs: {len(ophys_experiments)}')
    print(f'Total number of containers: '
          f'{len(set([x["ophys_container_id"] for x in oe_container_map]))}')
    print(f'Total number of sessions: '
          f'{len(set([x["ophys_session_id"] for x in oe_session_map]))}')

    for oe in ophys_experiments:
        rest_api_username = \
            app_config.airflow_rest_api_credentials.username.get_secret_value()
        rest_api_password = \
            app_config.airflow_rest_api_credentials.password.get_secret_value()
        auth = base64.b64encode(
            f'{rest_api_username}:{rest_api_password}'.encode('utf-8'))
        rest_api_port = get_rest_api_port()
        r = requests.post(
            url=f'http://0.0.0.0:{rest_api_port}/api/v1/dags/ophys_processing/'
                f'dagRuns',
            headers={
                'Authorization': f'Basic {auth.decode()}',
                'Content-type': 'application/json'
            },
            data=json.dumps({
                'logical_date': (
                        datetime.datetime.utcnow()
                        .strftime('%Y-%m-%dT%H:%M:%S%z') + '+00:00'),
                'conf': {'ophys_experiment_id': oe}
            })
        )
        print(r.json())

        # otherwise run ids are the same and get an error
        time.sleep(1)


if __name__ == '__main__':
    main()

"""This script triggers ophys processing for all experiments within
the same session and container. This should then trigger other downstream
processing once all have finished"""

import datetime

import base64

import json
import time

import requests
from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.utils.airflow_utils import get_rest_api_port


def main():
    ophys_experiment = OphysExperiment.from_id(id=1274961379)
    experiments_in_session = ophys_experiment.session\
        .get_ophys_experiment_ids(passed_or_qc_only=False)
    experiments_in_container = ophys_experiment.container\
        .get_ophys_experiment_ids(passed_or_qc_only=False)

    # getting all ophys experiments associated with ophys_experiment
    ophys_experiments = set(
        experiments_in_session +
        experiments_in_container
    )

    # need to get all experiments from all sessions so that decrosstalk is
    # run
    for ophys_experiment_id in experiments_in_container:
        for other_experiment_id in OphysExperiment.from_id(
                id=ophys_experiment_id).session.get_ophys_experiment_ids(
                passed_or_qc_only=False):
            ophys_experiments.add(other_experiment_id)

    ophys_experiments = list(ophys_experiments)

    containers = [
        OphysExperiment.from_id(id=ophys_experiment_id).container
        for ophys_experiment_id in ophys_experiments]
    n_containers = len(set([x.id for x in containers if x.id is not None]))

    sessions = [
        OphysExperiment.from_id(id=ophys_experiment_id).session
        for ophys_experiment_id in ophys_experiments]
    n_sessions = len(set([x.id for x in sessions]))

    print(f'Total number of ophys processing jobs: {len(ophys_experiments)}')
    print(f'Total number of containers: {n_containers}')
    print(f'Total number of sessions: {n_sessions}')

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

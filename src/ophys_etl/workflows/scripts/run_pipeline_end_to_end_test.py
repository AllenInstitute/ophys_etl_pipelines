"""This script triggers ophys processing for all experiments within
the same session and container. This should then trigger other downstream
processing once all have finished"""

import datetime

import time

from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.utils.airflow_utils import call_endpoint_with_retries


def main():
    ophys_experiment = OphysExperiment.from_id(id=1274961379)
    experiments_in_session = ophys_experiment.session\
        .get_ophys_experiment_ids(passed_or_qc_only=False)
    experiments_in_container = ophys_experiment.container\
        .get_ophys_experiment_ids(passed_or_qc_only=True)

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
        response = call_endpoint_with_retries(
            url=f'http://{app_config.webserver.host_name}:8080/api/v1/dags/'
                f'ophys_processing/dagRuns',
            http_method='POST',
            http_body={
                'logical_date': (
                        datetime.datetime.utcnow()
                        .strftime('%Y-%m-%dT%H:%M:%S%z') + '+00:00'),
                'conf': {'ophys_experiment_id': oe}
            }
        )
        print(response)

        # otherwise run ids are the same and get an error
        time.sleep(1)


if __name__ == '__main__':
    main()

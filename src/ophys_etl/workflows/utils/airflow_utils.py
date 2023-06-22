import configparser
import os
from pathlib import Path

# Max number of retries to make when attempting to query airflow rest api
MAX_REST_API_RETRIES = 20

# Seconds to wait before retrying again
REST_API_RETRY_SECONDS = 10


def get_rest_api_port() -> str:
    """Get web server port defined in airflow cfg

    Raises
    ------
    ValueError
        if AIRFLOW_HOME not set
    """
    airflow_home = os.getenv('AIRFLOW_HOME', None)
    if airflow_home is None:
        raise ValueError('Env var AIRFLOW_HOME not set')
    config = configparser.ConfigParser()
    config.read(f'{Path(airflow_home)}/airflow.cfg')
    return config['webserver']['web_server_port']

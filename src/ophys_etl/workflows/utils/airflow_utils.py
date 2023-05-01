import configparser
import os
from pathlib import Path


def get_rest_api_port() -> str:
    """Get web server port defined in airflow cfg

    Raises
    ------
    ValueError
        if AIRFLOW_HOME not set
    """
    airflow_home = os.getenv('AIRFLOW_HOME', None)
    if airflow_home is None:
        raise ValueError('Env var AIRFLOW not set')
    config = configparser.ConfigParser()
    config.read(f'{Path(airflow_home)}/airflow.cfg')
    return config['webserver']['web_server_port']

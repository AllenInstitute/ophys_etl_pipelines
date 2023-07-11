import configparser
import logging
import os
import time
from pathlib import Path

import base64
from typing import Any, Optional, Dict

import json
import requests

from ophys_etl.workflows.app_config.app_config import app_config

logger = logging.getLogger(__name__)


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


def call_endpoint_with_retries(
    url: str,
    http_method: str,
    http_body: Optional[Dict] = None,
    max_retries: int = 20,
    retry_seconds: int = 10,
) -> Any:
    """
    Call api endpoint and retry if it fails

    Parameters
    ----------
    url
        API endpoint
    http_method
        GET or POST or PATCH
    http_body
        if POST or PATCH, the http body
        https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages#body

        it can be an arbitrary json-encodable dictionary
    max_retries
        Max number of retries to make when attempting to query airflow rest api
    retry_seconds
        Seconds to wait before retrying again
    Returns
    -------
    API response
    """
    rest_api_username = \
        app_config.airflow_rest_api_credentials.username.get_secret_value()
    rest_api_password = \
        app_config.airflow_rest_api_credentials.password.get_secret_value()
    auth = base64.b64encode(
        f'{rest_api_username}:{rest_api_password}'.encode('utf-8'))

    num_tries = 0

    while True:
        try:
            if http_method == 'GET':
                r = requests.get(
                    url=url,
                    headers={
                        'Authorization': f'Basic {auth.decode()}',
                        'Accept': 'application/json'
                    }
                )
            elif http_method == 'POST':
                r = requests.post(
                    url=url,
                    data=json.dumps(http_body),
                    headers={
                        'Authorization': f'Basic {auth.decode()}',
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )
            elif http_method == 'PATCH':
                r = requests.patch(
                    url=url,
                    data=json.dumps(http_body),
                    headers={
                        'Authorization': f'Basic {auth.decode()}',
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )
            else:
                raise ValueError(f'Only GET,POST,PATCH supported, gave '
                                 f'{http_method}')
            break
        # This error is sporadically thrown
        except requests.exceptions.ConnectionError as e:
            num_tries += 1
            if num_tries > max_retries:
                logger.error('Reached max numb of retries')
                raise e
            logger.error(f'Call to {url} failed with {e}. Retrying again in '
                         f'{retry_seconds} seconds')
            time.sleep(retry_seconds)

    response = r.json()
    return response

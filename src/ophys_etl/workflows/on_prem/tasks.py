"""On-prem specific airflow tasks"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import jinja2
from airflow.decorators import task
from airflow.models import TaskInstance, clear_task_instances
from airflow.operators.python import get_current_context
from airflow.sensors.base import PokeReturnValue
from airflow.utils.session import provide_session
from ophys_etl.workflows.app_config.app_config import app_config
from paramiko import AuthenticationException

from ophys_etl.workflows.app_config.slurm import SlurmSettings
from ophys_etl.workflows.on_prem.slurm.slurm import (
    Slurm,
    SlurmJob,
    logger, SlurmJobFailedException,
)
from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession, OphysContainer
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.utils.airflow_utils import call_endpoint_with_retries
from ophys_etl.workflows.utils.json_utils import EnhancedJSONEncoder


@provide_session
def _clear_task(task_instance: TaskInstance, session=None):
    """Clears (retries) a task instance"""
    clear_task_instances(tis=[task_instance], session=session)


def _can_retry_task_instance(task_instance: TaskInstance):
    """Return whether the task instance has reached the max number of tries
    as defined in the config"""
    url = f'http://{app_config.webserver.host_name}:8080/api/v1/config'
    config = call_endpoint_with_retries(
        url=url,
        http_method='GET'
    )
    core_section = \
        [x for x in config['sections'] if x['name'] == 'core'][0]
    default_task_tries = [
        x['value'] for x in core_section['options'] if
        x['key'] == 'default_task_retries'][0]
    default_task_tries = int(default_task_tries)
    return task_instance.try_number < default_task_tries


def wait_for_job_to_finish(timeout: float) -> Callable:
    """
    Returns function which waits for batch job to finish
    It determines whether the job is finished by parsing the output from sacct
    command

    Notes
    ------
    Wrapping a `task.sensor` so that we can pass in a custom timeout

    Parameters
    ----------
    timeout
        Timeout in seconds

    Returns
    -------
    Callable decorated with `task.sensor`
    """

    @task.sensor(mode="reschedule", timeout=timeout)
    def wait_for_job_to_finish(
        job_id: str,
        module_outputs: List[Dict],
        storage_directory: str,
        log_path: str
    ):
        context = get_current_context()

        try:
            job = SlurmJob.from_job_id(job_id=job_id)
        except AuthenticationException as e:
            # This was periodically thrown. Catch it so that it doesn't fail
            # the task, and we can just retry the connection again
            logger.error(e)
            return PokeReturnValue(is_done=False, xcom_value=None)
        job_state = job.state.value if job.state is not None else None
        msg = f"job {job.id} state is {job_state}"
        logger.info(msg)

        if job.is_failed():
            all_tasks = context["dag_run"].get_task_instances()
            submit_job_instance: TaskInstance = \
                [x for x in all_tasks if 'submit_job' in x.task_id][0]
            if _can_retry_task_instance(task_instance=submit_job_instance):
                logger.info(f'Clearing {submit_job_instance.task_id}')
                _clear_task(task_instance=submit_job_instance)
            else:
                logger.info('Reached max number of tries.')
            raise SlurmJobFailedException(msg)

        if job.is_done():
            xcom_value = {
                "module_outputs": module_outputs,
                "storage_directory": storage_directory,
                "log_path": log_path,
                "start": str(job.start),
                "end": str(job.end),
            }
            xcom_value = json.dumps(xcom_value, cls=EnhancedJSONEncoder)
        else:
            xcom_value = None
        return PokeReturnValue(is_done=job.is_done(), xcom_value=xcom_value)

    return wait_for_job_to_finish


@task
def submit_job(
    module: Type[PipelineModule],
    config: SlurmSettings,
    docker_tag: str,
    module_kwargs: Optional[Dict] = None,
    **context,
) -> Dict:
    """

    Parameters
    ----------
    module
        `PipelineModule` to submit job for
    config
        Slurm config for this job
    module_kwargs
        Optional kwargs to send to `PipelineModule`
    docker_tag
        Docker tag to use to run job
    context
        airflow context

    Returns
    -------
    Dict with keys
        - job_id: slurm job id
        - module_outputs: Expected List[OutputFile] for module
        - storage_directory: where module is writing outputs to
        - log_path: where logs are being saved for this run
    """
    if module_kwargs is None:
        module_kwargs = {}

    ids_passed = []
    if context['params'].get('ophys_experiment_id') is not None:
        ophys_experiment_id = context["params"]["ophys_experiment_id"]
        ophys_experiment = OphysExperiment.from_id(id=ophys_experiment_id)
        ids_passed.append('ophys_experiment_id')
    else:
        ophys_experiment = None

    if context['params'].get('ophys_session_id') is not None:
        ophys_session_id = context["params"]["ophys_session_id"]
        ophys_session = OphysSession.from_id(id=ophys_session_id)
        ids_passed.append('ophys_session_id')

    else:
        ophys_session = None

    if context['params'].get('ophys_container_id') is not None:
        ophys_container_id = context["params"]["ophys_container_id"]
        ophys_container = OphysContainer.from_id(id=ophys_container_id)
        ids_passed.append('ophys_container_id')

    else:
        ophys_container = None

    if sum([ophys_session is not None,
            ophys_experiment is not None,
            ophys_container is not None]
           ) > 1:
        raise ValueError(
            'Expected one of  ophys_session_id, ophys_experiment_id, '
            f'ophys_container_id. Got {ids_passed}')

    if sum([ophys_session is None,
            ophys_experiment is None,
            ophys_container is None]
           ) == 0:
        raise ValueError(
            'Expected one of ophys_session_id, ophys_experiment_id, '
            'ophys_container_id to be passed as a DAG param')

    for k, v in module_kwargs.items():
        if isinstance(v, Dict) and 'path' in v and 'well_known_file_type' in v:
            # serialize to OutputFile
            # needed because unable to pass around OutputFile in tasks
            module_kwargs[k] = OutputFile.from_dict(x=v)

    mod = module(
        ophys_experiment=ophys_experiment,
        ophys_session=ophys_session,
        ophys_container=ophys_container,
        docker_tag=docker_tag,
        **module_kwargs,
    )
    mod.write_input_args()

    log_path = _get_log_path(task_instance=context["task_instance"])

    slurm = Slurm(
        pipeline_module=mod,
        config=config,
        log_path=log_path,
    )

    slurm.submit_job(
        *mod.executable_args["args"], **mod.executable_args["kwargs"]
    )

    # deserialize OutputFile to dict.
    # needed because unable to pass around OutputFile in tasks
    module_outputs = [
        {
            'path': str(output.path),
            'well_known_file_type': output.well_known_file_type.value
        }
        for output in mod.outputs]

    return {
        "job_id": slurm.job.id,
        "module_outputs": module_outputs,
        "storage_directory": str(mod.output_path),
        "log_path": str(log_path),
    }


def _get_log_path(
    task_instance: TaskInstance,
) -> Path:
    """Returns the path that the current task is writing logs to, so that
    we can write slurm job logs to the same file and view the slurm logs in
    the UI"""
    url = f'http://{app_config.webserver.host_name}:8080/api/v1/config'
    config = call_endpoint_with_retries(
        url=url,
        http_method='GET'
    )
    logging_section = \
        [x for x in config['sections'] if x['name'] == 'logging'][0]
    base_log_folder = [
        x['value'] for x in logging_section['options'] if
        x['key'] == 'base_log_folder'][0]
    log_filename_template = [
        x['value'] for x in logging_section['options'] if
        x['key'] == 'log_filename_template'][0]

    environment = jinja2.Environment()
    template = environment.from_string(log_filename_template)
    log_filename = template.render(
        ti=task_instance,
        try_number=task_instance.try_number)
    return Path(base_log_folder) / log_filename

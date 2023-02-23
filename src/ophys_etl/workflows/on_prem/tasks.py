"""On-prem specific airflow tasks"""
import json
from pathlib import Path
from typing import Callable, List, Type, Optional, Dict

from airflow.decorators import task
from airflow.models import TaskInstance
from airflow.sensors.base import PokeReturnValue
from airflow.utils.log.file_task_handler import FileTaskHandler
from paramiko import AuthenticationException

from ophys_etl.workflows.on_prem.slurm.slurm import SlurmJob, logger, \
    SlurmJobFailedException, Slurm
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.pipeline_module import PipelineModule, OutputFile
from ophys_etl.workflows.utils.json_utils import EnhancedJSONEncoder


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
    @task.sensor(mode='reschedule', timeout=timeout)
    def wait_for_job_to_finish(
            job_id: str,
            module_outputs: List[OutputFile],
            storage_directory: str
    ):
        try:
            job = SlurmJob.from_job_id(job_id=job_id)
        except AuthenticationException as e:
            # This was periodically thrown. Catch it so that it doesn't fail
            # the task, and we can just retry the connection again
            logger.error(e)
            return PokeReturnValue(
                is_done=False,
                xcom_value=None
            )
        job_state = job.state.value if job.state is not None else None
        msg = f'job {job.id} state is {job_state}'
        logger.info(msg)

        if job.is_failed():
            raise SlurmJobFailedException(msg)

        if job.is_done():
            xcom_value = {
                'module_outputs': module_outputs,
                'storage_directory': storage_directory,
                'start': str(job.start),
                'end': str(job.end)
            }
            xcom_value = json.dumps(xcom_value, cls=EnhancedJSONEncoder)
        else:
            xcom_value = None
        return PokeReturnValue(
            is_done=job.is_done(),
            xcom_value=xcom_value)
    return wait_for_job_to_finish


@task
def submit_job(
        module: Type[PipelineModule],
        config_path: str,
        module_kwargs: Optional[Dict] = None,
        docker_tag: str = 'main',
        **context
) -> Dict:
    if module_kwargs is None:
        module_kwargs = {}

    ophys_experiment_id = context['params']['ophys_experiment_id']
    oe = OphysExperiment.from_id(id=ophys_experiment_id)
    mod = module(
        ophys_experiment=oe,
        debug=context['params']['debug'],
        prevent_file_overwrites=context['params']['prevent_file_overwrites'],
        docker_tag=docker_tag,
        **module_kwargs
    )
    mod.write_input_args()

    slurm = Slurm(
        pipeline_module=mod,
        ophys_experiment_id=ophys_experiment_id,
        config_path=Path(config_path),
        log_path=_get_log_path(task_instance=context['task_instance'])
    )

    slurm.submit_job(
        *mod.executable_args['args'],
        **mod.executable_args['kwargs']
    )

    return {
        'job_id': slurm.job.id,
        'module_outputs': mod.outputs,
        'storage_directory': str(mod.output_path),
    }


def _get_log_path(
    task_instance: TaskInstance,
) -> Path:
    """Returns the path that the current task is writing logs to, so that
    we can write slurm job logs to the same file and view the slurm logs in
    the UI"""
    file_handler: FileTaskHandler = logger.handlers[0]
    log_dir = Path(file_handler.local_base)
    log_filename = file_handler._render_filename(
        ti=task_instance, try_number=task_instance.try_number)
    return log_dir / log_filename

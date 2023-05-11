from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.on_prem.tasks import (
    submit_job,
    wait_for_job_to_finish,
)
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.tasks import save_job_run_to_db
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


def run_workflow_step(
    module: Type[PipelineModule],
    workflow_name: WorkflowNameEnum,
    workflow_step_name: WorkflowStepEnum,
    docker_tag: Optional[str] = None,
    slurm_config_filename: Optional[str] = None,
    module_kwargs: Optional[Dict] = None,
    additional_db_inserts: Optional[Callable] = None,
) -> Any:
    """
    Runs a single workflow step

    Parameters
    ----------
    slurm_config_filename
        Slurm settings filename
        If not provided, will load default
    module
        What module to run
    workflow_step_name
        Workflow step name
    workflow_name
        workflow name
    docker_tag
        What docker tag to use.
        Uses default if not provided
    module_kwargs
        kwargs to send to module
    additional_db_inserts
        An optional function which inserts arbitrary data into the app DB


    Returns
    -------
    Dictionary mapping WellKnownFileType to OutputFile, but actually an
    airflow.models.XCom until pulled in a task
    """
    job_finish_res = submit_job_and_wait_to_finish(
        module=module,
        docker_tag=docker_tag,
        slurm_config_filename=slurm_config_filename,
        module_kwargs=module_kwargs,
    )
    run = save_job_run_to_db(
        workflow_name=workflow_name,
        workflow_step_name=workflow_step_name,
        job_finish_res=job_finish_res,
        additional_steps=additional_db_inserts,
    )

    return run["output_files"]


def submit_job_and_wait_to_finish(
    module: Type[PipelineModule],
    docker_tag: Optional[str] = None,
    slurm_config_filename: Optional[str] = None,
    module_kwargs: Optional[Dict] = None,
) -> str:
    """
    Submits slurm job and periodically checks whether it is finished by
    reading the output of sacct command

    Parameters
    ----------
    module
        See `run_workflow_step`
    docker_tag
        See `run_workflow_step`
    slurm_config_filename
        See `run_workflow_step`
    module_kwargs
        See `run_workflow_step`

    Returns
    -------
    See `ophys_etl.workflows.tasks.save_job_run_to_db`
    """
    slurm_config_filename = (
        "default.yml"
        if slurm_config_filename is None
        else slurm_config_filename
    )
    slurm_config = (
        Path(__file__).parent.parent
        / "slurm"
        / "configs"
        / slurm_config_filename
    )

    job_submit_res = submit_job(
        module=module,
        config_path=str(slurm_config),
        docker_tag=docker_tag,
        module_kwargs=module_kwargs,
    )

    job_finish_res = wait_for_job_to_finish(timeout=app_config.job_timeout)(
        job_id=job_submit_res["job_id"],
        storage_directory=job_submit_res["storage_directory"],
        log_path=job_submit_res["log_path"],
        module_outputs=(job_submit_res["module_outputs"]),
    )
    return job_finish_res

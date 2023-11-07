"""Airflow tasks"""

import datetime
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Literal

from airflow.decorators import task
from airflow.operators.python import get_current_context
from airflow.sensors.base import PokeReturnValue
from ophys_etl.workflows.app_config.app_config import app_config
from sqlalchemy.exc import NoResultFound

from ophys_etl.workflows.utils.dag_utils import get_latest_dag_run
from ophys_etl.workflows.workflow_step_runs import get_latest_workflow_step_run

from ophys_etl.workflows.ophys_experiment import OphysExperiment
from sqlmodel import Session

from ophys_etl.workflows.db import engine
from ophys_etl.workflows.db.db_utils import (
    save_job_run_to_db as _save_job_run_to_db,
)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

logger = logging.getLogger(__name__)


@task
def save_job_run_to_db(
    workflow_name: WorkflowNameEnum,
    workflow_step_name: WorkflowStepEnum,
    job_finish_res: str,
    additional_steps: Optional[Callable] = None,
    additional_steps_kwargs: Optional[Dict] = None,
    **context
) -> Dict[str, OutputFile]:
    """
    Finalizes job by persisting output data to a database

    Parameters
    ----------
    workflow_step_name
        Name of the workflow step to log data for
    workflow_name
        Name of the workflow
    job_finish_res
        Unfortunately cannot return Dict from `task.sensor` so we need
        to construct Dict from string here.
        The string needs to encode:
            - well_known_file_type_path_map
                Maps well known file type name to path
            - storage_directory
                Root dir where all files for this job run are written
            - log_path
                Path where logs for this run are saved
            - start
                When did this step of the workflow run start. Uses encoding
                %Y-%m-%d %H:%M:%S
            - end
                When did this step of the workflow run end. Uses encoding
                %Y-%m-%d %H:%M:%S
    additional_steps
        See `ophys_etl.workflows.db.db_utils.save_job_run_to_db for details
    additional_steps_kwargs
        Kwargs to send to `additional_steps`

    Returns
    -------
    Mapping between well known file type and `OutputFile` of all output files
    """
    job_finish_res = json.loads(job_finish_res)
    start = datetime.datetime.strptime(
        job_finish_res["start"], "%Y-%m-%d %H:%M:%S%z"
    )
    end = datetime.datetime.strptime(
        job_finish_res["end"], "%Y-%m-%d %H:%M:%S%z"
    )

    ophys_experiment_id = context["params"].get("ophys_experiment_id", None)
    ophys_session_id = context["params"].get("ophys_session_id", None)
    ophys_container_id = context["params"].get("ophys_container_id", None)

    module_outputs = job_finish_res["module_outputs"]

    module_outputs = [
        OutputFile(
            path=Path(x["path"]),
            # serialize to WellKnownFileTypeEnum
            well_known_file_type=getattr(
                WellKnownFileTypeEnum,
                # Unfortunately airflow deserializes Enum to "<class>.<value>"
                # split off the value
                x["well_known_file_type"].split(".")[-1],
            ),
        )
        for x in module_outputs
    ]

    with Session(engine) as session:
        _save_job_run_to_db(
            workflow_name=workflow_name,
            workflow_step_name=workflow_step_name,
            start=start,
            end=end,
            module_outputs=module_outputs,
            ophys_experiment_id=ophys_experiment_id,
            ophys_session_id=ophys_session_id,
            ophys_container_id=ophys_container_id,
            sqlalchemy_session=session,
            storage_directory=job_finish_res["storage_directory"],
            log_path=job_finish_res["log_path"],
            additional_steps=additional_steps,
            additional_steps_kwargs=additional_steps_kwargs,
        )
    return {
        x.well_known_file_type.value: {
            'path': str(x.path),
            'well_known_file_type': x.well_known_file_type.value
        }
        for x in module_outputs
    }


def wait_for_dag_to_finish(
    dag_id: str,
    workflow_step: WorkflowStepEnum,
    level: Literal['experiment', 'session', 'container'] = None,
    timeout: float = app_config.job_timeout
) -> Callable:
    """
    Returns function which waits for `dag_id` to finish

    Notes
    ------
    Wrapping a `task.sensor` so that we can pass in a custom timeout

    Parameters
    ----------
    dag_id
        dag_id to wait for
    timeout
        Timeout in seconds
    workflow_step
        WorkflowStepEnum associated with `dag_id`
    level
        What level the dag runs at. 'experiment', 'session', or 'container'
        None if it doesn't apply

    Returns
    -------
    Callable decorated with `task.sensor`
    """

    @task.sensor(
        mode="reschedule",
        timeout=timeout,
        task_id=f'wait_for_{dag_id}_to_finish')
    def wait_for_dag_to_finish():
        context = get_current_context()
        ophys_experiment_id = context['params']['ophys_experiment_id']
        ophys_experiment = OphysExperiment.from_id(id=ophys_experiment_id)

        if dag_id == 'decrosstalk':
            if not ophys_experiment.is_multiplane:
                logger.info(
                    f'Experiment {ophys_experiment.id} is not multiplane. '
                    f'Equipment type: {ophys_experiment.equipment_name}. '
                    f'Decrosstalk does not need to run')
                return PokeReturnValue(is_done=True)

        if level == 'experiment':
            kwargs = {
                'ophys_experiment_id': ophys_experiment.id
            }
        elif level == 'session':
            kwargs = {
                'ophys_session_id': ophys_experiment.session.id
            }
        elif level == 'container':
            kwargs = {
                'ophys_container_id': ophys_experiment.container.id
            }
        else:
            if level is not None:
                raise ValueError(f'Invalid level {level}. Only "experiment", '
                                 f'"session", "container" supported')
            else:
                kwargs = {}

        with Session(engine) as session:
            try:
                get_latest_workflow_step_run(
                    session=session,
                    workflow_step=workflow_step,
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    **kwargs
                )
                is_done = True
            except NoResultFound:
                is_done = False

        # Checking if `dag_id` is running to prevent a situation where
        # `dag_id` was rerun. We don't want to pull the old `dag_id`
        # values. So we are waiting for the running dag to finish
        is_running = get_latest_dag_run(
            dag_id=dag_id,
            states=['running']
        ) is not None

        return PokeReturnValue(is_done=is_done and not is_running)

    return wait_for_dag_to_finish

"""Airflow tasks"""

import datetime
import json
from pathlib import Path
from typing import Dict, Optional, Callable

from airflow.decorators import task

from ophys_etl.workflows.app_config.app_config import app_config
from sqlmodel import Session

from ophys_etl.workflows.db import engine
from ophys_etl.workflows.db.db_utils import save_job_run_to_db as \
    _save_job_run_to_db
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileType
from ophys_etl.workflows.workflow_names import WorkflowName
from ophys_etl.workflows.workflow_steps import WorkflowStep


@task
def save_job_run_to_db(
        workflow_name: WorkflowName,
        workflow_step_name: WorkflowStep,
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
    start = datetime.datetime.strptime(job_finish_res['start'],
                                       '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(job_finish_res['end'],
                                     '%Y-%m-%d %H:%M:%S')

    ophys_experiment_id = context['params'].get('ophys_experiment_id', None)

    module_outputs = job_finish_res['module_outputs']

    module_outputs = [
        OutputFile(
            path=Path(x['path']),
            # serialize to WellKnownFileType
            well_known_file_type=getattr(
                WellKnownFileType,
                # Unfortunately airflow deserializes Enum to "<class>.<value>"
                # split off the value
                x['well_known_file_type'].split('.')[-1])
        ) for x in module_outputs]

    with Session(engine) as session:
        _save_job_run_to_db(
            workflow_name=workflow_name,
            workflow_step_name=workflow_step_name,
            start=start,
            end=end,
            module_outputs=module_outputs,
            ophys_experiment_id=ophys_experiment_id,
            sqlalchemy_session=session,
            storage_directory=job_finish_res['storage_directory'],
            log_path=job_finish_res['log_path'],
            validate_files_exist=not app_config.is_debug,
            additional_steps=additional_steps,
            additional_steps_kwargs=additional_steps_kwargs
        )
    return {
        x.well_known_file_type.value: x for x in module_outputs
    }

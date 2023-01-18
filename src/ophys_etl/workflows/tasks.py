"""Airflow tasks"""

import datetime
import json
from pathlib import Path
from typing import Dict

from airflow.decorators import task
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from sqlmodel import Session

from ophys_etl.workflows.db.db_utils import save_job_run_to_db as \
    _save_job_run_to_db
from ophys_etl.workflows.pipeline_module import OutputFile


@task
def save_job_run_to_db(
        workflow_step_name: str,
        job_finish_res: str,
        **context
) -> Dict[str, OutputFile]:
    """
    Finalizes job by persisting output data to a database

    Parameters
    ----------
    workflow_step_name
        Name of the workflow step to log data for
    job_finish_res
        Unfortunately cannot return Dict from `task.sensor` so we need
        to construct Dict from string here.
        The string needs to encode:
            - well_known_file_type_path_map
                Maps well known file type name to path
            - storage_directory
                Root dir where all files for this job run are written
            - start
                When did this step of the workflow run start. Uses encoding
                %Y-%m-%d %H:%M:%S
            - end
                When did this step of the workflow run end. Uses encoding
                %Y-%m-%d %H:%M:%S

    Returns
    -------
    Mapping between well known file type and `OutputFile` of all output files
    """
    job_finish_res = json.loads(job_finish_res)
    start = datetime.datetime.strptime(job_finish_res['start'],
                                       '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(job_finish_res['end'],
                                     '%Y-%m-%d %H:%M:%S')

    ophys_experiment_id = context['params']['ophys_experiment_id']

    module_outputs = job_finish_res['module_outputs']
    module_outputs = [
        OutputFile(
            path=Path(x['path']),
            well_known_file_type=x['well_known_file_type']
        ) for x in module_outputs]

    hook = SqliteHook(sqlite_conn_id='ophys_workflow_db')
    engine = hook.get_sqlalchemy_engine()

    with Session(engine) as session:
        _save_job_run_to_db(
            workflow_step_name=workflow_step_name,
            start=start,
            end=end,
            module_outputs=module_outputs,
            ophys_experiment_id=ophys_experiment_id,
            sqlalchemy_session=session,
            storage_directory=job_finish_res['storage_directory'],
            validate_files_exist=not context['params']['debug']
        )
    return {
        x.well_known_file_type: x for x in module_outputs
    }

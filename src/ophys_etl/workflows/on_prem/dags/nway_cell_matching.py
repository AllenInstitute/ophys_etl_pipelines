import datetime

import pendulum
from airflow.decorators import task_group
from airflow.models import Param
from airflow.models.dag import dag
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.on_prem.dags._misc import INT_PARAM_DEFAULT_VALUE
from ophys_etl.workflows.pipeline_modules.nway_cell_matching import \
    NwayCellMatchingModule


from ophys_etl.workflows.on_prem.workflow_utils import run_workflow_step
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


@dag(
    dag_id="nway_cell_matching",
    schedule=None,
    catchup=False,
    start_date=pendulum.yesterday(),
    params={
        "ophys_container_id": Param(
            description="identifier for ophys container",
            type="integer",
            default=INT_PARAM_DEFAULT_VALUE
        )
    }
)
def nway_cell_matching():
    """Runs nway cell matching step"""

    @task_group
    def run_nway_cell_matching():
        run_workflow_step(
            module=NwayCellMatchingModule,
            docker_tag=app_config.pipeline_steps.nway_cell_matching.docker_tag,
            workflow_step_name=WorkflowStepEnum.NWAY_CELL_MATCHING,
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            additional_db_inserts=(
                NwayCellMatchingModule.save_matches_to_db),
            slurm_config=(app_config.pipeline_steps.nway_cell_matching.
                          slurm_settings)
        )
    run_nway_cell_matching()


nway_cell_matching()

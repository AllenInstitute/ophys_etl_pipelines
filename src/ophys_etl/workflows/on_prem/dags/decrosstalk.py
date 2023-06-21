import datetime

from airflow.decorators import task_group
from airflow.models import Param
from airflow.models.dag import dag
from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.on_prem.dags._misc import INT_PARAM_DEFAULT_VALUE

from ophys_etl.workflows.on_prem.workflow_utils import run_workflow_step
from ophys_etl.workflows.pipeline_modules.decrosstalk import DecrosstalkModule
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


@dag(
    dag_id="decrosstalk",
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now(),
    params={
        "ophys_session_id": Param(
            description="identifier for ophys session",
            type="integer",
            default=INT_PARAM_DEFAULT_VALUE
        )
    }
)
def decrosstalk():
    """Runs decrosstalk step"""

    @task_group
    def run_decrosstalk():
        run_workflow_step(
            module=DecrosstalkModule,
            workflow_step_name=WorkflowStepEnum.DECROSSTALK,
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            additional_db_inserts=(
                DecrosstalkModule.save_decrosstalk_flags_to_db),
        )
    run_decrosstalk()


decrosstalk()

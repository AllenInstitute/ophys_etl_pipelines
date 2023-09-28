from typing import Optional

from airflow.decorators import task
from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.db import engine
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import \
    get_well_known_file_for_latest_run
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


@task
def get_denoised_movie_for_experiment(
    experiment_id: Optional[str] = None,
    **context
):
    if experiment_id is None:
        experiment_id = context['params']['ophys_experiment_id']

    denoised_ophys_movie_file = get_well_known_file_for_latest_run(
        engine=engine,
        well_known_file_type=(
            WellKnownFileTypeEnum.DEEPINTERPOLATION_DENOISED_MOVIE
        ),
        workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
        workflow_step=WorkflowStepEnum.DENOISING_INFERENCE,
        ophys_experiment_id=experiment_id,
    )
    return {
        'path': str(denoised_ophys_movie_file),
        'well_known_file_type': (
            WellKnownFileTypeEnum.DEEPINTERPOLATION_DENOISED_MOVIE.value)
    }


@task
def get_rois_for_experiment(
    experiment_id: Optional[str] = None,
    **context
):
    if experiment_id is None:
        experiment_id = context['params']['ophys_experiment_id']

    exp = OphysExperiment.from_id(id=experiment_id)
    return [x.to_dict() for x in exp.rois]
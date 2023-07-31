import datetime
from typing import Dict

from airflow.decorators import task_group, task
from airflow.models import Param
from airflow.models.dag import dag
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.on_prem.dags._misc import INT_PARAM_DEFAULT_VALUE

from ophys_etl.workflows.on_prem.dags.cell_classification.utils import \
    get_denoised_movie_for_experiment, get_rois_for_experiment
from ophys_etl.workflows.pipeline_modules import roi_classification

from ophys_etl.workflows.on_prem.workflow_utils import run_workflow_step
from sqlmodel import Session, select

from ophys_etl.workflows.db import engine
from ophys_etl.workflows.db.schemas import ROIClassifierEnsemble
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import \
    get_latest_workflow_step_run, get_well_known_file_for_latest_run
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


@task
def _get_roi_classifier() -> int:
    with Session(engine) as session:
        roi_classifier_training_run = get_latest_workflow_step_run(
            session=session,
            workflow_name=WorkflowNameEnum.ROI_CLASSIFIER_TRAINING,
            workflow_step=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING,
        )
        ensemble_id = session.exec(
            select(ROIClassifierEnsemble.id).where(
                ROIClassifierEnsemble.workflow_step_run_id
                == roi_classifier_training_run
            )
        ).one()
        return ensemble_id


@task
def _get_motion_correction_shifts_file(**context) -> Dict:
    motion_correction_shifts_path = get_well_known_file_for_latest_run(
        engine=engine,
        workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
        workflow_step=WorkflowStepEnum.MOTION_CORRECTION,
        well_known_file_type=WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA,
        ophys_experiment_id=context['params']['ophys_experiment_id']
    )
    return {
        'path': str(motion_correction_shifts_path),
        'well_known_file_type': (
            WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA.value)
    }


@dag(
    dag_id="cell_classifier_inference",
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now(),
    params={
        "ophys_experiment_id": Param(
            description="identifier for ophys experiment",
            type="integer",
            default=INT_PARAM_DEFAULT_VALUE
        )
    }
)
def cell_classifier_inference():
    """Classify ROIs output by segmentation as cells"""

    @task_group
    def correlation_projection_generation(denoised_ophys_movie_file):
        module_outputs = run_workflow_step(
            slurm_config=(app_config.pipeline_steps.roi_classification.
                          generate_correlation_projection.slurm_settings),
            module=roi_classification.GenerateCorrelationProjectionModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH # noqa E501
            ),
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            module_kwargs={
                "denoised_ophys_movie_file": denoised_ophys_movie_file
            },
        )
        return module_outputs[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value # noqa E501
        ]

    @task_group
    def generate_thumbnails(rois, correlation_graph_file):
        motion_correction_shifts_file = _get_motion_correction_shifts_file()

        module_outputs = run_workflow_step(
            slurm_config=(app_config.pipeline_steps.roi_classification.
                          generate_thumbnails.slurm_settings),
            module=roi_classification.GenerateThumbnailsModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS
            ),
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            module_kwargs={
                "denoised_ophys_movie_file": denoised_ophys_movie_file,
                "rois": rois,
                "correlation_projection_graph_file": correlation_graph_file, # noqa E501
                "is_training": False,
                "motion_correction_shifts_file": motion_correction_shifts_file
            },
        )
        return module_outputs[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_THUMBNAIL_IMAGES.value
        ]

    @task_group
    def run_inference(thumbnails_dir):
        ensemble_id = _get_roi_classifier()
        run_workflow_step(
            module=roi_classification.InferenceModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE
            ),
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            module_kwargs={
                "ensemble_id": ensemble_id,
                "thumbnails_dir": thumbnails_dir
            },
            slurm_config=(app_config.pipeline_steps.roi_classification.
                          inference.slurm_settings)
        )

    denoised_ophys_movie_file = get_denoised_movie_for_experiment()

    correlation_graph_file = correlation_projection_generation(
        denoised_ophys_movie_file=denoised_ophys_movie_file
    )

    rois = get_rois_for_experiment()
    thumbnail_dir = generate_thumbnails(
        rois=rois,
        correlation_graph_file=correlation_graph_file
    )
    thumbnail_dir >> run_inference(thumbnails_dir=thumbnail_dir)


cell_classifier_inference()

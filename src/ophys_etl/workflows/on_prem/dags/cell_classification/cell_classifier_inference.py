import datetime

from airflow.decorators import task_group
from airflow.models import Param
from airflow.models.dag import dag

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.on_prem.dags.cell_classification.utils import \
    get_denoised_movie_for_experiment, get_rois_for_experiment
from ophys_etl.workflows.pipeline_modules import roi_classification

from ophys_etl.workflows.on_prem.workflow_utils import run_workflow_step
from sqlmodel import Session, select

from ophys_etl.workflows.db import engine
from ophys_etl.workflows.db.schemas import ROIClassifierEnsemble
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_latest_workflow_step_run
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


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


@dag(
    dag_id="cell_classifier_inference",
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now(),
    params={
        "ophys_experiment_id": Param(
            description="identifier for ophys experiment", default=None
        )
    }
)
def cell_classifier_inference():
    """Classify ROIs output by segmentation as cells"""

    @task_group
    def correlation_projection_generation(denoised_ophys_movie_file):
        module_outputs = run_workflow_step(
            slurm_config_filename="correlation_projection.yml",
            module=roi_classification.GenerateCorrelationProjectionModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH # noqa E501
            ),
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            docker_tag=(
                app_config.pipeline_steps.roi_classification.generate_correlation_projection.docker_tag # noqa E501
            ),
            module_kwargs={
                "denoised_ophys_movie_file": denoised_ophys_movie_file
            },
        )
        return module_outputs[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value # noqa E501
        ]

    @task_group
    def generate_thumbnails(rois_file, correlation_graph_file):
        module_outputs = run_workflow_step(
            slurm_config_filename="correlation_projection.yml",
            module=roi_classification.GenerateThumbnailsModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS
            ),
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            docker_tag=(
                app_config.pipeline_steps.roi_classification.generate_thumbnails.docker_tag # noqa E501
            ),
            module_kwargs={
                "denoised_ophys_movie_file": denoised_ophys_movie_file,
                "rois_file": rois_file,
                "correlation_projection_graph_file": correlation_graph_file, # noqa E501
            },
        )
        return module_outputs[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_THUMBNAIL_IMAGES.value
        ]

    @task_group
    def run_inference():
        ensemble_id = _get_roi_classifier()
        run_workflow_step(
            module=roi_classification.InferenceModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE
            ),
            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            docker_tag=(
                app_config.pipeline_steps.roi_classification.inference.docker_tag # noqa E501
            ),
            module_kwargs={"ensemble_id": ensemble_id},
        )

    denoised_ophys_movie_file = get_denoised_movie_for_experiment()

    correlation_graph_file = correlation_projection_generation(
        denoised_ophys_movie_file=denoised_ophys_movie_file
    )

    rois_file = get_rois_for_experiment()
    thumbnail_dir = generate_thumbnails(
        rois_file=rois_file,
        correlation_graph_file=correlation_graph_file
    )
    thumbnail_dir >> run_inference()


cell_classifier_inference()

"""Ophys processing DAG"""
import datetime
from pathlib import Path
from typing import Type, Optional, Dict, Any, Callable

from airflow.decorators import task_group
from airflow.models import Param
from airflow.models.dag import dag

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.db.db_utils import enable_fk_if_sqlite
from ophys_etl.workflows.on_prem.tasks import submit_job, \
    wait_for_job_to_finish
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.pipeline_modules import roi_classification
from ophys_etl.workflows.pipeline_modules.denoising.denoising_finetuning \
    import \
    DenoisingFinetuningModule
from ophys_etl.workflows.pipeline_modules.denoising.denoising_inference \
    import \
    DenoisingInferenceModule
from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule
from ophys_etl.workflows.pipeline_modules.segmentation import \
    SegmentationModule
from ophys_etl.workflows.tasks import save_job_run_to_db
from ophys_etl.workflows.well_known_file_types import WellKnownFileType
from ophys_etl.workflows.workflow_steps import WorkflowStep


def _run_workflow_step(
    slurm_config_filename: str,
    module: Type[PipelineModule],
    workflow_step_name: WorkflowStep,
    docker_tag: str,
    module_kwargs: Optional[Dict] = None,
    additional_db_inserts: Optional[Callable] = None
) -> Any:
    """
    Runs a single workflow step

    Parameters
    ----------
    slurm_config_filename
        Slurm settings filename
    module
        What module to run
    workflow_step_name
        Workflow step name
    docker_tag
        What docker tag to use
    module_kwargs
        kwargs to send to module

    Returns
    -------
    Dictionary mapping WellKnownFileType to OutputFile, but actually an
    airflow.models.XCom until pulled in a task
    """
    slurm_config = (Path(__file__).parent.parent / 'slurm' / 'configs' /
                    slurm_config_filename)
    job_submit_res = submit_job(
        module=module,
        config_path=str(slurm_config),
        docker_tag=docker_tag,
        module_kwargs=module_kwargs
    )

    job_finish_res = wait_for_job_to_finish(
        timeout=app_config.job_timeout
    )(
        job_id=job_submit_res['job_id'],
        storage_directory=job_submit_res['storage_directory'],
        module_outputs=(
            job_submit_res['module_outputs'])
    )
    run = save_job_run_to_db(
        workflow_step_name=workflow_step_name,
        job_finish_res=job_finish_res,
        additional_steps=additional_db_inserts
    )

    return run['output_files']


@dag(
    dag_id='ophys_processing',
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now(),
    # For required arguments, need to pass default of None due to
    # https://github.com/apache/airflow/issues/28940
    # This makes the argument optional. Unfortunately it is impossible to
    # make it required.
    # TODO once fixed, remove default
    params={
        'ophys_experiment_id': Param(
            description='identifier for ophys experiment',
            default=None
        ),
        'prevent_file_overwrites': Param(
            description='If True, will fail job run if a file output by '
                        'module already exists',
            default=True
        )
    }
)
def ophys_processing():
    @task_group
    def motion_correction():
        """Motion correct raw ophys movie"""
        module_outputs = _run_workflow_step(
            slurm_config_filename='motion_correction.yml',
            module=MotionCorrectionModule,
            workflow_step_name=WorkflowStep.MOTION_CORRECTION,
            docker_tag=app_config.pipeline_steps.motion_correction.docker_tag,
            additional_db_inserts=MotionCorrectionModule.save_metadata_to_db
        )
        return module_outputs[
            WellKnownFileType.MOTION_CORRECTED_IMAGE_STACK.value]

    @task_group
    def denoising(
        motion_corrected_ophys_movie_file
    ):
        @task_group
        def denoising_finetuning(motion_corrected_ophys_movie_file):
            """Finetune deepinterpolation model on a single ophys movie"""
            module_outputs = _run_workflow_step(
                slurm_config_filename='denoising_finetuning.yml',
                module=DenoisingFinetuningModule,
                workflow_step_name=WorkflowStep.DENOISING_FINETUNING,
                docker_tag=app_config.pipeline_steps.denoising.docker_tag,
                module_kwargs={
                    'motion_corrected_ophys_movie_file':
                        motion_corrected_ophys_movie_file
                }
            )
            return module_outputs[
                WellKnownFileType.DEEPINTERPOLATION_FINETUNED_MODEL.value]

        @task_group
        def denoising_inference(
                motion_corrected_ophys_movie_file,
                trained_denoising_model_file
        ):
            """Runs denoising inference on a single ophys movie"""
            module_outputs = _run_workflow_step(
                slurm_config_filename='denoising_inference.yml',
                module=DenoisingInferenceModule,
                workflow_step_name=WorkflowStep.DENOISING_INFERENCE,
                docker_tag=app_config.pipeline_steps.denoising.docker_tag,
                module_kwargs={
                    'motion_corrected_ophys_movie_file':
                        motion_corrected_ophys_movie_file,
                    'trained_denoising_model_file':
                        trained_denoising_model_file
                }
            )
            return module_outputs[
                WellKnownFileType.DEEPINTERPOLATION_DENOISED_MOVIE.value]

        trained_denoising_model_file = denoising_finetuning(
            motion_corrected_ophys_movie_file=(
                motion_corrected_ophys_movie_file))
        denoised_movie = denoising_inference(
            motion_corrected_ophys_movie_file=(
                motion_corrected_ophys_movie_file),
            trained_denoising_model_file=trained_denoising_model_file)
        return denoised_movie

    @task_group
    def segmentation(denoised_ophys_movie_file):
        module_outputs = _run_workflow_step(
            slurm_config_filename='segmentation.yml',
            module=SegmentationModule,
            workflow_step_name=WorkflowStep.SEGMENTATION,
            docker_tag=app_config.pipeline_steps.segmentation.docker_tag,
            additional_db_inserts=SegmentationModule.save_rois_to_db,
            module_kwargs={
                'denoised_ophys_movie_file':
                    denoised_ophys_movie_file
            }
        )
        return module_outputs[
            WellKnownFileType.OPHYS_ROIS.value]

    @task_group
    def classify_rois(
        denoised_ophys_movie_file,
        rois_file
    ):
        @task_group
        def correlation_projection_generation():
            module_outputs = _run_workflow_step(
                slurm_config_filename='correlation_projection.yml',
                module=roi_classification.GenerateCorrelationProjectionModule,
                workflow_step_name=(
                    WorkflowStep.
                    ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH),
                docker_tag=(app_config.pipeline_steps.roi_classification.
                            generate_correlation_projection.docker_tag),
                module_kwargs={
                    'denoised_ophys_movie_file':
                        denoised_ophys_movie_file
                }
            )
            return module_outputs[
                WellKnownFileType.
                ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value]

        @task_group
        def generate_thumbnails(
            correlation_graph_file
        ):
            module_outputs = _run_workflow_step(
                slurm_config_filename='correlation_projection.yml',
                module=roi_classification.GenerateThumbnailsModule,
                workflow_step_name=(
                    WorkflowStep.ROI_CLASSIFICATION_GENERATE_THUMBNAILS),
                docker_tag=(app_config.pipeline_steps.roi_classification.
                            generate_thumbnails.docker_tag),
                module_kwargs={
                    'denoised_ophys_movie_file':
                        denoised_ophys_movie_file,
                    'rois_file': rois_file,
                    'correlation_projection_graph_file': correlation_graph_file
                }
            )
            return module_outputs[
                WellKnownFileType.ROI_CLASSIFICATION_THUMBNAIL_IMAGES]

        correlation_graph_file = correlation_projection_generation(
            denoised_ophys_movie_file=denoised_ophys_movie_file)
        generate_thumbnails(
            correlation_graph_file=correlation_graph_file)

    motion_corrected_ophys_movie_file = motion_correction()
    denoised_movie_file = denoising(
        motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file)
    rois_file = segmentation(denoised_ophys_movie_file=denoised_movie_file)
    classify_rois(
        denoised_ophys_movie_file=denoised_movie_file,
        rois_file=rois_file
    )


enable_fk_if_sqlite()

ophys_processing()

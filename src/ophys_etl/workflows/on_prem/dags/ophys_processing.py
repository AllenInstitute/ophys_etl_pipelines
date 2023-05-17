"""Ophys processing DAG"""
import datetime

from airflow.decorators import task_group
from airflow.models import Param
from airflow.models.dag import dag

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.on_prem.workflow_utils import run_workflow_step
from ophys_etl.workflows.pipeline_modules.denoising.denoising_finetuning import (  # noqa E501
    DenoisingFinetuningModule,
)
from ophys_etl.workflows.pipeline_modules.denoising.denoising_inference import (  # noqa E501
    DenoisingInferenceModule,
)
from ophys_etl.workflows.pipeline_modules.motion_correction import (
    MotionCorrectionModule,
)
from ophys_etl.workflows.pipeline_modules.segmentation import (
    SegmentationModule,
)
from ophys_etl.workflows.pipeline_modules.trace_processing.trace_extraction import (  # noqa E501
    TraceExtractionModule,
)
from ophys_etl.workflows.pipeline_modules.trace_processing.demix_traces import (  # noqa E501
    DemixTracesModule,
)
from ophys_etl.workflows.tasks import wait_for_decrosstalk_to_finish
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

WORKFLOW_NAME = WorkflowNameEnum.OPHYS_PROCESSING


@dag(
    dag_id="ophys_processing",
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now(),
    # For required arguments, need to pass default of None due to
    # https://github.com/apache/airflow/issues/28940
    # This makes the argument optional. Unfortunately it is impossible to
    # make it required.
    # TODO once fixed, remove default
    params={
        "ophys_experiment_id": Param(
            description="identifier for ophys experiment", default=None
        ),
        "prevent_file_overwrites": Param(
            description="If True, will fail job run if a file output by "
            "module already exists",
            default=True,
        ),
    },
)
def ophys_processing():
    @task_group
    def motion_correction():
        """Motion correct raw ophys movie"""
        module_outputs = run_workflow_step(
            slurm_config_filename="motion_correction.yml",
            module=MotionCorrectionModule,
            workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
            workflow_name=WORKFLOW_NAME,
            docker_tag=app_config.pipeline_steps.motion_correction.docker_tag,
            additional_db_inserts=MotionCorrectionModule.save_metadata_to_db,
        )
        return module_outputs[
            WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK.value
        ]

    @task_group
    def denoising(motion_corrected_ophys_movie_file):
        @task_group
        def denoising_finetuning(motion_corrected_ophys_movie_file):
            """Finetune deepinterpolation model on a single ophys movie"""
            module_outputs = run_workflow_step(
                slurm_config_filename="denoising_finetuning.yml",
                module=DenoisingFinetuningModule,
                workflow_step_name=WorkflowStepEnum.DENOISING_FINETUNING,
                workflow_name=WORKFLOW_NAME,
                docker_tag=app_config.pipeline_steps.denoising.docker_tag,
                module_kwargs={
                    "motion_corrected_ophys_movie_file": motion_corrected_ophys_movie_file # noqa E501
                },
            )
            return module_outputs[
                WellKnownFileTypeEnum.DEEPINTERPOLATION_FINETUNED_MODEL.value
            ]

        @task_group
        def denoising_inference(
            motion_corrected_ophys_movie_file, trained_denoising_model_file
        ):
            """Runs denoising inference on a single ophys movie"""
            module_outputs = run_workflow_step(
                slurm_config_filename="denoising_inference.yml",
                module=DenoisingInferenceModule,
                workflow_step_name=WorkflowStepEnum.DENOISING_INFERENCE,
                workflow_name=WORKFLOW_NAME,
                docker_tag=app_config.pipeline_steps.denoising.docker_tag,
                module_kwargs={
                    "motion_corrected_ophys_movie_file": motion_corrected_ophys_movie_file, # noqa E501
                    "trained_denoising_model_file": trained_denoising_model_file, # noqa E501
                },
            )
            return module_outputs[
                WellKnownFileTypeEnum.DEEPINTERPOLATION_DENOISED_MOVIE.value
            ]

        trained_denoising_model_file = denoising_finetuning(
            motion_corrected_ophys_movie_file=(
                motion_corrected_ophys_movie_file
            )
        )
        denoised_movie = denoising_inference(
            motion_corrected_ophys_movie_file=(
                motion_corrected_ophys_movie_file
            ),
            trained_denoising_model_file=trained_denoising_model_file,
        )
        return denoised_movie

    @task_group
    def segmentation(denoised_ophys_movie_file):
        module_outputs = run_workflow_step(
            slurm_config_filename="segmentation.yml",
            module=SegmentationModule,
            workflow_step_name=WorkflowStepEnum.SEGMENTATION,
            workflow_name=WORKFLOW_NAME,
            docker_tag=app_config.pipeline_steps.segmentation.docker_tag,
            additional_db_inserts=SegmentationModule.save_rois_to_db,
            module_kwargs={
                "denoised_ophys_movie_file": denoised_ophys_movie_file
            },
        )
        return module_outputs[WellKnownFileTypeEnum.OPHYS_ROIS.value]

    @task_group
    def trace_processing(motion_corrected_ophys_movie_file,
                         rois_file):
        @task_group
        def trace_extraction(motion_corrected_ophys_movie_file,
                             rois_file):
            module_outputs = run_workflow_step(
                slurm_config_filename="trace_extraction.yml",
                module=TraceExtractionModule,
                workflow_step_name=WorkflowStepEnum.TRACE_EXTRACTION,
                workflow_name=WORKFLOW_NAME,
                docker_tag=app_config.pipeline_steps.trace_extraction.docker_tag,  # noqa E501
                module_kwargs={
                    "motion_corrected_ophys_movie_file": motion_corrected_ophys_movie_file,  # noqa E501
                    "rois_file": rois_file
                },
            )

            return module_outputs[WellKnownFileTypeEnum.ROI_TRACE.value]

        @task_group
        def demix_traces(motion_corrected_ophys_movie_file, roi_traces_file):
            run_workflow_step(
                slurm_config_filename="demix_traces.yml",
                module=DemixTracesModule,
                workflow_step_name=WorkflowStepEnum.DEMIX_TRACES,
                workflow_name=WORKFLOW_NAME,
                docker_tag=app_config.pipeline_steps.demix_traces.docker_tag,
                module_kwargs={
                    "motion_corrected_ophys_movie_file": motion_corrected_ophys_movie_file,  # noqa E501
                    "roi_traces_file": roi_traces_file
                },
                pre_submit_sensor=wait_for_decrosstalk_to_finish(
                    timeout=app_config.job_timeout)
            )

        roi_traces_file = trace_extraction(
            motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file,  # noqa E501
            rois_file=rois_file
        )
        demix_traces(
            motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file,  # noqa E501
            roi_traces_file=roi_traces_file
        )

    motion_corrected_ophys_movie_file = motion_correction()
    denoised_movie_file = denoising(
        motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file
    )
    rois_file = segmentation(denoised_ophys_movie_file=denoised_movie_file)
    trace_processing(
        motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file,
        rois_file=rois_file)


ophys_processing()

"""Ophys processing DAG"""
import datetime

from airflow.decorators import task_group, task
from airflow.models import Param
from airflow.models.dag import dag
from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysContainer, OphysSession

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.on_prem.dags._misc import INT_PARAM_DEFAULT_VALUE
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
from ophys_etl.workflows.pipeline_modules.trace_processing.neuropil_correction import (  # noqa E501
    NeuropilCorrection
)
from ophys_etl.workflows.pipeline_modules.trace_processing.dff_calculation import (  # noqa E501
    DFOverFCalculation,
)
from ophys_etl.workflows.pipeline_modules.trace_processing.event_detection import (  # noqa E501
    EventDetection,
)
from ophys_etl.workflows.tasks import wait_for_decrosstalk_to_finish
from ophys_etl.workflows.utils.dag_utils import trigger_dag_run
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_most_recent_run
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

WORKFLOW_NAME = WorkflowNameEnum.OPHYS_PROCESSING


@dag(
    dag_id="ophys_processing",
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now(),
    params={
        "ophys_experiment_id": Param(
            description="identifier for ophys experiment",
            type="integer",
            default=INT_PARAM_DEFAULT_VALUE
        ),
        "prevent_file_overwrites": Param(
            description="If True, will fail job run if a file output by "
            "module already exists",
            default=True
        ),
        "run_cell_classification": Param(
            description='Whether to run cell classification',
            default=True,
            type="boolean"
        )
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
        run_workflow_step(
            slurm_config_filename="segmentation.yml",
            module=SegmentationModule,
            workflow_step_name=WorkflowStepEnum.SEGMENTATION,
            workflow_name=WORKFLOW_NAME,
            additional_db_inserts=SegmentationModule.save_rois_to_db,
            module_kwargs={
                "denoised_ophys_movie_file": denoised_ophys_movie_file
            },
        )

    @task
    def check_run_cell_classification(**context):
        return context['params']['run_cell_classification']

    @task
    def run_cell_classification(do_run: bool, **context):
        if do_run:
            trigger_dag_run(
                conf={
                    'ophys_experiment_id':
                        context['params']['ophys_experiment_id']
                },
                context=context,
                task_id='run_cell_classification',
                trigger_dag_id='cell_classifier_inference'
            )

    @task
    def check_run_decrosstalk(**context):
        ophys_experiment = OphysExperiment.from_id(
            id=context['params']['ophys_experiment_id'])
        # avoiding a race condition. Only want to return true for a single
        # ophys experiment within session
        is_most_recent = ophys_experiment.id == get_most_recent_run(
            workflow_step=WorkflowStepEnum.SEGMENTATION,
            ophys_experiment_ids=(
                OphysSession.from_id(id=ophys_experiment.session.id)
                .get_ophys_experiment_ids())
        )

        is_session_complete = \
            ophys_experiment.session.has_completed_workflow_step(
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )

        return ophys_experiment.is_multiplane and \
            is_session_complete and \
            is_most_recent

    @task
    def run_decrosstalk(do_run: bool, **context):
        if do_run:
            ophys_experiment = OphysExperiment.from_id(
                id=context['params']['ophys_experiment_id'])
            trigger_dag_run(
                conf={'ophys_session_id': ophys_experiment.session.id},
                context=context,
                task_id='trigger_decrosstalk_for_ophys_session',
                trigger_dag_id='decrosstalk'
            )

    @task
    def check_run_nway_cell_matching(**context):
        ophys_experiment = OphysExperiment.from_id(
            id=context['params']['ophys_experiment_id'])

        # some experiments are not assigned to a container
        if ophys_experiment.container.id is None:
            return False

        # avoiding a race condition. Only want to return true for a single
        # ophys experiment within container
        is_most_recent = ophys_experiment.id == get_most_recent_run(
            workflow_step=WorkflowStepEnum.SEGMENTATION,
            ophys_experiment_ids=(
                OphysContainer.from_id(ophys_experiment.container.id)
                .get_ophys_experiment_ids()
            )
        )

        is_container_complete = \
            ophys_experiment.container.has_completed_workflow_step(
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )

        return is_container_complete and is_most_recent

    @task
    def run_nway_cell_matching(do_run: bool, **context):
        if do_run:
            ophys_experiment = OphysExperiment.from_id(
                id=context['params']['ophys_experiment_id'])
            trigger_dag_run(
                conf={'ophys_container_id': ophys_experiment.container.id},
                context=context,
                task_id='trigger_nway_cell_matching_for_ophys_container',
                trigger_dag_id='nway_cell_matching'
            )

    @task_group
    def trace_processing(motion_corrected_ophys_movie_file):
        @task_group
        def trace_extraction(motion_corrected_ophys_movie_file):
            module_outputs = run_workflow_step(
                module=TraceExtractionModule,
                workflow_step_name=WorkflowStepEnum.TRACE_EXTRACTION,
                workflow_name=WORKFLOW_NAME,
                module_kwargs={
                    "motion_corrected_ophys_movie_file": motion_corrected_ophys_movie_file,  # noqa E501
                }
            )

            return module_outputs

        @task_group
        def demix_traces(motion_corrected_ophys_movie_file, roi_traces_file):
            module_outputs = run_workflow_step(
                slurm_config_filename="demix_traces.yml",
                module=DemixTracesModule,
                workflow_step_name=WorkflowStepEnum.DEMIX_TRACES,
                workflow_name=WORKFLOW_NAME,
                module_kwargs={
                    "motion_corrected_ophys_movie_file": motion_corrected_ophys_movie_file,  # noqa E501
                    "roi_traces_file": roi_traces_file
                },
                pre_submit_sensor=wait_for_decrosstalk_to_finish(
                    timeout=app_config.job_timeout)
            )
            return module_outputs[WellKnownFileTypeEnum.DEMIXED_TRACES.value]

        @task_group
        def neuropil_correction(
                demixed_roi_traces_file,
                neuropil_traces_file):
            module_outputs = run_workflow_step(
                slurm_config_filename="neuropil_correction.yml",
                module=NeuropilCorrection,
                workflow_step_name=WorkflowStepEnum.NEUROPIL_CORRECTION,
                workflow_name=WORKFLOW_NAME,
                module_kwargs={
                    "demixed_roi_traces_file": demixed_roi_traces_file,
                    "neuropil_traces_file": neuropil_traces_file
                }
            )
            return module_outputs[
                WellKnownFileTypeEnum.NEUROPIL_CORRECTED_TRACES.value]

        @task_group
        def dff(neuropil_corrected_traces):
            module_outputs = run_workflow_step(
                slurm_config_filename="dff.yml",
                module=DFOverFCalculation,
                workflow_step_name=WorkflowStepEnum.DFF,
                workflow_name=WORKFLOW_NAME,
                module_kwargs={
                    "neuropil_corrected_traces": neuropil_corrected_traces,
                }
            )
            return module_outputs[WellKnownFileTypeEnum.DFF_TRACES.value]

        @task_group
        def event_detection(dff_traces):
            run_workflow_step(
                slurm_config_filename="event_detection.yml",
                module=EventDetection,
                workflow_step_name=WorkflowStepEnum.EVENT_DETECTION,
                workflow_name=WORKFLOW_NAME,
                module_kwargs={
                    "dff_traces": dff_traces,
                }
            )

        trace_outputs = trace_extraction(
            motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file,  # noqa E501
        )
        demixed_traces = demix_traces(
            motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file,  # noqa E501
            roi_traces_file=trace_outputs[
                WellKnownFileTypeEnum.ROI_TRACE.value]
        )
        neuropil_corrected_traces = neuropil_correction(
            demixed_roi_traces_file=demixed_traces,
            neuropil_traces_file=trace_outputs[
                WellKnownFileTypeEnum.NEUROPIL_TRACE.value]
        )
        dff_traces = dff(neuropil_corrected_traces=neuropil_corrected_traces)
        event_detection(dff_traces=dff_traces)

    motion_corrected_ophys_movie_file = motion_correction()
    denoised_movie_file = denoising(
        motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file
    )
    segmentation_run = segmentation(
        denoised_ophys_movie_file=denoised_movie_file)

    do_run_cell_classification = check_run_cell_classification()
    do_run_decrosstalk = check_run_decrosstalk()
    do_run_nway_cell_matching = check_run_nway_cell_matching()
    segmentation_run >> [do_run_cell_classification,
                         do_run_decrosstalk,
                         do_run_nway_cell_matching]

    run_cell_classification(do_run=do_run_cell_classification)
    run_decrosstalk(do_run=do_run_decrosstalk)
    run_nway_cell_matching(do_run=do_run_nway_cell_matching)
    segmentation_run >> trace_processing(
        motion_corrected_ophys_movie_file=motion_corrected_ophys_movie_file)


ophys_processing()

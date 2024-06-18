"""ROI classifier training DAG"""
import json
import time
from typing import Dict, List, Any

from airflow.decorators import task, task_group
from airflow.models import XCom
from airflow.models.dag import dag
from deepcell.cli.modules.create_dataset import construct_dataset
from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.on_prem.dags.cell_classification.utils import \
    get_denoised_movie_for_experiment, get_rois_for_experiment
from ophys_etl.workflows.on_prem.workflow_utils import (
    run_workflow_step,
    submit_job_and_wait_to_finish,
)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules import roi_classification
from ophys_etl.workflows.pipeline_modules.roi_classification.utils.model_utils import ( # noqa E501
    download_trained_model,
)
from ophys_etl.workflows.tasks import save_job_run_to_db
from ophys_etl.workflows.utils.dag_utils import START_DATE
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

WORKFLOW_NAME = WorkflowNameEnum.ROI_CLASSIFIER_TRAINING


@task
def _download_trained_model(job_finish_res: str, mlflow_run_name: str):
    """Downloads trained model from s3 to local disk"""
    job_finish_res = json.loads(job_finish_res)
    trained_model_dest: OutputFile = job_finish_res["module_outputs"][
        WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL
    ]

    download_trained_model(
        mlflow_run_name=mlflow_run_name, model_dest=trained_model_dest
    )


@task
def _get_labeled_experiment_ids():
    labels = construct_dataset(
        cell_labeling_app_host=(
            app_config.pipeline_steps.roi_classification.cell_labeling_app_host
        ),
        vote_tallying_strategy=(
            app_config.pipeline_steps.roi_classification.training
            .voting_strategy
        )
    )
    experiment_ids = labels["experiment_id"].astype(int).tolist()
    return experiment_ids


@task
def _create_thumbnails_for_experiments(
    correlation_projection_graphs: Dict[str, str],
    experiment_ids: List[str]
):
    thumbnail_dirs = {}
    for experiment_id in experiment_ids:
        denoised_ophys_movie_file = get_denoised_movie_for_experiment(
            experiment_id=experiment_id)
        rois_file = get_rois_for_experiment(experiment_id=experiment_id)
        module_outputs = run_workflow_step(
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS
            ),
            module=roi_classification.GenerateThumbnailsModule,
            workflow_name=WORKFLOW_NAME,
            module_kwargs={
                "is_training": True,
                "denoised_ophys_movie_file": denoised_ophys_movie_file,
                "rois_file": rois_file,
                "correlation_projection_graph_file": (
                    correlation_projection_graphs[experiment_id]
                ),
            },
            slurm_config=(app_config.pipeline_steps.roi_classification.
                          generate_thumbnails.slurm_settings)
        )
        thumbnail_dirs[experiment_id] = module_outputs[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_THUMBNAIL_IMAGES.value
        ]


@task
def _generate_correlation_projections_for_experiments(
        experiment_ids: List[str]):
    runs = {}
    for experiment_id in experiment_ids:
        denoised_ophys_movie_file = get_denoised_movie_for_experiment(
            experiment_id=experiment_id)
        module_outputs = run_workflow_step(
            slurm_config=(app_config.pipeline_steps.roi_classification.
                          generate_correlation_projection.slurm_settings),
            module=roi_classification.GenerateCorrelationProjectionModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH   # noqa E402
            ),
            workflow_name=WORKFLOW_NAME,
            module_kwargs={
                "denoised_ophys_movie_file": denoised_ophys_movie_file
            },
        )
        runs[experiment_id] = module_outputs[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value   # noqa E402
        ]
        return runs


def _get_roi_meta_for_experiment_ids(
    experiment_ids: List[int]
) -> Dict[str, Dict[str, Dict[Any, Any]]]:
    """

    Parameters
    ----------
    experiment_ids

    Returns
    -------
    Map from experiment id to dict
        keys: roi id
        values:
            - is_inside_motion_border: whether roi is inside the motion border
    """
    exp_rois_meta = {}
    for exp_id in experiment_ids:
        oe = OphysExperiment.from_id(id=int(exp_id))
        roi_meta = {}
        for roi in oe.rois:
            roi_meta[str(roi.id)] = {
                'is_inside_motion_border': not roi.is_in_motion_border
            }
        exp_rois_meta[str(exp_id)] = roi_meta
    return exp_rois_meta


@task(task_id='_get_roi_meta_for_experiment_ids')
def _get_roi_meta_for_experiment_ids_task(
    experiment_ids: List[int]
):
    return _get_roi_meta_for_experiment_ids(
        experiment_ids=experiment_ids
    )


@dag(
    dag_id="roi_classifier_training",
    schedule=None,
    catchup=False,
    start_date=START_DATE,
)
def roi_classifier_training():
    @task_group
    def generate_correlation_projections(experiment_ids):
        """Create correlation projections for all experiments in training
        set"""
        runs = _generate_correlation_projections_for_experiments(
            experiment_ids=experiment_ids)

        return runs

    @task_group
    def create_thumbnails(
        correlation_projection_graphs: Dict[str, XCom],
        experiment_ids
    ):
        """Create training thumbnails for all ROIs in training set"""
        thumbnail_dirs = _create_thumbnails_for_experiments(
            correlation_projection_graphs=correlation_projection_graphs,
            experiment_ids=experiment_ids)

        return thumbnail_dirs

    @task_group
    def create_train_test_split(
        thumbnail_dirs,
        experiment_ids
    ):
        """Create train/test split"""
        exp_roi_meta_map = _get_roi_meta_for_experiment_ids_task(
            experiment_ids=experiment_ids
        )
        module_outputs = run_workflow_step(
            module=roi_classification.CreateTrainTestSplitModule,
            workflow_step_name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT
            ),
            workflow_name=WORKFLOW_NAME,
            module_kwargs={
                "thumbnail_dirs": thumbnail_dirs,
                "exp_roi_meta_map": exp_roi_meta_map
            },
            slurm_config=(app_config.pipeline_steps.roi_classification.
                          training.slurm_settings)
        )
        return module_outputs[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAIN_SET.value
        ]

    @task_group
    def train_model(train_set_path):
        """Trains model on AWS"""
        mlflow_run_name = f"CV-{int(time.time())}"
        job_finish_res = submit_job_and_wait_to_finish(
            module=roi_classification.TrainingModule,
            module_kwargs={
                "train_set_path": train_set_path,
                "mlflow_run_name": mlflow_run_name,
            },
            slurm_config=(app_config.pipeline_steps.roi_classification.
                          training.slurm_settings)
        )
        _download_trained_model(
            job_finish_res=job_finish_res, mlflow_run_name=mlflow_run_name
        )

        save_job_run_to_db(
            workflow_name=WORKFLOW_NAME,
            workflow_step_name=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING,
            job_finish_res=job_finish_res,
            additional_steps=(
                roi_classification.TrainingModule.save_trained_model_to_db
            ),
            additional_steps_kwargs={
                "mlflow_parent_run_name": mlflow_run_name
            },
        )

    experiment_ids = _get_labeled_experiment_ids()
    correlation_projection_graphs = generate_correlation_projections(
        experiment_ids=experiment_ids)
    thumbnail_dirs = create_thumbnails(
        correlation_projection_graphs=correlation_projection_graphs,
        experiment_ids=experiment_ids
    )
    train_set_path = create_train_test_split(
        thumbnail_dirs=thumbnail_dirs,
        experiment_ids=experiment_ids
    )
    train_model(train_set_path=train_set_path)


roi_classifier_training()

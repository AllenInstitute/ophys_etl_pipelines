"""ROI classifier training DAG"""
import datetime
import json
import time
from typing import Dict

from airflow.decorators import task_group, task
from airflow.models import XCom
from airflow.models.dag import dag
from deepcell.cli.modules.create_dataset import construct_dataset
from ophys_etl.workflows.tasks import save_job_run_to_db

from ophys_etl.workflows.pipeline_module import OutputFile

from ophys_etl.workflows.db import engine
from ophys_etl.workflows.pipeline_modules import roi_classification
from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .model_utils \
    import \
    download_trained_model
from ophys_etl.workflows.well_known_file_types import WellKnownFileType

from ophys_etl.workflows.workflow_names import WorkflowName
from ophys_etl.workflows.workflow_step_runs import \
    get_well_known_file_for_latest_run
from ophys_etl.workflows.workflow_steps import WorkflowStep

from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.on_prem.workflow_utils import run_workflow_step, \
    submit_job_and_wait_to_finish

WORKFLOW_NAME = WorkflowName.ROI_CLASSIFIER_TRAINING


@task
def _download_trained_model(
    job_finish_res: str,
    mlflow_run_name: str
):
    """Downloads trained model from s3 to local disk"""
    job_finish_res = json.loads(job_finish_res)
    trained_model_dest: OutputFile = (
        job_finish_res['module_outputs']
        [WellKnownFileType.ROI_CLASSIFICATION_TRAINED_MODEL])

    download_trained_model(
        mlflow_run_name=mlflow_run_name,
        model_dest=trained_model_dest
    )


@dag(
    dag_id='roi_classifier_training',
    schedule=None,
    catchup=False,
    start_date=datetime.datetime.now()
)
def roi_classifier_training():
    @task_group
    def generate_correlation_projections():
        """Create correlation projections for all experiments in training
        set"""
        labels = construct_dataset(
            cell_labeling_app_host=(
                app_config.pipeline_steps.roi_classification.
                cell_labeling_app_host)
        )
        experiment_ids = labels['experiment_id'].tolist()
        runs = {}
        for experiment_id in experiment_ids:
            denoised_ophys_movie_file = get_well_known_file_for_latest_run(
                engine=engine,
                well_known_file_type=(
                    WellKnownFileType.DEEPINTERPOLATION_DENOISED_MOVIE),
                workflow_name=WORKFLOW_NAME,
                workflow_step=WorkflowStep.DENOISING_INFERENCE,
                ophys_experiment_id=experiment_id
            )

            module_outputs = run_workflow_step(
                slurm_config_filename='correlation_projection.yml',
                module=roi_classification.GenerateCorrelationProjectionModule,
                workflow_step_name=(
                    WorkflowStep.
                    ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH),
                workflow_name=WORKFLOW_NAME,
                docker_tag=(app_config.pipeline_steps.roi_classification.
                            generate_correlation_projection.docker_tag),
                module_kwargs={
                    'denoised_ophys_movie_file':
                        denoised_ophys_movie_file
                }
            )
            runs[experiment_id] = (
                module_outputs[
                    WellKnownFileType.
                    ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH])
        return runs

    @task_group
    def create_thumbnails(correlation_projection_graphs: Dict[str, XCom]):
        """Create training thumbnails for all ROIs in training set"""
        labels = construct_dataset(
            cell_labeling_app_host=(
                app_config.pipeline_steps.roi_classification.
                cell_labeling_app_host)
        )
        thumbnail_dirs = {}
        experiment_ids = labels['experiment_id'].tolist()
        for experiment_id in experiment_ids:
            denoised_ophys_movie_file = get_well_known_file_for_latest_run(
                engine=engine,
                well_known_file_type=(
                    WellKnownFileType.DEEPINTERPOLATION_DENOISED_MOVIE),
                workflow_name=WORKFLOW_NAME,
                workflow_step=WorkflowStep.DENOISING_INFERENCE,
                ophys_experiment_id=experiment_id
            )
            rois_file = get_well_known_file_for_latest_run(
                engine=engine,
                well_known_file_type=WellKnownFileType.OPHYS_ROIS,
                workflow_name=WORKFLOW_NAME,
                workflow_step=WorkflowStep.SEGMENTATION,
                ophys_experiment_id=experiment_id
            )
            module_outputs = run_workflow_step(
                workflow_step_name=(
                    WorkflowStep.ROI_CLASSIFICATION_GENERATE_THUMBNAILS),
                module=roi_classification.GenerateThumbnailsModule,
                workflow_name=WORKFLOW_NAME,
                experiment_id=experiment_id,
                module_kwargs={
                    'is_training': True,
                    'denoised_ophys_movie_file': denoised_ophys_movie_file,
                    'rois_file': rois_file,
                    'correlation_projection_graph_file': (
                        correlation_projection_graphs[experiment_id])
                }
            )
            thumbnail_dirs[experiment_id] = \
                module_outputs[
                    WellKnownFileType.ROI_CLASSIFICATION_THUMBNAIL_IMAGES]
        return thumbnail_dirs

    @task_group
    def create_train_test_split(thumbnail_dirs):
        """Create train/test split"""
        module_outputs = run_workflow_step(
            module=roi_classification.CreateTrainTestSplitModule,
            workflow_step_name=(
                WorkflowStep.ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT),
            workflow_name=WORKFLOW_NAME,
            module_kwargs={
                'thumbnail_dirs': thumbnail_dirs
            }
        )
        return module_outputs[WellKnownFileType.ROI_CLASSIFICATION_TRAIN_SET]

    @task_group
    def train_model(train_set_path):
        """Trains model on AWS"""
        mlflow_run_name = f'CV-{int(time.time())}'
        job_finish_res = submit_job_and_wait_to_finish(
            module=roi_classification.TrainingModule,
            module_kwargs={
                'train_set_path': train_set_path,
                'mlflow_run_name': mlflow_run_name
            }
        )
        _download_trained_model(
            job_finish_res=job_finish_res,
            mlflow_run_name=mlflow_run_name
        )

        save_job_run_to_db(
            workflow_name=WORKFLOW_NAME,
            workflow_step_name=WorkflowStep.ROI_CLASSIFICATION_TRAINING,
            job_finish_res=job_finish_res,
            additional_steps=(
                roi_classification.TrainingModule.save_trained_model_to_db),
            additional_steps_kwargs={
                'mlflow_parent_run_name': mlflow_run_name
            }

        )

    correlation_projection_graphs = generate_correlation_projections()
    thumbnail_dirs = \
        create_thumbnails(
            correlation_projection_graphs=correlation_projection_graphs)
    train_set_path = create_train_test_split(thumbnail_dirs=thumbnail_dirs)
    train_model(train_set_path=train_set_path)
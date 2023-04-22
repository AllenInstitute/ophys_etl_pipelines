"""Script to create and populate tables"""
import datetime
from pathlib import Path
from typing import Dict, Optional

import argschema
import marshmallow
from ophys_etl.workflows.pipeline_modules import roi_classification

from ophys_etl.workflows.pipeline_module import OutputFile

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from sqlmodel import SQLModel, Session

from ophys_etl.workflows.db import get_engine
from ophys_etl.workflows.db.schemas import Workflow, WorkflowStep, \
    WellKnownFileType
from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .model_utils import \
    download_trained_model
from ophys_etl.workflows.well_known_file_types import WellKnownFileType as \
    WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowName
from ophys_etl.workflows.workflow_steps import WorkflowStep as WorkflowStepEnum


class _TrainedROIClassifierSchema(argschema.ArgSchema):
    """In production, the trained ROI classifier should be saved through the
    `roi_classifier_training` workflow. However, in development, we can add
    an roi classifier training run manually by specifying values for this
    schema"""
    mlflow_parent_run_name = argschema.fields.String(
        required=True,
        description='MLFlow parent run name for the training run'
    )
    trained_model_dest = argschema.fields.OutputDir(
        required=True,
        description='Where to save trained model'
    )


class InitializeDBSchema(argschema.ArgSchema):
    db_url = argschema.fields.String(
        required=True,
        description='db url, see '
                    'https://docs.sqlalchemy.org/en/20/core/engines.html'
    )
    add_roi_classifier_training_run = argschema.fields.Boolean(
        default=False,
        description='Whether to add roi classifier training run. Needed for '
                    'development when testing out `ophys_processing` workflow'
    )
    roi_classifier_args = argschema.fields.Nested(
        _TrainedROIClassifierSchema,
        default=None,
        allow_none=True
    )

    @marshmallow.pre_load
    def validate_roi_classifier_args(self, data: dict, **kwargs):
        if data['add_roi_classifier_training_run']:
            if data['roi_classifier_args'] is None:
                raise ValueError(
                    'If `add_roi_classifier_training_run`, then '
                    'this is a dev database. Specify `roi_classifier_args` '
                    'to manually add an roi classifier training run')
        return data


def create_db_and_tables(engine):
    """Creates db and empty tables"""
    SQLModel.metadata.create_all(engine)


def _create_workflows(session):
    ophys_processing = _create_workflow(
        session=session,
        workflow=Workflow(name=WorkflowName.OPHYS_PROCESSING.value))
    roi_classifier_training = _create_workflow(
        session=session,
        workflow=Workflow(name=WorkflowName.ROI_CLASSIFIER_TRAINING.value))
    return {
        WorkflowName.OPHYS_PROCESSING: ophys_processing,
        WorkflowName.ROI_CLASSIFIER_TRAINING: roi_classifier_training
    }


def _create_workflow(session, workflow: Workflow):
    """Adds Workflow"""
    session.add(workflow)
    session.commit()
    return workflow


def _create_workflow_steps_for_ophys_processing(
        session,
        workflow: Workflow
):
    """Adds workflow steps for ophys processing"""
    workflow_steps = {
        'motion_correction': WorkflowStep(
            name=WorkflowStepEnum.MOTION_CORRECTION.value,
            workflow_id=workflow.id
        ),
        'denoising_finetuning': WorkflowStep(
            name=WorkflowStepEnum.DENOISING_FINETUNING.value,
            workflow_id=workflow.id
        ),
        'denoising_inference': WorkflowStep(
            name=WorkflowStepEnum.DENOISING_INFERENCE.value,
            workflow_id=workflow.id
        ),
        'segmentation': WorkflowStep(
            name=WorkflowStepEnum.SEGMENTATION.value,
            workflow_id=workflow.id
        ),
        'roi_classification_generate_correlation_projection': WorkflowStep(
            name=(WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH.value),    # noqa E402
            workflow_id=workflow.id
        ),
        'roi_classification_generate_thumbnails': WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS.value,
            workflow_id=workflow.id
        ),
        'roi_classification_inference': WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE.value,
            workflow_id=workflow.id
        )
    }
    for workflow_step in workflow_steps.values():
        session.add(workflow_step)
    session.commit()
    return workflow_steps


def _create_workflow_steps_for_roi_classifier_training(
        session,
        workflow: Workflow
):
    """Adds workflow steps for roi classifier training"""
    workflow_steps = {
        'roi_classification_generate_correlation_projection': WorkflowStep(
            name=(WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH.value),    # noqa E402
            workflow_id=workflow.id
        ),
        'roi_classification_generate_thumbnails': WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS.value,
            workflow_id=workflow.id
        ),
        'ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT': WorkflowStep(
            name=WorkflowStepEnum.
            ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT.value,
            workflow_id=workflow.id
        ),
        'ROI_CLASSIFICATION_TRAINING': WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING.value,
            workflow_id=workflow.id
        )
    }
    for workflow_step in workflow_steps.values():
        session.add(workflow_step)
    session.commit()
    return workflow_steps


def _create_motion_correction_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.MAX_INTENSITY_PROJECTION_IMAGE.value,
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.AVG_INTENSITY_PROJECTION_IMAGE.value,
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.REGISTRATION_SUMMARY_IMAGE.value,
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                  .value),
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA.value,
            workflow_step_id=workflow_steps['motion_correction'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.MOTION_PREVIEW.value,
            workflow_step_id=workflow_steps['motion_correction'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_denoising_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.DEEPINTERPOLATION_FINETUNED_MODEL.value,
            workflow_step_id=workflow_steps['denoising_finetuning'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.DEEPINTERPOLATION_DENOISED_MOVIE.value,
            workflow_step_id=workflow_steps['denoising_inference'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_segmentation_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.OPHYS_ROIS.value,
            workflow_step_id=workflow_steps['segmentation'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_roi_classification_inference_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=(WellKnownFileTypeEnum
                  .ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value),
            workflow_step_id=(workflow_steps['roi_classification_generate_correlation_projection'].id)  # noqa E402
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum
                  .ROI_CLASSIFICATION_THUMBNAIL_IMAGES.value),
            workflow_step_id=workflow_steps[
                'roi_classification_generate_thumbnails'].id
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum.
                  ROI_CLASSIFICATION_EXPERIMENT_PREDICTIONS.value),
            workflow_step_id=workflow_steps['roi_classification_inference'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_roi_classification_training_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=(WellKnownFileTypeEnum
                  .ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value),
            workflow_step_id=(workflow_steps['roi_classification_generate_correlation_projection'].id)  # noqa E402
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum
                  .ROI_CLASSIFICATION_THUMBNAIL_IMAGES.value),
            workflow_step_id=workflow_steps[
                'roi_classification_generate_thumbnails'].id
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum
                  .ROI_CLASSIFICATION_TRAIN_SET.value),
            workflow_step_id=workflow_steps[
                'ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT'].id
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum
                  .ROI_CLASSIFICATION_TEST_SET.value),
            workflow_step_id=workflow_steps[
                'ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT'].id
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum
                  .ROI_CLASSIFICATION_TRAINED_MODEL.value),
            workflow_step_id=workflow_steps[
                'ROI_CLASSIFICATION_TRAINING'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_well_known_file_types_for_ophys_processing(
        session,
        workflow_steps: Dict[str, WorkflowStep]
):
    """Adds well known file types for ophys processing"""
    _create_motion_correction_well_known_file_types(
        session=session,
        workflow_steps=workflow_steps
    )
    _create_denoising_well_known_file_types(
        session=session,
        workflow_steps=workflow_steps
    )
    _create_segmentation_well_known_file_types(
        session=session,
        workflow_steps=workflow_steps
    )
    _create_roi_classification_inference_well_known_file_types(
        session=session,
        workflow_steps=workflow_steps
    )
    session.commit()


def _create_well_known_file_types_for_roi_classifier_training(
        session,
        workflow_steps: Dict[str, WorkflowStep]
):
    """Adds well known file types for roi classifier training"""
    _create_roi_classification_training_well_known_file_types(
        session=session,
        workflow_steps=workflow_steps
    )
    session.commit()


def _add_roi_classifier_training_run(
    session: Session,
    mlflow_parent_run_name: str,
    trained_model_dest: OutputFile,
    logger
):
    """Manually adds roi classifier training run

    Parameters
    ----------
    mlflow_parent_run_name
        MLFlow parent run name for the training run
    trained_model_dest
        Where to save trained model
    logger
        Logger
    """
    logger.info(f'Downloading trained roi classifier model to '
                f'{trained_model_dest.path}')
    download_trained_model(
        mlflow_run_name=mlflow_parent_run_name,
        model_dest=trained_model_dest
    )
    save_job_run_to_db(
        workflow_name=WorkflowName.ROI_CLASSIFIER_TRAINING,
        workflow_step_name=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING,
        start=datetime.datetime.now(),
        end=datetime.datetime.now(),
        module_outputs=[trained_model_dest],
        sqlalchemy_session=session,
        storage_directory=trained_model_dest.path.parent,
        log_path='',
        validate_files_exist=True,
        additional_steps=(
            roi_classification.TrainingModule.save_trained_model_to_db),
        additional_steps_kwargs={
            'mlflow_parent_run_name': mlflow_parent_run_name
        }
    )


def _populate_db(
    engine,
    logger,
    add_roi_classifier_training_run: bool = False,
    mlflow_parent_run_name: Optional[str] = None,
    trained_model_dest: Optional[OutputFile] = None
):
    """Populates tables

    Parameters
    ----------
    engine:
        Sqlalchemy engine
    logger:
        Logger
    add_roi_classifier_training_run
        Whether to add roi classifier training run (needed for development)
    mlflow_parent_run_name
        MLFlow parent run name for the training run. Only needed if
        `add_roi_classifier_training_run`
    trained_model_dest
        Where to save trained model. Only needed if
        `add_roi_classifier_training_run`

    """
    if add_roi_classifier_training_run:
        if mlflow_parent_run_name is None:
            raise ValueError('Specify mlflow_parent_run_name if '
                             'add_roi_classifier_training_run')
        if trained_model_dest is None:
            raise ValueError('Specify trained_model_dest if '
                             'add_roi_classifier_training_run')
    with Session(engine) as session:
        workflows = _create_workflows(session=session)

        # 1. create ophys processing workflow steps and well known file types
        ophys_processing_workflow_steps = \
            _create_workflow_steps_for_ophys_processing(
                session=session,
                workflow=workflows[WorkflowName.OPHYS_PROCESSING])
        _create_well_known_file_types_for_ophys_processing(
            session=session,
            workflow_steps=ophys_processing_workflow_steps)

        # 2. create roi classifier training workflow steps and well known
        # file types
        roi_classifier_training_workflow_steps = \
            _create_workflow_steps_for_roi_classifier_training(
                session=session,
                workflow=workflows[WorkflowName.ROI_CLASSIFIER_TRAINING])
        _create_well_known_file_types_for_roi_classifier_training(
            session=session,
            workflow_steps=roi_classifier_training_workflow_steps
        )

        if add_roi_classifier_training_run:
            _add_roi_classifier_training_run(
                session=session,
                mlflow_parent_run_name=mlflow_parent_run_name,
                trained_model_dest=trained_model_dest,
                logger=logger
            )


class IntializeDBRunner(argschema.ArgSchemaParser):
    default_schema = InitializeDBSchema

    def run(self):
        engine = get_engine(db_conn=self.args['db_url'])
        create_db_and_tables(engine)
        if self.args['roi_classifier_args'] is None:
            roi_classifier_args = {}
        else:
            roi_classifier_args = self.args['roi_classifier_args']

        _populate_db(
            engine,
            add_roi_classifier_training_run=(
                self.args['add_roi_classifier_training_run']),
            mlflow_parent_run_name=(
                roi_classifier_args.get('mlflow_parent_run_name')
            ),
            trained_model_dest=OutputFile(
                path=(
                    Path(roi_classifier_args.get('trained_model_dest'))
                    if roi_classifier_args else None),
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL)
            ),
            logger=self.logger
        )
        return engine


if __name__ == "__main__":
    init_db = IntializeDBRunner()
    init_db.run()

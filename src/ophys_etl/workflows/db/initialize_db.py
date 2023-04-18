"""Script to create and populate tables"""
from typing import Dict

import argschema
from sqlmodel import SQLModel, Session

from ophys_etl.workflows.db import get_engine
from ophys_etl.workflows.db.schemas import Workflow, WorkflowStep, \
    WellKnownFileType
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class InitializeDBSchema(argschema.ArgSchema):
    db_url = argschema.fields.String(
        required=True,
        description='db url'
    )


def create_db_and_tables(engine):
    """Creates db and empty tables"""
    SQLModel.metadata.create_all(engine)


def _create_workflows(session):
    ophys_processing = _create_workflow(
        session=session,
        workflow=Workflow(name=WorkflowNameEnum.OPHYS_PROCESSING.value))
    roi_classifier_training = _create_workflow(
        session=session,
        workflow=Workflow(name=WorkflowNameEnum.ROI_CLASSIFIER_TRAINING.value))
    return {
        WorkflowNameEnum.OPHYS_PROCESSING: ophys_processing,
        WorkflowNameEnum.ROI_CLASSIFIER_TRAINING: roi_classifier_training
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
        ),
        'trace_extraction': WorkflowStep(
            name=WorkflowStepEnum.TRACE_EXTRACTION.value,
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
    _create_trace_extraction_well_known_file_types(
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

def _create_trace_extraction_well_known_file_types(
    session,
    workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.NEUROPIL_TRACE.value,
            workflow_step_id=workflow_steps['trace_extraction'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.ROI_TRACE.value,
            workflow_step_id=workflow_steps['trace_extraction'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.NEUROPIL_MASK.value,
            workflow_step_id=workflow_steps['trace_extraction'].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.TRACE_EXTRACTION_EXCLUSION_LABELS.value,
            workflow_step_id=workflow_steps['trace_extraction'].id
        ),
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _populate_db(engine):
    """Populates tables"""
    with Session(engine) as session:
        workflows = _create_workflows(session=session)

        # 1. create ophys processing workflow steps and well known file types
        ophys_processing_workflow_steps = \
            _create_workflow_steps_for_ophys_processing(
                session=session,
                workflow=workflows[WorkflowNameEnum.OPHYS_PROCESSING])
        _create_well_known_file_types_for_ophys_processing(
            session=session,
            workflow_steps=ophys_processing_workflow_steps)

        # 2. create roi classifier training workflow steps and well known
        # file types
        roi_classifier_training_workflow_steps = \
            _create_workflow_steps_for_roi_classifier_training(
                session=session,
                workflow=workflows[WorkflowNameEnum.ROI_CLASSIFIER_TRAINING])
        _create_well_known_file_types_for_roi_classifier_training(
            session=session,
            workflow_steps=roi_classifier_training_workflow_steps
        )


class InitializeDBRunner(argschema.ArgSchemaParser):
    default_schema = InitializeDBSchema

    def run(self):
        engine = get_engine(db_conn=self.args['db_url'])
        create_db_and_tables(engine)
        _populate_db(engine)
        return engine


if __name__ == "__main__":
    init_db = InitializeDBRunner()
    init_db.run()

"""Script to create and populate tables"""
import datetime
import os
from pathlib import Path
from typing import Dict, Optional

import argschema
import marshmallow

from ophys_etl.workflows.app_config.app_config import app_config
from sqlmodel import Session, SQLModel

from ophys_etl.workflows.db import get_engine
from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.db.schemas import (
    WellKnownFileType,
    Workflow,
    WorkflowStep,
)
from ophys_etl.workflows.pipeline_module import OutputFile
from ophys_etl.workflows.pipeline_modules import roi_classification
from ophys_etl.workflows.pipeline_modules.roi_classification.utils.model_utils import ( # noqa E501
    download_trained_model,
)
from ophys_etl.workflows.well_known_file_types import (
    WellKnownFileTypeEnum as WellKnownFileTypeEnum,
)
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class _TrainedROIClassifierSchema(argschema.ArgSchema):
    """In production, the trained ROI classifier should be saved through the
    `roi_classifier_training` workflow. However, in development, we can add
    an roi classifier training run manually by specifying values for this
    schema"""

    trained_model_dest = argschema.fields.OutputDir(
        required=True, description="Where to save trained model"
    )


class InitializeDBSchema(argschema.ArgSchema):
    db_url = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="db url, see "
        "https://docs.sqlalchemy.org/en/20/core/engines.html. If not provided,"
                    "uses the value from app_config.app_db.conn_string",
    )
    add_roi_classifier_training_run = argschema.fields.Boolean(
        default=False,
        description="Whether to add roi classifier training run. Needed for "
        "development when testing out `ophys_processing` workflow",
    )
    roi_classifier_args = argschema.fields.Nested(
        _TrainedROIClassifierSchema, default=None, allow_none=True
    )

    @marshmallow.pre_load
    def validate_roi_classifier_args(self, data: dict, **kwargs):
        if data["add_roi_classifier_training_run"]:
            if data["roi_classifier_args"] is None:
                data["roi_classifier_args"] = {
                    "trained_model_dest": (
                            app_config.output_dir / "cell_classifier_model")
                }
        return data


def create_db_and_tables(engine):
    """Creates db and empty tables"""
    SQLModel.metadata.create_all(engine)


def _create_workflows(session):
    ophys_processing = _create_workflow(
        session=session,
        workflow=Workflow(name=WorkflowNameEnum.OPHYS_PROCESSING.value),
    )
    roi_classifier_training = _create_workflow(
        session=session,
        workflow=Workflow(name=WorkflowNameEnum.ROI_CLASSIFIER_TRAINING.value),
    )
    return {
        WorkflowNameEnum.OPHYS_PROCESSING: ophys_processing,
        WorkflowNameEnum.ROI_CLASSIFIER_TRAINING: roi_classifier_training,
    }


def _create_workflow(session, workflow: Workflow):
    """Adds Workflow"""
    session.add(workflow)
    session.commit()
    return workflow


def _create_workflow_steps_for_ophys_processing(session, workflow: Workflow):
    """Adds workflow steps for ophys processing"""
    workflow_steps = {
        "motion_correction": WorkflowStep(
            name=WorkflowStepEnum.MOTION_CORRECTION.value,
            workflow_id=workflow.id,
        ),
        "denoising_finetuning": WorkflowStep(
            name=WorkflowStepEnum.DENOISING_FINETUNING.value,
            workflow_id=workflow.id,
        ),
        "denoising_inference": WorkflowStep(
            name=WorkflowStepEnum.DENOISING_INFERENCE.value,
            workflow_id=workflow.id,
        ),
        "segmentation": WorkflowStep(
            name=WorkflowStepEnum.SEGMENTATION.value, workflow_id=workflow.id
        ),
        "roi_classification_generate_correlation_projection": WorkflowStep(
            name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH.value # noqa E501
            ),
            workflow_id=workflow.id,
        ),
        "roi_classification_generate_thumbnails": WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS.value,
            workflow_id=workflow.id,
        ),
        "roi_classification_inference": WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE.value,
            workflow_id=workflow.id,
        ),
        "trace_extraction": WorkflowStep(
            name=WorkflowStepEnum.TRACE_EXTRACTION.value,
            workflow_id=workflow.id,
        ),
        "demix_traces": WorkflowStep(
            name=WorkflowStepEnum.DEMIX_TRACES.value,
            workflow_id=workflow.id,
        ),
        "neuropil_correction": WorkflowStep(
            name=WorkflowStepEnum.NEUROPIL_CORRECTION.value,
            workflow_id=workflow.id
        ),
        "dff_calculation": WorkflowStep(
            name=WorkflowStepEnum.DFF.value,
            workflow_id=workflow.id
        ),
        "decrosstalk": WorkflowStep(
            name=WorkflowStepEnum.DECROSSTALK.value,
            workflow_id=workflow.id
        ),
        "event_detection": WorkflowStep(
            name=WorkflowStepEnum.EVENT_DETECTION.value,
            workflow_id=workflow.id,
        ),
        "nway_cell_matching": WorkflowStep(
            name=WorkflowStepEnum.NWAY_CELL_MATCHING.value,
            workflow_id=workflow.id
        )
    }
    for workflow_step in workflow_steps.values():
        session.add(workflow_step)
    session.commit()
    return workflow_steps


def _create_workflow_steps_for_roi_classifier_training(
    session, workflow: Workflow
):
    """Adds workflow steps for roi classifier training"""
    workflow_steps = {
        "roi_classification_generate_correlation_projection": WorkflowStep(
            name=(
                WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH.value # noqa E501
            ),
            workflow_id=workflow.id,
        ),
        "roi_classification_generate_thumbnails": WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS.value,
            workflow_id=workflow.id,
        ),
        "ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT": WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT.value, # noqa E501
            workflow_id=workflow.id,
        ),
        "ROI_CLASSIFICATION_TRAINING": WorkflowStep(
            name=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING.value,
            workflow_id=workflow.id,
        ),
    }
    for workflow_step in workflow_steps.values():
        session.add(workflow_step)
    session.commit()
    return workflow_steps


def _create_motion_correction_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.MAX_INTENSITY_PROJECTION_IMAGE.value,
            workflow_step_id=workflow_steps["motion_correction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.AVG_INTENSITY_PROJECTION_IMAGE.value,
            workflow_step_id=workflow_steps["motion_correction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.REGISTRATION_SUMMARY_IMAGE.value,
            workflow_step_id=workflow_steps["motion_correction"].id,
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK.value),
            workflow_step_id=workflow_steps["motion_correction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA.value,
            workflow_step_id=workflow_steps["motion_correction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.MOTION_PREVIEW.value,
            workflow_step_id=workflow_steps["motion_correction"].id,
        ),
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_denoising_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.DEEPINTERPOLATION_FINETUNED_MODEL.value,
            workflow_step_id=workflow_steps["denoising_finetuning"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.DEEPINTERPOLATION_DENOISED_MOVIE.value,
            workflow_step_id=workflow_steps["denoising_inference"].id,
        ),
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_segmentation_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.OPHYS_ROIS.value,
            workflow_step_id=workflow_steps["segmentation"].id,
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_roi_classification_inference_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=(
                WellKnownFileTypeEnum.ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value # noqa E501
            ),
            workflow_step_id=(
                workflow_steps[
                    "roi_classification_generate_correlation_projection"
                ].id
            ),
        ),
        WellKnownFileType(
            name=(
                WellKnownFileTypeEnum.ROI_CLASSIFICATION_THUMBNAIL_IMAGES.value
            ),
            workflow_step_id=workflow_steps[
                "roi_classification_generate_thumbnails"
            ].id,
        ),
        WellKnownFileType(
            name=(
                WellKnownFileTypeEnum.ROI_CLASSIFICATION_EXPERIMENT_PREDICTIONS.value # noqa E501
            ),
            workflow_step_id=workflow_steps["roi_classification_inference"].id,
        ),
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_roi_classification_training_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=(
                WellKnownFileTypeEnum.ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH.value # noqa E501
            ),
            workflow_step_id=(
                workflow_steps[
                    "roi_classification_generate_correlation_projection"
                ].id
            ),
        ),
        WellKnownFileType(
            name=(
                WellKnownFileTypeEnum.ROI_CLASSIFICATION_THUMBNAIL_IMAGES.value
            ),
            workflow_step_id=workflow_steps[
                "roi_classification_generate_thumbnails"
            ].id,
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAIN_SET.value),
            workflow_step_id=workflow_steps[
                "ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT"
            ].id,
        ),
        WellKnownFileType(
            name=(WellKnownFileTypeEnum.ROI_CLASSIFICATION_TEST_SET.value),
            workflow_step_id=workflow_steps[
                "ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT"
            ].id,
        ),
        WellKnownFileType(
            name=(
                WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL.value
            ),
            workflow_step_id=workflow_steps["ROI_CLASSIFICATION_TRAINING"].id,
        ),
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_nway_cell_matching_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.NWAY_CELL_MATCHING_METADATA.value,
            workflow_step_id=workflow_steps['nway_cell_matching'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_well_known_file_types_for_ophys_processing(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    """Adds well known file types for ophys processing"""
    _create_motion_correction_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    _create_denoising_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    _create_segmentation_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )

    _create_trace_extraction_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    _create_decrosstalk_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    _create_event_detection_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    _create_roi_classification_inference_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    _create_nway_cell_matching_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    session.commit()


def _create_well_known_file_types_for_roi_classifier_training(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    """Adds well known file types for roi classifier training"""
    _create_roi_classification_training_well_known_file_types(
        session=session, workflow_steps=workflow_steps
    )
    session.commit()


def _create_trace_extraction_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.NEUROPIL_TRACE.value,
            workflow_step_id=workflow_steps["trace_extraction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.ROI_TRACE.value,
            workflow_step_id=workflow_steps["trace_extraction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.NEUROPIL_MASK.value,
            workflow_step_id=workflow_steps["trace_extraction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.TRACE_EXTRACTION_EXCLUSION_LABELS.value,
            workflow_step_id=workflow_steps["trace_extraction"].id,
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.DEMIXED_TRACES.value,
            workflow_step_id=workflow_steps["demix_traces"].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.NEUROPIL_CORRECTED_TRACES.value,
            workflow_step_id=workflow_steps["neuropil_correction"].id
        ),
        WellKnownFileType(
            name=WellKnownFileTypeEnum.DFF_TRACES.value,
            workflow_step_id=workflow_steps["dff_calculation"].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_decrosstalk_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.DECROSSTALK_FLAGS.value,
            workflow_step_id=workflow_steps['decrosstalk'].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _create_event_detection_well_known_file_types(
    session, workflow_steps: Dict[str, WorkflowStep]
):
    well_known_file_types = [
        WellKnownFileType(
            name=WellKnownFileTypeEnum.EVENTS.value,
            workflow_step_id=workflow_steps["event_detection"].id
        )
    ]
    for wkft in well_known_file_types:
        session.add(wkft)


def _add_roi_classifier_training_run(
    session: Session,
    trained_model_dest: OutputFile,
    logger,
):
    """Manually adds roi classifier training run

    Parameters
    ----------
    trained_model_dest
        Where to save trained model
    logger
        Logger
    """
    logger.info(
        f"Downloading trained roi classifier model to "
        f"{trained_model_dest.path}"
    )
    if trained_model_dest.path.is_file():
        os.remove(trained_model_dest.path)

    mlflow_parent_run_name = (
            app_config.pipeline_steps.roi_classification.inference.
            mlflow_parent_run_name)
    download_trained_model(
        model_dest=trained_model_dest,
        mlflow_run_name=mlflow_parent_run_name
    )
    save_job_run_to_db(
        workflow_name=WorkflowNameEnum.ROI_CLASSIFIER_TRAINING,
        workflow_step_name=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING,
        start=datetime.datetime.now(),
        end=datetime.datetime.now(),
        module_outputs=[trained_model_dest],
        sqlalchemy_session=session,
        storage_directory=trained_model_dest.path.parent,
        log_path="",
        validate_files_exist=True,
        additional_steps=(
            roi_classification.TrainingModule.save_trained_model_to_db
        ),
        additional_steps_kwargs={
            "mlflow_parent_run_name": mlflow_parent_run_name
        },
    )


def _populate_db(
    engine,
    logger,
    add_roi_classifier_training_run: bool = False,
    trained_model_dest: Optional[OutputFile] = None,
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
    trained_model_dest
        Where to save trained model. Only needed if
        `add_roi_classifier_training_run`

    """
    if add_roi_classifier_training_run:
        if trained_model_dest is None:
            raise ValueError(
                "Specify trained_model_dest if "
                "add_roi_classifier_training_run"
            )
    with Session(engine) as session:
        workflows = _create_workflows(session=session)

        # 1. create ophys processing workflow steps and well known file types
        ophys_processing_workflow_steps = (
            _create_workflow_steps_for_ophys_processing(
                session=session,
                workflow=workflows[WorkflowNameEnum.OPHYS_PROCESSING],
            )
        )
        _create_well_known_file_types_for_ophys_processing(
            session=session, workflow_steps=ophys_processing_workflow_steps
        )

        # 2. create roi classifier training workflow steps and well known
        # file types
        roi_classifier_training_workflow_steps = (
            _create_workflow_steps_for_roi_classifier_training(
                session=session,
                workflow=workflows[WorkflowNameEnum.ROI_CLASSIFIER_TRAINING],
            )
        )
        _create_well_known_file_types_for_roi_classifier_training(
            session=session,
            workflow_steps=roi_classifier_training_workflow_steps,
        )

        if add_roi_classifier_training_run:
            _add_roi_classifier_training_run(
                session=session,
                trained_model_dest=trained_model_dest,
                logger=logger,
            )


class InitializeDBRunner(argschema.ArgSchemaParser):
    default_schema = InitializeDBSchema

    def run(self):
        if self.args["db_url"] is None:
            db_conn = app_config.app_db.conn_string
        else:
            db_conn = self.args["db_url"]
        engine = get_engine(db_conn=db_conn)
        create_db_and_tables(engine)
        if self.args["roi_classifier_args"] is None:
            roi_classifier_args = {}
        else:
            roi_classifier_args = self.args["roi_classifier_args"]

        _populate_db(
            engine,
            add_roi_classifier_training_run=(
                self.args["add_roi_classifier_training_run"]
            ),
            trained_model_dest=OutputFile(
                path=(
                    Path(roi_classifier_args.get("trained_model_dest"))
                    if roi_classifier_args
                    else None
                ),
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL
                ),
            ),
            logger=self.logger,
        )
        return engine


if __name__ == "__main__":
    init_db = InitializeDBRunner()
    init_db.run()

"""App database schemas"""
import datetime
from typing import Optional

from sqlalchemy import Column, Enum, UniqueConstraint
from sqlmodel import SQLModel, Field

from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class Workflow(SQLModel, table=True):
    __tablename__ = 'workflow'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(
        sa_column=Column(
            'name', Enum(WorkflowNameEnum),
            unique=True,
            nullable=False
        )
    )


class WorkflowStep(SQLModel, table=True):
    __tablename__ = 'workflow_step'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(
        sa_column=Column(
            'name', Enum(WorkflowStepEnum),
            nullable=False
        )
    )
    workflow_id: int = Field(foreign_key='workflow.id')

    __table_args__ = (
        # Makes the combination of name and workflow id unique,
        # since we don't want to allow a given workflow to contain the same
        # workflow step multiple times
        UniqueConstraint(
            'name', 'workflow_id', name='workflow_workflow_step_name_uc'), )


class WorkflowStepRun(SQLModel, table=True):
    __tablename__ = 'workflow_step_run'

    id: Optional[int] = Field(default=None, primary_key=True)
    ophys_experiment_id: Optional[str] = Field(index=True)
    workflow_step_id: int = Field(foreign_key='workflow_step.id')
    log_path: str
    storage_directory: str
    start: datetime.datetime
    end: datetime.datetime


class MotionCorrectionRun(SQLModel, table=True):
    """Motion correction run"""
    __tablename__ = 'motion_correction_run'

    workflow_step_run_id: int = Field(
        foreign_key='workflow_step_run.id',
        primary_key=True
    )
    max_correction_up: float
    max_correction_down: float
    max_correction_right: float
    max_correction_left: float


class OphysROI(SQLModel, table=True):
    """Ophys ROI"""
    __tablename__ = 'ophys_roi'

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_run_id: int = Field(
        index=True,
        foreign_key='workflow_step_run.id'
    )
    x: int
    y: int
    width: int
    height: int


class OphysROIMaskValue(SQLModel, table=True):
    """Stores a single value of an ROI mask as row, col"""
    __tablename__ = 'ophys_roi_mask_value'

    id: Optional[int] = Field(default=None, primary_key=True)
    ophys_roi_id: int = Field(
        foreign_key='ophys_roi.id'
    )
    row_index: int
    col_index: int


class ROIClassifierTrainingRun(SQLModel, table=True):
    """ROI classifier training run.
    Each row represents a single trained model. There can be multiple rows per
    workflow_step_run_id since we train an ensemble"""
    __tablename__ = 'roi_classifier_training_run'

    mlflow_run_id: str = Field(
        description='mlflow run id. MLFlow is used for tracking training '
                    'metadata',
        primary_key=True
    )
    ensemble_id: int = Field(
        foreign_key='roi_classifier_ensemble.id',
        index=True
    )
    sagemaker_job_id: str = Field(
        description='sagemaker job id. Model is trained on sagemaker.'
    )


class ROIClassifierEnsemble(SQLModel, table=True):
    """Each row is a single trained roi classifier ensemble"""
    __tablename__ = 'roi_classifier_ensemble'

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_run_id: int = Field(
        foreign_key='workflow_step_run.id'
    )
    mlflow_run_id: str = Field(
        description='mlflow parent run for the ensemble training run'
    )


class ROIClassifierInferenceResults(SQLModel, table=True):
    """Each row is classifier prediction for a single ROI"""

    __tablename__ = 'roi_classifier_inference_results'

    roi_id: int = Field(primary_key=True,
                        foreign_key='ophys_roi.id')
    ensemble_id: int = Field(
        primary_key=True,
        foreign_key='roi_classifier_ensemble.id'
    )
    score: float = Field(
        description='Classifier confidence that this ROI is a cell')


class WellKnownFileType(SQLModel, table=True):
    __tablename__ = 'well_known_file_type'

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_id: int = Field(foreign_key='workflow_step.id')
    name: str = Field(
        sa_column=Column(
            'name', Enum(WellKnownFileTypeEnum),
            nullable=False
        )
    )


class WellKnownFile(SQLModel, table=True):
    __tablename__ = 'well_known_file'

    workflow_step_run_id: int = Field(foreign_key='workflow_step_run.id',
                                      primary_key=True)
    well_known_file_type_id: int = Field(foreign_key='well_known_file_type.id',
                                         primary_key=True)
    path: str

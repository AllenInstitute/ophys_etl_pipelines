"""App database schemas"""
import datetime
from typing import Optional

from sqlalchemy import Column, String, Enum
from sqlmodel import SQLModel, Field

from ophys_etl.workflows.well_known_file_types import WellKnownFileType as \
    WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStep as WorkflowStepEnum


class Workflow(SQLModel, table=True):
    __tablename__ = 'workflow'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column('name', String, unique=True))


class WorkflowStep(SQLModel, table=True):
    __tablename__ = 'workflow_step'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column('name', Enum(WorkflowStepEnum),
                                       unique=True))
    workflow_id: int = Field(foreign_key='workflow.id')


class WorkflowStepRun(SQLModel, table=True):
    __tablename__ = 'workflow_step_run'

    id: Optional[int] = Field(default=None, primary_key=True)
    ophys_experiment_id: str = Field(index=True)
    workflow_step_id: int = Field(foreign_key='workflow_step.id')
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


class WellKnownFileType(SQLModel, table=True):
    __tablename__ = 'well_known_file_type'

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_id: int = Field(foreign_key='workflow_step.id')
    name: str = Field(sa_column=Column(
        'name', Enum(WellKnownFileTypeEnum)))


class WellKnownFile(SQLModel, table=True):
    __tablename__ = 'well_known_file'

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_run_id: int = Field(foreign_key='workflow_step_run.id')
    well_known_file_type_id: int = Field(foreign_key='well_known_file_type.id')
    path: str

"""App database schemas"""
import datetime
from typing import Optional, List, Dict

import numpy as np
from pydantic import PrivateAttr
from sqlalchemy import Column, Enum, UniqueConstraint
from sqlmodel import Field, SQLModel

from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class Workflow(SQLModel, table=True):
    __tablename__ = "workflow"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(
        sa_column=Column(
            "name", Enum(WorkflowNameEnum), unique=True, nullable=False
        )
    )


class WorkflowStep(SQLModel, table=True):
    __tablename__ = "workflow_step"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(
        sa_column=Column("name", Enum(WorkflowStepEnum), nullable=False)
    )
    workflow_id: int = Field(foreign_key="workflow.id")

    __table_args__ = (
        # Makes the combination of name and workflow id unique,
        # since we don't want to allow a given workflow to contain the same
        # workflow step multiple times
        UniqueConstraint(
            "name", "workflow_id", name="workflow_workflow_step_name_uc"
        ),
    )


class WorkflowStepRun(SQLModel, table=True):
    __tablename__ = "workflow_step_run"

    id: Optional[int] = Field(default=None, primary_key=True)
    ophys_experiment_id: Optional[str] = Field(
        index=True,
        description='Ophys experiment id from LIMS that this workflow step run'
                    'is associated with. None if not associated with a '
                    'specific experiment'
    )
    ophys_session_id: Optional[str] = Field(
        index=True,
        description='Ophys session id from LIMS that this workflow step run'
                    'is associated with. None if not associated with a '
                    'specific session'
    )
    ophys_container_id: Optional[str] = Field(
        index=True,
        description='Ophys container id from LIMS that this workflow step run'
                    'is associated with. None if not associated with a '
                    'specific container'
    )
    workflow_step_id: int = Field(foreign_key="workflow_step.id")
    log_path: str
    storage_directory: str
    start: datetime.datetime
    end: datetime.datetime


class MotionCorrectionRun(SQLModel, table=True):
    """Motion correction run"""

    __tablename__ = "motion_correction_run"

    workflow_step_run_id: int = Field(
        foreign_key="workflow_step_run.id", primary_key=True
    )
    max_correction_up: float
    max_correction_down: float
    max_correction_right: float
    max_correction_left: float

    def to_dict(self):
        return {
            "x0": self.max_correction_left,
            "x1": self.max_correction_right,
            "y0": self.max_correction_up,
            "y1": self.max_correction_down,
        }


class OphysROIMaskValue(SQLModel, table=True):
    """Stores a single value of an ROI mask as row, col"""

    __tablename__ = "ophys_roi_mask_value"

    id: Optional[int] = Field(default=None, primary_key=True)
    ophys_roi_id: int = Field(foreign_key="ophys_roi.id")
    row_index: int
    col_index: int


class OphysROI(SQLModel, table=True):
    """Ophys ROI"""

    __tablename__ = "ophys_roi"

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_run_id: int = Field(
        index=True, foreign_key="workflow_step_run.id"
    )
    x: int
    y: int
    width: int
    height: int
    is_in_motion_border: bool  # Set at by segmentation
    is_decrosstalk_invalid_raw: Optional[bool] = None
    is_decrosstalk_invalid_raw_active: Optional[bool] = None
    is_decrosstalk_invalid_unmixed: Optional[bool] = None
    is_decrosstalk_invalid_unmixed_active: Optional[bool] = None
    is_decrosstalk_ghost: Optional[bool] = None

    # This field is not persisted to DB as a field in OphysROI and is
    # populated by OphysExperiment.rois
    _mask_values: List[OphysROIMaskValue] = PrivateAttr()

    # populated by OphysExperiment.rois
    _is_cell: bool = PrivateAttr()

    def to_dict(self, include_exclusion_labels: bool = False) -> Dict:
        """
        Converts OphysROI to dict

        Parameters
        ----------
        include_exclusion_labels
            Whether to include list of exclusion labels

        Returns
        -------
        Dict
            dict representation of OphysROI
        """
        if getattr(self, '_mask_values', None) is None:
            raise ValueError('_mask_values must be set in order to produce '
                             'mask values. It is set by OphysExperiment.rois')
        d = {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "mask": self._generate_binary_mask(self._mask_values),
        }

        if include_exclusion_labels:
            d["exclusion_labels"] = self._get_list_of_exclusion_label_strings()

        return d

    def _get_list_of_exclusion_label_strings(self) -> List[str]:
        exclusion_labels = []
        if self.is_in_motion_border:
            exclusion_labels.append("motion_border")
        if self.is_decrosstalk_ghost:
            exclusion_labels.append("decrosstalk_ghost")
        if self.is_decrosstalk_invalid_raw_active:
            exclusion_labels.append("decrosstalk_invalid_raw_active")
        if self.is_decrosstalk_invalid_raw:
            exclusion_labels.append("decrosstalk_invalid_raw")
        if self.is_decrosstalk_invalid_unmixed:
            exclusion_labels.append("decrosstalk_invalid_unmixed")
        if self.is_decrosstalk_invalid_unmixed_active:
            exclusion_labels.append("decrosstalk_invalid_unmixed_active")

        return exclusion_labels

    def _generate_binary_mask(
            self,
            ophys_roi_mask_values: List[OphysROIMaskValue]
    ) -> List[List[bool]]:
        """
        Generate binary mask for an ROI

        Parameters
        ----------
        ophys_roi_mask_values
            The ophys ROI mask values

        Returns
        -------
        List[List[bool]]
            A list of lists of booleans representing the binary mask
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        for ophys_roi_mask_value in ophys_roi_mask_values:
            mask[
                ophys_roi_mask_value.row_index, ophys_roi_mask_value.col_index
            ] = True
        return mask.tolist()


class NwayCellMatch(SQLModel, table=True):
    """Nway cell match"""

    __tablename__ = "nway_cell_match"

    nway_cell_matching_run_id: int = Field(
        foreign_key="workflow_step_run.id",
        primary_key=True
    )
    ophys_roi_id: int = Field(
        foreign_key='ophys_roi.id',
        primary_key=True
    )
    match_id: str = Field(
        description='An id for the matching rois (it will be the same for all'
                    'matching rois)',
        index=True
    )


class ROIClassifierTrainingRun(SQLModel, table=True):
    """ROI classifier training run.
    Each row represents a single trained model. There can be multiple rows per
    workflow_step_run_id since we train an ensemble"""

    __tablename__ = "roi_classifier_training_run"

    mlflow_run_id: str = Field(
        description="mlflow run id. MLFlow is used for tracking training "
        "metadata",
        primary_key=True,
    )
    ensemble_id: int = Field(
        foreign_key="roi_classifier_ensemble.id", index=True
    )
    sagemaker_job_id: str = Field(
        description="sagemaker job id. Model is trained on sagemaker."
    )


class ROIClassifierEnsemble(SQLModel, table=True):
    """Each row is a single trained roi classifier ensemble"""

    __tablename__ = "roi_classifier_ensemble"

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_run_id: int = Field(foreign_key="workflow_step_run.id")
    mlflow_run_id: str = Field(
        description="mlflow parent run for the ensemble training run"
    )
    classification_threshold: float = Field(
        default=0.5, description='classification threshold'
    )


class ROIClassifierInferenceResults(SQLModel, table=True):
    """Each row is classifier prediction for a single ROI"""

    __tablename__ = "roi_classifier_inference_results"

    roi_id: int = Field(primary_key=True, foreign_key="ophys_roi.id")
    ensemble_id: int = Field(
        primary_key=True, foreign_key="roi_classifier_ensemble.id"
    )
    score: float = Field(
        description="Classifier confidence that this ROI is a cell"
    )
    is_cell: bool = Field(
        description='Whether the ROI is predicted to be a cell'
    )


class WellKnownFileType(SQLModel, table=True):
    __tablename__ = "well_known_file_type"

    id: Optional[int] = Field(default=None, primary_key=True)
    workflow_step_id: int = Field(foreign_key="workflow_step.id")
    name: str = Field(
        sa_column=Column("name", Enum(WellKnownFileTypeEnum), nullable=False)
    )


class WellKnownFile(SQLModel, table=True):
    __tablename__ = "well_known_file"

    workflow_step_run_id: int = Field(
        foreign_key="workflow_step_run.id", primary_key=True
    )
    well_known_file_type_id: int = Field(
        foreign_key="well_known_file_type.id", primary_key=True
    )
    path: str

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from sqlmodel import Session, select

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.db import engine
from ophys_etl.workflows.db.schemas import (
    MotionCorrectionRun,
    OphysROI, OphysROIMaskValue
)
from ophys_etl.workflows.utils.lims_utils import LIMSDB
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_latest_run
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


@dataclass
class Specimen:
    """Container for a specimen"""

    id: str


@dataclass(frozen=True)
class ImagingPlaneGroup:
    id: int
    group_order: int


@dataclass
class OphysSession:
    """Container for an ophys session"""

    id: str
    specimen: Specimen

    @classmethod
    def from_id(cls, id: str) -> "OphysSession":
        """Returns an `OphysSession` given a LIMS id for an
        ophys session

        Parameters
        ----------
        id
            LIMS ID for the ophys session

        """
        query = f"""
            SELECT
                os.specimen_id
            FROM ophys_sessions os
            WHERE os.id = {id}
        """
        lims_db = LIMSDB()
        res = lims_db.query(query=query)

        if len(res) == 0:
            raise ValueError(
                f"Could not fetch OphysSession "
                f"for ophys session id "
                f"{id}"
            )
        res = res[0]

        specimen = Specimen(id=res["specimen_id"])

        return cls(
            id=id,
            specimen=specimen
        )

    @property
    def output_dir(self) -> Path:
        """Where to output files to for this session"""
        base_dir = app_config.output_dir

        output_dir = (
                Path(base_dir)
                / f"specimen_{self.specimen.id}"
                / f"session_{self.id}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @property
    def ophys_experiment_ids(self) -> List[int]:
        query = f"""
            SELECT
                oe.id as ophys_experiment_id
            FROM ophys_experiments oe
            WHERE oe.ophys_session_id = {self.id}
        """
        lims_db = LIMSDB()
        res = lims_db.query(query=query)
        return [x['ophys_experiment_id'] for x in res]


@dataclass
class OphysExperiment:
    """Container for an ophys experiment"""

    id: str
    session: OphysSession
    specimen: Specimen
    storage_directory: Path
    raw_movie_filename: Path
    movie_frame_rate_hz: float
    imaging_plane_group: Optional[ImagingPlaneGroup] = None

    @property
    def output_dir(self) -> Path:
        """Where to output files to for this experiment"""
        base_dir = app_config.output_dir

        output_dir = (
            Path(base_dir)
            / f"specimen_{self.specimen.id}"
            / f"session_{self.session.id}"
            / f"experiment_{self.id}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @classmethod
    def from_id(cls, id: str) -> "OphysExperiment":
        """Returns an `OphysExperiment` given a LIMS id for an
        ophys experiment

        Parameters
        ----------
        id
            LIMS ID for the ophys experiment

        """
        query = f"""
            SELECT
                oe.storage_directory,
                oe.ophys_session_id as session_id,
                os.specimen_id,
                oe.movie_frame_rate_hz,
                oe.imaging_plane_group_id,
                oipg.group_order as imaging_plane_group_order,
                images.jp2 as raw_movie_filename
            FROM ophys_experiments oe
            JOIN images on images.id = oe.ophys_primary_image_id
            JOIN ophys_sessions os on os.id = oe.ophys_session_id
            LEFT JOIN ophys_imaging_plane_groups oipg on 
                oipg.id = oe.imaging_plane_group_id
            WHERE oe.id = {id}
        """
        lims_db = LIMSDB()
        res = lims_db.query(query=query)

        if len(res) == 0:
            raise ValueError(
                f"Could not fetch OphysExperiment "
                f"for ophys experiment id "
                f"{id}"
            )
        res = res[0]

        specimen = Specimen(id=res["specimen_id"])
        session = OphysSession(id=res["session_id"], specimen=specimen)
        if res['ophys_imaging_plane_group_id'] is not None:
            imaging_plane_group = ImagingPlaneGroup(
                id=res['ophys_imaging_plane_group_id'],
                group_order=res['imaging_plane_group_order']
            )
        else:
            imaging_plane_group = None

        return cls(
            id=id,
            storage_directory=Path(res["storage_directory"]),
            movie_frame_rate_hz=res["movie_frame_rate_hz"],
            raw_movie_filename=res["raw_movie_filename"],
            session=session,
            specimen=specimen,
            imaging_plane_group=imaging_plane_group
        )

    @property
    def motion_border(self) -> Dict:
        """
        Motion border for an ophys experiment

        Returns
        -------
        Dict[int]
            A dictionary containing motion border data
        """

        with Session(engine) as session:
            workflow_step_run_id = get_latest_run(
                session,
                WorkflowStepEnum.MOTION_CORRECTION,
                WorkflowNameEnum.OPHYS_PROCESSING,
            )
            query = select(MotionCorrectionRun,).where(
                MotionCorrectionRun.workflow_step_run_id
                == workflow_step_run_id
            )

            result = session.execute(query).one()
            motion_border = result[0]
            return {
                "x0": motion_border.max_correction_left,
                "x1": motion_border.max_correction_right,
                "y0": motion_border.max_correction_up,
                "y1": motion_border.max_correction_down,
            }

    @property
    def rois(self) -> List[OphysROI]:
        """
        ROIs for ophys experiment

        Returns
        -------
        List[OphysROI]
            A list of OphysROI
        """
        with Session(engine) as session:
            workflow_step_run_id = get_latest_run(
                session,
                WorkflowStepEnum.SEGMENTATION,
                WorkflowNameEnum.OPHYS_PROCESSING,
                self.id,
            )
            rois: List[OphysROI] = session.execute(
                select(OphysROI)
                .where(OphysROI.workflow_step_run_id == workflow_step_run_id)
            ).scalars().all()

            roi_mask_values: List[OphysROIMaskValue] = session.execute(
                select(OphysROIMaskValue)
                .join(OphysROI,
                      onclause=OphysROI.id == OphysROIMaskValue.ophys_roi_id)
                .where(OphysROI.workflow_step_run_id == workflow_step_run_id)
            ).scalars().all()

            # Accumulate mask values for each roi
            roi_mask_value_map: Dict[int, List[OphysROIMaskValue]] = \
                defaultdict(list)
            for row in roi_mask_values:
                roi_mask_value_map[row.ophys_roi_id].append(row)

            # Update each roi with list of mask values
            for roi in rois:
                roi._mask_values = roi_mask_value_map[roi.id]

            return rois

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
    OphysROI, OphysROIMaskValue, ROIClassifierInferenceResults
)
from ophys_etl.workflows.utils.lims_utils import LIMSDB
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_latest_workflow_step_run
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

    id: int
    specimen: Specimen

    @classmethod
    def from_id(cls, id: int) -> "OphysSession":
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
class OphysContainer:
    """Ophys experiment container"""
    id: int
    specimen: Specimen

    @classmethod
    def from_id(cls, id: int) -> "OphysContainer":
        """Returns an `OphysContainer` given a LIMS id for an
        ophys container

        Parameters
        ----------
        id
            LIMS ID for the ophys container

        """
        specimen_query = f"""
            SELECT
                DISTINCT os.specimen_id
            FROM ophys_experiments_visual_behavior_experiment_containers oevbec
            JOIN ophys_experiments oe ON oe.id = oevbec.ophys_experiment_id
            JOIN ophys_sessions os ON os.id = oe.ophys_session_id
            WHERE oevbec.visual_behavior_experiment_container_id = {id}
        """
        lims_db = LIMSDB()
        res = lims_db.query(query=specimen_query)

        if len(res) == 0:
            raise ValueError(
                f"Could not fetch specimen "
                f"for ophys container id "
                f"{id}"
            )
        elif len(res) > 1:
            raise RuntimeError(f'Ophys container {id} returned {len(res)} '
                               f'specimen ids. Expected 1.')
        res = res[0]

        specimen = Specimen(id=res["specimen_id"])

        return cls(
            id=id,
            specimen=specimen
        )

    @property
    def output_dir(self) -> Path:
        """Where to output files to for this container"""
        base_dir = app_config.output_dir

        output_dir = (
                Path(base_dir)
                / f"specimen_{self.specimen.id}"
                / f"experiment_container_{self.id}"
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_ophys_experiment_ids(
        self,
        passed_or_qc_only: bool = True
    ) -> List[int]:
        """Gets list of experiment ids in this container

        Parameters
        ----------
        passed_or_qc_only
            Whether to only return experiments with workflow_state of
            "passed" or "qc"
        """
        where_clause = \
            f'oevbec.visual_behavior_experiment_container_id = {self.id}'

        if passed_or_qc_only:
            where_clause += " AND oe.workflow_state IN ('passed', 'qc')"

        query = f"""
            SELECT
                oe.id as ophys_experiment_id
            FROM ophys_experiments_visual_behavior_experiment_containers oevbec
            JOIN ophys_experiments oe ON oe.id = oevbec.ophys_experiment_id
            WHERE {where_clause}
        """

        lims_db = LIMSDB()
        res = lims_db.query(query=query)
        return [x['ophys_experiment_id'] for x in res]


@dataclass
class OphysExperiment:
    """Container for an ophys experiment"""

    id: int
    session: OphysSession
    specimen: Specimen
    storage_directory: Path
    raw_movie_filename: Path
    movie_frame_rate_hz: float
    full_genotype: str
    equipment_name: str
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
    def from_id(cls, id: int) -> "OphysExperiment":
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
                oe.ophys_imaging_plane_group_id,
                oipg.group_order as imaging_plane_group_order,
                images.jp2 as raw_movie_filename,
                oevbec.visual_behavior_experiment_container_id as ophys_container_id,
                dr.full_genotype as full_genotype,
                equipment.name as equipment_name
            FROM ophys_experiments oe
            JOIN images on images.id = oe.ophys_primary_image_id
            JOIN ophys_sessions os on os.id = oe.ophys_session_id
            JOIN specimens sp on sp.id = os.specimen_id
            JOIN donors dr on dr.id = sp.donor_id
            LEFT JOIN ophys_experiments_visual_behavior_experiment_containers oevbec 
                ON oevbec.ophys_experiment_id = oe.id
            LEFT JOIN ophys_imaging_plane_groups oipg on
                oipg.id = oe.ophys_imaging_plane_group_id
            LEFT OUTER JOIN equipment ON equipment.id = os.equipment_id
            WHERE oe.id = {id}
        """     # noqa E402

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
            full_genotype=res["full_genotype"],
            equipment_name=res["equipment_name"],
            raw_movie_filename=res["raw_movie_filename"],
            session=session,
            specimen=specimen,
            imaging_plane_group=imaging_plane_group
        )

    @property
    def motion_border(self) -> MotionCorrectionRun:
        """
        Motion border for an ophys experiment

        Returns
        -------
        Dict[int]
            A dictionary containing motion border data
        """

        with Session(engine) as session:
            workflow_step_run_id = get_latest_workflow_step_run(
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
            return motion_border

    @property
    def is_multiplane(self) -> bool:
        """Is this a multiplane experiment"""
        return self.equipment_name.startswith('MESO')

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
            workflow_step_run_id = get_latest_workflow_step_run(
                session=session,
                workflow_step=WorkflowStepEnum.SEGMENTATION,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                ophys_experiment_id=self.id
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

            # add classification decision
            classifier_inference_res: List[ROIClassifierInferenceResults] = \
                session.execute(
                select(ROIClassifierInferenceResults)
                .join(OphysROI,
                      onclause=(OphysROI.id ==
                                ROIClassifierInferenceResults.roi_id)
                      )
                .where(OphysROI.workflow_step_run_id == workflow_step_run_id)
            ).scalars().all()

            roi_id_classifier_res_map: Dict[
                int, ROIClassifierInferenceResults] = {}
            for row in classifier_inference_res:
                roi_id_classifier_res_map[row.roi_id] = row

            for roi in rois:
                roi_inference_res = roi_id_classifier_res_map.get(roi.id)
                if roi_inference_res is not None:
                    roi._is_cell = roi_inference_res.is_cell

            return rois

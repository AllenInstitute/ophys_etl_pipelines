import json
from types import ModuleType
from typing import Dict, List

from sqlmodel import Session, select

from ophys_etl.modules import trace_extraction
from ophys_etl.modules.trace_extraction.schemas import (
    TraceExtractionInputSchema,
)  # noqa: E501
from ophys_etl.workflows.db.schemas import OphysROI
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class TraceExtractionModule(PipelineModule):
    """Trace extraction module"""

    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs,
    ):
        motion_corrected_ophys_movie_file: OutputFile = kwargs[
            "motion_corrected_ophys_movie_file"
        ]
        self._motion_corrected_ophys_movie_file = str(
            motion_corrected_ophys_movie_file.path
        )
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs,
        )

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.TRACE_EXTRACTION

    @property
    def module_schema(self) -> TraceExtractionInputSchema:
        return TraceExtractionInputSchema()

    @property
    def inputs(self) -> Dict:
        return {
            "storage_directory": str(self.output_path),
            "motion_border": self.ophys_experiment.motion_border.to_dict(),
            "motion_corrected_stack": (self._motion_corrected_ophys_movie_file), # noqa E501
            "rois": [x.to_dict() for x in self.ophys_experiment.rois],
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.TRACE_EXTRACTION_EXCLUSION_LABELS,  # noqa E501
                path=self.output_metadata_path,
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.ROI_TRACE,
                path=self.output_path / "roi_traces.h5",
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_TRACE,
                path=self.output_path / "neuropil_traces.h5",
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_MASK,
                path=self.output_path / "neuropil_masks.json",
            ),
        ]

    @staticmethod
    def save_exclusion_labels_to_db(
        output_files: Dict[str, OutputFile],
        session: Session,
        run_id: int,
        **kwargs
    ):
        """
        Saves trace extract exclusion labels to rois in the db

        Parameters
        ----------
        output_files
            Files output by this module
        session
            sqlalchemy session
        run_id
            workflow step run id
        """
        exclusion_labels_file_path = output_files[
            WellKnownFileTypeEnum.TRACE_EXTRACTION_EXCLUSION_LABELS.value
        ].path
        with open(exclusion_labels_file_path) as f:
            output_json = json.load(f)

        # exclusion_labels: List[Dict]
        # e.g. {"roi_id": 123, "exclusion_label_name": "name"}
        exclusion_labels = output_json["exclusion_labels"]

        for exclusion_label in exclusion_labels:
            # 1. Get ROI
            try:
                roi = session.exec(
                    select(OphysROI).where(OphysROI.id == exclusion_label["roi_id"]) # noqa E501
                ).first()
            except Exception as e:
                raise Exception(
                    f"ROI with id {exclusion_label['roi_id']}"
                    "not found in db"
                ) from e

            # 2. Add exclusion label
            if "empty_neuropil_mask" in exclusion_label["exclusion_label_name"]: # noqa E501
                roi.empty_neuropil_mask = True

            # 3. Save updated roi to db
            session.add(roi)

            # flush to get roi id
            session.flush()

    @property
    def executable(self) -> ModuleType:
        return trace_extraction

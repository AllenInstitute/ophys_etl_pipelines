import json
import logging
from types import ModuleType
from typing import Dict, List

from sqlmodel import Session

from ophys_etl.modules import segment_postprocess
from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class SegmentationModule(PipelineModule):
    """Segmentation module"""

    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs
    ):
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

        denoised_ophys_movie_file: OutputFile = kwargs[
            "denoised_ophys_movie_file"
        ]
        self._denoised_ophys_movie_file = str(denoised_ophys_movie_file.path)

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.SEGMENTATION

    @property
    def inputs(self) -> Dict:
        return {
            "log_level": logging.DEBUG,
            "suite2p_args": {
                "h5py": self._denoised_ophys_movie_file,
                "movie_frame_rate_hz": (
                    self.ophys_experiment.movie_frame_rate_hz
                ),
            },
            "postprocess_args": {},
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.OPHYS_ROIS,
                path=self.output_metadata_path,
            )
        ]

    @property
    def _executable(self) -> ModuleType:
        return segment_postprocess

    @staticmethod
    def save_rois_to_db(
        output_files: Dict[str, OutputFile], session: Session, run_id: int
    ):
        """
        Saves segmentation run rois to db

        Parameters
        ----------
        output_files
            Files output by this module
        session
            sqlalchemy session
        run_id
            workflow step run id
        """
        rois_file_path = output_files[
            WellKnownFileTypeEnum.OPHYS_ROIS.value
        ].path
        with open(rois_file_path) as f:
            rois = json.load(f)

        for roi in rois:
            # 1. Add ROI
            mask = roi['mask_matrix']
            motion_border = 'motion_border' in roi['exclusion_labels']
            small_size = 'small_size' in roi['exclusion_labels']
            roi = OphysROI(
                x=roi['x'],
                y=roi['y'],
                width=roi['width'],
                height=roi['height'],
                workflow_step_run_id=run_id,
                is_in_motion_border=motion_border,
                is_small_size=small_size,
            )
            session.add(roi)

            # flush to get roi id
            session.flush()

            # 2. Add mask
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]:
                        mask_val = OphysROIMaskValue(
                            ophys_roi_id=roi.id, row_index=i, col_index=j
                        )
                        session.add(mask_val)

import json
from types import ModuleType
from typing import Dict, List

import numpy as np

from ophys_etl.workflows.app_config.app_config import app_config
from sqlmodel import Session
from ophys_etl.modules.segment_postprocess.schemas import SegmentPostProcessSchema  # noqa: E501
from ophys_etl.modules import segment_postprocess
from ophys_etl.utils.rois import is_inside_motion_border
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

        denoised_ophys_movie_file: OutputFile = kwargs[
            "denoised_ophys_movie_file"
        ]
        self._denoised_ophys_movie_file = str(denoised_ophys_movie_file.path)

        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.SEGMENTATION

    @property
    def module_schema(self) -> SegmentPostProcessSchema:
        return SegmentPostProcessSchema()

    @property
    def inputs(self) -> Dict:
        return {
            "suite2p_args": {
                "h5py": self._denoised_ophys_movie_file,
                "movie_frame_rate_hz": (
                    self.ophys_experiment.movie_frame_rate_hz
                )
            },
            "postprocess_args": {},
            "output_json": self.output_metadata_path
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
    def executable(self) -> ModuleType:
        return segment_postprocess

    @staticmethod
    def save_rois_to_db(
        output_files: Dict[str, OutputFile],
        session: Session,
        run_id: int,
        **kwargs
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

        if app_config.is_debug:
            # replacing rois with dummy rois since we want to ensure at least
            # 1 roi was detected for testing
            rois = SegmentationModule._create_dummy_rois()

        for roi in rois:
            # 1. Add ROI
            mask = roi['mask_matrix']
            motion_border = OphysExperiment.from_id(
                id=kwargs['ophys_experiment_id']).motion_border

            roi['max_correction_right'] = motion_border.max_correction_right
            roi['max_correction_left'] = motion_border.max_correction_left
            roi['max_correction_up'] = motion_border.max_correction_up
            roi['max_correction_down'] = motion_border.max_correction_down

            is_in_motion_border = not is_inside_motion_border(
                roi=roi,
                movie_shape=app_config.fov_shape
            )
            roi = OphysROI(
                x=roi['x'],
                y=roi['y'],
                width=roi['width'],
                height=roi['height'],
                workflow_step_run_id=run_id,
                is_in_motion_border=is_in_motion_border
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

    @staticmethod
    def _create_dummy_rois() -> List[Dict]:
        """Returns a list of dummy rois to be used for testing purposes"""
        roi = {
            'x': 100,
            'y': 100,
            'width': 10,
            'height': 10
        }
        mask = np.zeros((roi['height'], roi['width']), dtype='uint8')
        for i in range(5, 8):
            for j in range(5, 8):
                mask[i, j] = 1
        roi['mask_matrix'] = mask
        return [roi]

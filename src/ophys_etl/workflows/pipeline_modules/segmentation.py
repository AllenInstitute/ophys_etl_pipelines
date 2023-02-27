import logging
from typing import List, Dict

import json
from sqlmodel import Session

from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue
from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.pipeline_module import PipelineModule, OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileType


class SegmentationModule(PipelineModule):
    """Segmentation module"""
    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        debug: bool = False,
        prevent_file_overwrites: bool = True,
        **kwargs
    ):
        super().__init__(
            ophys_experiment=ophys_experiment,
            debug=debug,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

        denoised_ophys_movie_file: OutputFile = \
            kwargs['denoised_ophys_movie_file']
        self._denoised_ophys_movie_file = str(denoised_ophys_movie_file.path)

    @property
    def queue_name(self) -> str:
        return 'SUITE2P_SEGMENTATION_QUEUE'

    @property
    def inputs(self) -> Dict:
        return {
            'log_level': logging.DEBUG,
            'suite2p_args': {
                'h5py': self._denoised_ophys_movie_file,
                'movie_frame_rate_hz': (
                    self.ophys_experiment.movie_frame_rate_hz)
            },
            'postprocess_args': {}
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileType.OPHYS_ROIS,
                path=self.output_metadata_path)
        ]

    @property
    def _executable(self) -> str:
        return 'ophys_etl.modules.segment_postprocess'

    @staticmethod
    def save_rois_to_db(
            output_files: Dict[str, OutputFile],
            session: Session,
            run_id: int
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
        rois_file_path = \
            output_files[WellKnownFileType.OPHYS_ROIS.value].path
        with open(rois_file_path) as f:
            rois = json.load(f)

        for roi in rois:
            # 1. Add ROI
            mask = roi['mask_matrix']
            roi = OphysROI(
                x=roi['x'],
                y=roi['y'],
                width=roi['width'],
                height=roi['height'],
                workflow_step_run_id=run_id
            )
            session.add(roi)

            # flush to get roi id
            session.flush()

            # 2. Add mask
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]:
                        mask_val = OphysROIMaskValue(
                            ophys_roi_id=roi.id,
                            row_index=i,
                            col_index=j
                        )
                        session.add(mask_val)

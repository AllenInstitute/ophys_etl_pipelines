from typing import List, Dict

import json
from deepcell.datasets.channel import Channel, channel_filename_prefix_map
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.well_known_file_types import WellKnownFileType

from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.pipeline_module import PipelineModule, OutputFile


class GenerateThumbnailsModule(PipelineModule):
    """Generates thumbnail images"""
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
        rois_file: OutputFile = kwargs['rois_file']
        correlation_projection_graph_file: OutputFile = \
            kwargs['correlation_projection_graph_file']

        self._denoised_ophys_movie_file = str(denoised_ophys_movie_file.path)
        self._rois_file = str(rois_file.path)
        self._correlation_graph_file = correlation_projection_graph_file

    @property
    def queue_name(self) -> str:
        return 'ROI_CLASSIFICATION_GENERATE_THUMBNAILS'

    @property
    def inputs(self) -> Dict:
        return {
            'video_path': self._denoised_ophys_movie_file,
            'roi_path': self._rois_file,
            'graph_path': self._correlation_graph_file,
            'out_dir': self.output_path / 'thumbnails',
            'channels': (
                app_config.pipeline_steps.roi_classification.input_channels)
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.ROI_CLASSIFICATION_THUMBNAIL_IMAGES),
                path=self.inputs['out_dir']
            )
        ]

    @property
    def _executable(self) -> str:
        return 'ophys_etl.modules.roi_cell_classifier.compute_classifier_artifacts' # noqa E402

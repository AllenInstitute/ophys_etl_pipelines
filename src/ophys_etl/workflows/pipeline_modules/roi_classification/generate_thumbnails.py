from types import ModuleType
from typing import Dict, List

from ophys_etl.modules.roi_cell_classifier import compute_classifier_artifacts
from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class GenerateThumbnailsModule(PipelineModule):
    """Generates thumbnail images"""

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
        rois_file: OutputFile = kwargs["rois_file"]
        correlation_projection_graph_file: OutputFile = kwargs[
            "correlation_projection_graph_file"
        ]
        is_training: bool = kwargs["is_training"]

        self._denoised_ophys_movie_file = str(denoised_ophys_movie_file.path)
        self._rois_file = str(rois_file.path)
        self._correlation_graph_file = correlation_projection_graph_file
        self._is_training = is_training

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_THUMBNAILS

    @property
    def inputs(self) -> Dict:
        d = {
            "experiment_id": self._ophys_experiment.id,
            "video_path": self._denoised_ophys_movie_file,
            "roi_path": self._rois_file,
            "graph_path": self._correlation_graph_file,
            "channels": (
                app_config.pipeline_steps.roi_classification.input_channels
            ),
            "out_dir": self.output_path / "thumbnails",
            "is_training": self._is_training,
        }
        if self._is_training:
            d[
                "cell_labeling_app_host"
            ] = app_config.pipeline_steps.roi_classification
        return d

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_THUMBNAIL_IMAGES
                ),
                path=self.inputs["out_dir"],
            )
        ]

    @property
    def executable(self) -> ModuleType:
        return compute_classifier_artifacts

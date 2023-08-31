from types import ModuleType
from typing import Dict, List

from ophys_etl.modules.segmentation.modules.schemas import CalculateEdgesInputSchema # noqa E501
from ophys_etl.modules.segmentation.modules import calculate_edges
from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class GenerateCorrelationProjectionModule(PipelineModule):
    """Generate correlation projection graph"""

    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs,
    ):
        denoised_ophys_movie_file: OutputFile = kwargs[
            "denoised_ophys_movie_file"
        ]
        self._denoised_ophys_movie_file = str(denoised_ophys_movie_file.path)
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs,
        )

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return (
            WorkflowStepEnum.ROI_CLASSIFICATION_GENERATE_CORRELATION_PROJECTION_GRAPH # noqa E501
        )

    @property
    def module_schema(self) -> CalculateEdgesInputSchema:
        return CalculateEdgesInputSchema()

    @property
    def inputs(self) -> Dict:
        return {
            "video_path": self._denoised_ophys_movie_file,
            "graph_output": (
                self.output_path
                / f"{self._ophys_experiment.id}_correlation_graph.pkl"
            ),
            "attribute_name": "filtered_hnc_Gaussian",
            "neighborhood_radius": 7,
            "n_parallel_workers": (
                app_config.pipeline_steps.roi_classification.generate_correlation_projection.n_workers # noqa E501
            ),
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_CORRELATION_PROJECTION_GRAPH # noqa E501
                ),
                path=(
                    self.output_path
                    / f"{self._ophys_experiment.id}_correlation_graph.pkl"
                ),
            )
        ]

    @property
    def executable(self) -> ModuleType:
        return calculate_edges

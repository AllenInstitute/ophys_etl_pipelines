from types import ModuleType
from typing import List

from ophys_etl.modules import demix
from ophys_etl.modules.demix.schemas import DemixJobSchema
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum


class DemixTracesModule(PipelineModule):
    """Wrapper around trace demixing module"""

    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs
    ):

        motion_corrected_ophys_movie_file: OutputFile = kwargs[
            "motion_corrected_ophys_movie_file"
        ]
        self._motion_corrected_ophys_movie_file = str(
            motion_corrected_ophys_movie_file.path
        )
        roi_traces_file: OutputFile = kwargs[
            "roi_traces_file"
        ]
        self._roi_trace_file = str(
            roi_traces_file.path
        )

        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

    @property
    def executable(self) -> ModuleType:
        return demix

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.DEMIX_TRACES

    @property
    def module_schema(self) -> DemixJobSchema:
        return DemixJobSchema()

    @property
    def inputs(self):
        return {
            "movie_h5": self._motion_corrected_ophys_movie_file,
            "traces_h5": self._roi_trace_file,
            "output_file": str(
                self.output_path
                / f"{self._ophys_experiment.id}_demixed_traces.h5"
            ),
            "roi_masks": [x.to_dict(include_exclusion_labels=True)
                          for x in self.ophys_experiment.rois],
        }

    @property
    def outputs(self) -> List[OutputFile]:
        # Currently not loading the outputs negative_transient_roi_ids and
        # negative_baseline_roi_ids from the output json. These outputs are
        # lists of ROI ids that "suffer" from these processing issues.
        # These could be added as flag columns in the ROI fields in the
        # future if needed.
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DEMIXED_TRACES
                ),
                path=(
                    self.output_path
                    / f"{self._ophys_experiment.id}_demixed_traces.h5"
                ),
            ),
        ]

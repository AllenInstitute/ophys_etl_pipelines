from types import ModuleType
from typing import Dict, List

from deepcell.cli.modules import create_dataset
from deepcell.cli.modules.create_dataset import CreateDatasetInputSchema

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class CreateTrainTestSplitModule(PipelineModule):
    """Splits training data into a train and test set and outputs json files
    with these splits"""

    def __init__(self, prevent_file_overwrites: bool = True, **kwargs):
        thumbnail_dirs: OutputFile = kwargs["thumbnail_dirs"]
        self._thumbnail_dirs = thumbnail_dirs
        super().__init__(
            ophys_experiment=None,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.ROI_CLASSIFICATION_CREATE_TRAIN_TEST_SPLIT

    @property
    def module_schema(self) -> CreateDatasetInputSchema:
        return CreateDatasetInputSchema()

    @property
    def inputs(self) -> Dict:
        return {
            "cell_labeling_app_host": (
                app_config.pipeline_steps.roi_classification.cell_labeling_app_host # noqa E501
            ),
            "lims_db_username": app_config.lims_db.username.get_secret_value(),
            "lims_db_password": app_config.lims_db.password.get_secret_value(),
            "output_dir": self.output_path,
            "channels": (
                app_config.pipeline_steps.roi_classification.input_channels
            ),
            "artifact_dir": self._thumbnail_dirs,
            "test_size": (
                app_config.pipeline_steps.roi_classification.training.train_test_split.test_size # noqa E501
            ),
            "seed": 1234,
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAIN_SET
                ),
                path=self.output_path / "train_rois.json",
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_TEST_SET
                ),
                path=self.output_path / "test_rois.json",
            ),
        ]

    @property
    def executable(self) -> ModuleType:
        return create_dataset

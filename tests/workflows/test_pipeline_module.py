import datetime
import json
import os
import shutil
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from ophys_etl.test_utils.workflow_utils import setup_app_config
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

setup_app_config(
    ophys_workflow_app_config_path=(
        Path(__file__).parent / "resources" / "config.yml"
    ),
    test_di_base_model_path=Path(__file__).parent
    / "resources"
    / "di_model.h5",
)

import tempfile  # noqa #402
from typing import Dict, List  # noqa #402

from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,  # noqa #402
    OphysSession,
    Specimen,
)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import (
    ModuleOutputFileExistsError,
)  # noqa #402
from ophys_etl.workflows.pipeline_module import PipelineModule


class _DummyMod(PipelineModule):
    _temp_out = tempfile.TemporaryDirectory()

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE

    @property
    def inputs(self) -> Dict:
        return {"foo": 1, "bar": 2}

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                path=Path(self._temp_out.name) / "output1",
                well_known_file_type="foo",
            )
        ]

    @property
    def executable(self) -> ModuleType:
        class DummyExecutable:
            @staticmethod
            def __name__():
                return "foo.bar"
        DummyExecutable: ModuleType
        return DummyExecutable


class TestPipelineModule:
    @classmethod
    def setup_class(cls):
        cls._now = datetime.datetime(
            year=2023,
            month=1,
            day=1,
            hour=1,
            minute=1,
            second=1,
            microsecond=1,
        )

        with patch("datetime.datetime", wraps=datetime.datetime) as mock_dt:
            mock_dt.now.return_value = cls._now
            cls._dummy_mod = _DummyMod(
                ophys_experiment=OphysExperiment(
                    id="1",
                    session=OphysSession(id="2", specimen=Specimen("1")),
                    specimen=Specimen(id="3"),
                    storage_directory=Path("/storage_dir"),
                    raw_movie_filename=Path("mov.h5"),
                    movie_frame_rate_hz=11.0,
                ),
                docker_tag="main",
            )

    @classmethod
    def teardown_class(cls):
        if cls._dummy_mod.input_args_path.exists():
            os.remove(cls._dummy_mod.input_args_path)
        shutil.rmtree(cls._dummy_mod._temp_out.name)

    @patch("datetime.datetime", wraps=datetime.datetime)
    def test_output_path(self, mock_dt):
        mock_dt.now.return_value = self._now

        assert (
            self._dummy_mod.output_path
            == Path("/tmp")
            / "specimen_3"
            / "session_2"
            / "experiment_1"
            / self._dummy_mod.queue_name.value
            / "2023-01-01_01-01-01-000001"
        )

    @patch("datetime.datetime", wraps=datetime.datetime)
    def test_output_metadata_path(self, mock_dt):
        mock_dt.now.return_value = self._now

        assert (
            self._dummy_mod.output_metadata_path
            == Path("/tmp")
            / "specimen_3"
            / "session_2"
            / "experiment_1"
            / self._dummy_mod.queue_name.value
            / "2023-01-01_01-01-01-000001"
            / f"{self._dummy_mod.queue_name.value}_output.json"
        )

    @patch("datetime.datetime", wraps=datetime.datetime)
    def test_input_args_path(self, mock_dt):
        mock_dt.now.return_value = self._now

        assert (
            self._dummy_mod.input_args_path
            == Path("/tmp")
            / "specimen_3"
            / "session_2"
            / "experiment_1"
            / self._dummy_mod.queue_name.value
            / "2023-01-01_01-01-01-000001"
            / f"{self._dummy_mod.queue_name.value}_input.json"
        )

    @patch("datetime.datetime", wraps=datetime.datetime)
    def test_write_input_args(self, mock_dt):
        mock_dt.now.return_value = self._now

        self._dummy_mod.write_input_args()
        with open(self._dummy_mod.input_args_path) as f:
            input_args = json.load(f)

        assert input_args == self._dummy_mod.inputs

    @pytest.mark.parametrize("file_exists", (True, False))
    def test_validate_file_overwrites(self, file_exists):
        for out in self._dummy_mod.outputs:
            if out.path.exists():
                os.remove(out.path)

        if file_exists:
            for out in self._dummy_mod.outputs:
                with open(out.path, "w") as f:
                    f.write("foo")

        if file_exists:
            with pytest.raises(ModuleOutputFileExistsError):
                self._dummy_mod._validate_file_overwrite()
        else:
            self._dummy_mod._validate_file_overwrite()

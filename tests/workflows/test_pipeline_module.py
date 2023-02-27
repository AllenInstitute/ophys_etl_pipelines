import os
import shutil
from pathlib import Path

import json

import pytest

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent / 'resources' / 'config.yml'),
    test_di_base_model_path=Path(__file__).parent / 'resources' / 'di_model.h5'
)

import tempfile # noqa #402
from typing import List, Dict   # noqa #402

from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession, Specimen  # noqa #402
from ophys_etl.workflows.pipeline_module import PipelineModule, \
    ModuleOutputFileExistsError, OutputFile  # noqa #402


class _DummyMod(PipelineModule):
    _temp_out = tempfile.TemporaryDirectory()

    @property
    def queue_name(self) -> str:
        return 'queue_name'

    @property
    def inputs(self) -> Dict:
        return {
            'foo': 1,
            'bar': 2
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                path=Path(self._temp_out.name) / 'output1',
                well_known_file_type='foo'
            )
        ]

    @property
    def _executable(self) -> str:
        return 'foo.bar'


class TestPipelineModule:
    @classmethod
    def setup_class(cls):
        cls._dummy_mod = _DummyMod(
            ophys_experiment=OphysExperiment(
                id='1',
                session=OphysSession(id='2'),
                specimen=Specimen(id='3'),
                storage_directory=Path('/storage_dir'),
                raw_movie_filename=Path('mov.h5'),
                movie_frame_rate_hz=11.0
            )
        )

    @classmethod
    def teardown_class(cls):
        if cls._dummy_mod.input_args_path.exists():
            os.remove(cls._dummy_mod.input_args_path)
        shutil.rmtree(cls._dummy_mod._temp_out.name)

    def test_output_path(self):
        assert self._dummy_mod.output_path == \
               Path('/tmp') / 'specimen_3' / 'session_2' / 'experiment_1' / \
               'queue_name'

    def test_output_metadata_path(self):
        assert self._dummy_mod.output_metadata_path == \
               Path('/tmp') / 'specimen_3' / 'session_2' / 'experiment_1' / \
               'queue_name' / 'queue_name_1_output.json'

    def test_input_args_path(self):
        assert self._dummy_mod.input_args_path == \
               Path('/tmp') / 'specimen_3' / 'session_2' / 'experiment_1' / \
               'queue_name' / 'queue_name_1_input.json'

    def test_write_input_args(self):
        self._dummy_mod.write_input_args()
        with open(self._dummy_mod.input_args_path) as f:
            input_args = json.load(f)

        assert input_args == self._dummy_mod.inputs

    @pytest.mark.parametrize('file_exists', (True, False))
    def test_validate_file_overwrites(self, file_exists):
        for out in self._dummy_mod.outputs:
            if out.path.exists():
                os.remove(out.path)

        if file_exists:
            for out in self._dummy_mod.outputs:
                with open(out.path, 'w') as f:
                    f.write('foo')

        if file_exists:
            with pytest.raises(ModuleOutputFileExistsError):
                self._dummy_mod._validate_file_overwrite()
        else:
            self._dummy_mod._validate_file_overwrite()

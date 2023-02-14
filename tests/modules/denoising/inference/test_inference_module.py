import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import json
import numpy as np
import pytest

try:
    from deepinterpolation.cli.inference import Inference

    from ophys_etl.modules.denoising.inference.__main__ import InferenceRunner
except ImportError:
    pass


class TestInferenceRunner:
    @classmethod
    def setup(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.mov_path = Path(cls.tmpdir.name) / 'mov.h5'
        cls.mov_length = 160
        with h5py.File(cls.mov_path, 'w') as f:
            f.create_dataset(name='data',
                             data=np.random.random((cls.mov_length, 512, 512)))
            cls.runner = cls._create_dummy_runner()

    @classmethod
    def teardown_class(cls):
        cls.tmpdir.cleanup()

    @classmethod
    def _create_dummy_runner(cls):
        with open(Path(__file__).parent / 'test_data' /
                  'inference_input.json') as f:
            inference_input = json.load(f)

        # Create a dummy model
        dummy_model_path = Path(cls.tmpdir.name) / 'model.h5'
        with h5py.File(dummy_model_path, 'w') as f:
            f.create_group(name='model_weights')
        inference_input['inference_params']['model_source']['local_path'] = \
            str(dummy_model_path)

        inference_input['inference_params']['output_file'] = \
            Path(cls.tmpdir.name) / 'mov_denoised.h5'

        def dummy_init():
            return None

        with patch.object(InferenceRunner, '__init__', wraps=dummy_init):
            runner = InferenceRunner()
            runner.args = inference_input
        return runner

    @pytest.mark.deepinterpolation_only
    def test_run(self):
        """Smoke test that the Inference interface can be called with
        test arguments"""
        def dummy_run():
            return None

        with patch.object(Inference, 'run', wraps=dummy_run):
            self.runner.run()

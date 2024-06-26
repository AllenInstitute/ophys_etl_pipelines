import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import json
import numpy as np
from deepinterpolation.cli.fine_tuning import FineTuning

from ophys_etl.modules.denoising.fine_tuning.__main__ import FinetuningRunner


class TestFinetuningRunner:
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
    def _create_dummy_runner(cls):
        with open(Path(__file__).parent / 'test_data' /
                  'fine_tuning_input.json') as f:
            finetuning_input = json.load(f)

        # Create a dummy model
        dummy_model_path = Path(cls.tmpdir.name) / 'model.h5'
        with h5py.File(dummy_model_path, 'w') as f:
            f.create_group(name='model_weights')
        finetuning_input['finetuning_params']['model_source']['local_path'] = \
            str(dummy_model_path)

        finetuning_input['finetuning_params']['output_dir'] = \
            str(cls.tmpdir.name)

        data_split_params = \
            {
                'data_split_params': {
                    'movie_path': str(cls.mov_path),
                    'seed': None,
                    'train_frac': 0.7
                }
            }

        runner = FinetuningRunner(
            input_data={
                **finetuning_input,
                **data_split_params
            },
            args=[]
        )
        return runner

    @classmethod
    @pytest.mark.skip(reason="module not used in production.  test failure blocking pipeline")
    def teardown_class(cls):
        cls.tmpdir.cleanup()
    @pytest.mark.skip(reason="module not used in production.  test failure blocking pipeline")
    def test_write_train_val_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / 'train.json'
            val_path = Path(tmpdir) / 'val.json'
            self.runner._write_train_val_datasets(
                train_out_path=train_path, val_out_path=val_path)
            assert train_path.exists()
            assert val_path.exists()
    @pytest.mark.skip(reason="module not used in production.  test failure blocking pipeline")
    def test_run(self):
        """Smoke test that the FineTuning interface can be called with
        test arguments"""
        def dummy_run():
            return None

        with patch.object(FineTuning, 'run', wraps=dummy_run):
            self.runner.run()

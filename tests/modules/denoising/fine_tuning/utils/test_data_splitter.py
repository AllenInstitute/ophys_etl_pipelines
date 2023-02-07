import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from ophys_etl.modules.denoising.fine_tuning.utils.data_splitter import \
    DataSplitter


class TestDataSplitter:
    @classmethod
    def setup_class(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.mov_path = Path(cls.tmpdir.name) / 'mov.h5'
        cls.mov_length = 102
        cls.window_size = 1
        with h5py.File(cls.mov_path, 'w') as f:
            f.create_dataset(name='data',
                             data=np.random.random((cls.mov_length, 512, 512)))

    @classmethod
    def teardown_class(cls):
        cls.tmpdir.cleanup()

    @pytest.mark.parametrize('downsample_frac', (0.0, 0.1))
    @pytest.mark.parametrize('train_frac', (0.5, 0.7))
    def test_data_splitter(self, train_frac, downsample_frac):
        data_splitter = DataSplitter(
            movie_path=self.mov_path,
            downsample_frac=downsample_frac

        )
        train, val = data_splitter.get_train_val_split(
            train_frac=train_frac,
            window_size=self.window_size
        )

        # num frames at beginning and end which need to be cut off
        # due to the window around the first/last example
        bookend_len = self.window_size * 2

        # Buffer around last training example and first val example
        val_buffer = self.window_size * 2

        # subtract bookends
        mov_len = self.mov_length - bookend_len

        # downsample
        mov_len -= int(mov_len * downsample_frac)

        expected_train_len = int(mov_len * train_frac)
        expected_val_len = mov_len - expected_train_len - val_buffer

        assert len(train) == expected_train_len
        assert len(val) == expected_val_len
        assert len(set(train).intersection(val)) == 0

import pathlib
import shutil
import tempfile
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

from ophys_etl.types import ExtractROI
from \
    ophys_etl.modules.roi_cell_classifier.compute_classifier_artifacts import \
    ClassifierArtifactsGenerator, _pad_frames, _downsample_frames


class TestComputeClassifierArtifacts:

    @classmethod
    def setup_class(cls):
        file_loc = pathlib.Path(__file__)
        resource_loc = file_loc.parent / 'resources'
        cls.test_files = np.sort(
            [str(t_file) for t_file in resource_loc.glob('*')])

        cls.rng = np.random.default_rng(1234)

        cls.frames_image_size = 100
        cls.x0_y0_width_height = 10
        cls.centroid = 15
        cls.exp_id = 12345
        cls.extract_roi = ExtractROI(id=1,
                                     x=cls.x0_y0_width_height,
                                     y=cls.x0_y0_width_height,
                                     width=cls.x0_y0_width_height,
                                     height=cls.x0_y0_width_height,
                                     valid=True,
                                     mask=np.ones(
                                         (cls.x0_y0_width_height,
                                          cls.x0_y0_width_height)).tolist())

        cls.data = cls.rng.integers(low=0,
                                    high=2,
                                    size=(cls.frames_image_size,
                                          cls.frames_image_size,
                                          cls.frames_image_size),
                                    dtype=int)
        cls.output_path = tempfile.mkdtemp()
        _, cls.video_path = tempfile.mkstemp(
            dir=cls.output_path,
            prefix=f'{cls.exp_id}_',
            suffix='.h5')

        with h5py.File(cls.video_path, 'w') as h5_file:
            h5_file.create_dataset(name='data', data=cls.data)

        cls.args = {'video_path': cls.video_path,
                    'roi_path': cls.video_path,
                    'is_training': False,
                    'experiment_id': str(cls.exp_id),
                    'out_dir': cls.output_path,
                    'cutout_size': 128}

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.output_path)

    @pytest.mark.parametrize('is_training', (True, False))
    def test_run(self, is_training: bool):
        """Test run method works and the images produced have the expected
        values.
        """
        args = self.args
        if is_training:
            args['is_training'] = True
            args['cell_labeling_app_host'] = 'foo'

        labels = pd.read_csv(pathlib.Path(__file__).parent / 'resources' /
                             'labels.csv', dtype={'experiment_id': str})
        with patch('ophys_etl.modules.roi_cell_classifier.'
                   'compute_classifier_artifacts.sanitize_extract_roi_list',
                    Mock(return_value=[self.extract_roi])), \
            patch('ophys_etl.modules.roi_cell_classifier.'
                  'compute_classifier_artifacts.json.loads',
                  Mock()), \
            patch('ophys_etl.modules.roi_cell_classifier.'
                  'compute_classifier_artifacts.construct_dataset',
                  Mock(return_value=labels)):
            gen = ClassifierArtifactsGenerator(
                args=[], input_data=args)
            gen.run()

        output_frames = np.load(str(pathlib.Path(self.output_path) /
                                f'{self.exp_id}_{self.extract_roi["id"]}.npy'))

        assert output_frames.shape == (
            gen.args['n_frames'],
            gen.args['cutout_size'],
            gen.args['cutout_size'],
            3
        )

    def test__pad_frames_no_padding(self):
        with h5py.File(self.video_path, 'r') as f:
            mov = f['data'][()]
        frames = np.ones_like(mov, shape=(20, *mov.shape[1:]))
        frames_padded = _pad_frames(
            frames=frames,
            desired_seq_len=20
        )
        np.testing.assert_array_equal(frames, frames_padded)

    def test__pad_frames(self):
        with h5py.File(self.video_path, 'r') as f:
            mov = f['data'][()]
        desired_seq_len = 20
        frames = np.ones_like(mov, shape=(10, *mov.shape[1:]))
        frames_padded = _pad_frames(
            frames=frames,
            desired_seq_len=desired_seq_len,
        )

        expected = np.concatenate([
            frames,
            np.zeros_like(frames,
                          shape=(desired_seq_len - len(frames),
                                 *frames.shape[1:]))
        ])
        np.testing.assert_array_equal(expected, frames_padded)

    @pytest.mark.parametrize('downsampling_factor', (2, 4))
    def test_temporal_downsampling(self, downsampling_factor):
        with h5py.File(self.video_path, 'r') as f:
            mov = f['data'][()]

        desired_n_frames = 16
        n_frames = desired_n_frames * downsampling_factor
        center = 50
        frames = mov[center - int(n_frames / 2):
                     center + int(n_frames/2)]
        frames = _downsample_frames(
            frames=frames,
            downsampling_factor=downsampling_factor
        )

        assert len(frames) == desired_n_frames

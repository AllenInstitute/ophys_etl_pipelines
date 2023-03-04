from glob import glob
import pathlib
import shutil
import tempfile
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
from PIL import Image
from deepcell.datasets.channel import Channel, channel_filename_prefix_map

from ophys_etl.types import ExtractROI
from \
    ophys_etl.modules.roi_cell_classifier.compute_classifier_artifacts import \
    ClassifierArtifactsGenerator
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.utils.rois import extract_roi_to_ophys_roi


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
                    'graph_path': cls.video_path,
                    'channels': [
                        Channel.CORRELATION_PROJECTION.value,
                        Channel.MAX_PROJECTION.value,
                        Channel.MASK.value],
                    'is_training': False,
                    'experiment_id': str(cls.exp_id),
                    'out_dir': cls.output_path,
                    'low_quantile': 0.2,
                    'high_quantile': 0.99,
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
            args['labels_path'] = str(pathlib.Path(__file__).parent /
                                      'resources' / 'labels.csv')
        image_value = 100

        corr_img = np.ones((self.frames_image_size, self.frames_image_size))
        corr_img[self.centroid, self.centroid] = image_value

        with patch('ophys_etl.modules.roi_cell_classifier.'
                   'compute_classifier_artifacts.graph_to_img',
                   Mock(return_value=corr_img)), \
                patch('ophys_etl.modules.roi_cell_classifier.'
                      'compute_classifier_artifacts.sanitize_extract_roi_list',
                      Mock(return_value=[self.extract_roi])), \
                patch('ophys_etl.modules.roi_cell_classifier.'
                      'compute_classifier_artifacts.json.loads',
                      Mock()):
            classArtifacts = ClassifierArtifactsGenerator(
                args=[], input_data=args)
            classArtifacts.run()

        output_file_list = np.sort(glob(f'{self.output_path}/*.png'))
        assert len(output_file_list) == len(self.args['channels'])

        for output_file in output_file_list:
            for test_file in self.test_files:
                if pathlib.Path(test_file).name == \
                        pathlib.Path(output_file).name:
                    image = Image.open(output_file)
                    test = Image.open(test_file)
                    np.testing.assert_array_equal(np.array(image),
                                                  np.array(test))

    def test_write_thumbnails(self):
        """Test that artifact thumbnails are written.
        """
        image_value = 100

        max_img = np.ones((self.frames_image_size, self.frames_image_size),
                          dtype='uint8')
        max_img[self.centroid, self.centroid] = image_value
        avg_img = np.ones((self.frames_image_size, self.frames_image_size),
                          dtype='uint8')
        avg_img[self.centroid, self.centroid] = image_value
        corr_img = np.ones((self.frames_image_size, self.frames_image_size),
                           dtype='uint8')
        corr_img[self.centroid, self.centroid] = image_value

        classArtifacts = ClassifierArtifactsGenerator(args=[],
                                                      input_data=self.args)

        roi = extract_roi_to_ophys_roi(roi=self.extract_roi)

        mask = classArtifacts._generate_mask_image(
            roi=roi
        )

        imgs = {
            Channel.MAX_PROJECTION: max_img,
            Channel.AVG_PROJECTION: avg_img,
            Channel.CORRELATION_PROJECTION: corr_img,
            Channel.MASK: mask
        }

        classArtifacts._write_thumbnails(
            roi=roi,
            imgs=imgs,
            exp_id=str(self.exp_id))

        output_file_list = np.sort(glob(f'{self.output_path}/*.png'))
        self.assertEqual(len(output_file_list), len(self.args['channels']))

    def test__generate_max_activation_image(self):
        """Test that max activation image is generated correctly"""
        roi = extract_roi_to_ophys_roi(roi=self.extract_roi)
        fov_shape = (512, 512)
        mov = np.zeros((5, *fov_shape))
        mov[3] = 1
        max_activation = \
            ClassifierArtifactsGenerator._generate_max_activation_image(
                mov=mov,
                roi=roi
            )
        np.testing.assert_array_equal(max_activation, np.ones(fov_shape))

    def test__write_thumbnails_max_activation(self):
        """Tests that max activation thumbnail written correctly"""
        args = self.args
        args['low_quantile'] = 0
        args['high_quantile'] = 1
        args['channels'] = [Channel.MAX_ACTIVATION.value]
        gen = ClassifierArtifactsGenerator(
            input_data=args,
            args=[]
        )
        roi = self.extract_roi
        roi['x'] = 200
        roi['y'] = 200
        roi = extract_roi_to_ophys_roi(roi=self.extract_roi)

        fov_shape = (512, 512)
        mov = np.zeros((5, *fov_shape))
        mov[3] = np.random.random((fov_shape))
        imgs = {
            Channel.MAX_ACTIVATION: (
                ClassifierArtifactsGenerator._generate_max_activation_image(
                    mov=mov,
                    roi=roi
                )
            )
        }

        exp_id = '0'
        gen._write_thumbnails(
            roi=roi,
            exp_id=exp_id,
            imgs=imgs
        )
        filename = \
            f'{channel_filename_prefix_map[Channel.MAX_ACTIVATION]}_' \
            f'{exp_id}_' \
            f'{roi.roi_id}.png'
        img = Image.open(pathlib.Path(self.output_path) / filename)
        img = np.array(img)

        expected = normalize_array(
            roi.get_centered_cutout(
                image=mov[3],
                height=self.args['cutout_size'],
                width=self.args['cutout_size ']
            )
        )
        np.testing.assert_array_equal(
            img,
            expected
        )

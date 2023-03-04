from glob import glob
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from PIL import Image
from deepcell.datasets.channel import Channel, channel_filename_prefix_map

from ophys_etl.types import ExtractROI
from \
    ophys_etl.modules.roi_cell_classifier.compute_classifier_artifacts import \
    ClassifierArtifactsGenerator
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.utils.rois import extract_roi_to_ophys_roi


class TestComputeClassifierArtifacts(unittest.TestCase):

    def setUp(self):
        file_loc = pathlib.Path(__file__)
        resource_loc = file_loc.parent / 'resources'
        self.test_files = np.sort(
            [str(t_file) for t_file in resource_loc.glob('*')])

        self.rng = np.random.default_rng(1234)

        self.frames_image_size = 100
        self.x0_y0_width_height = 10
        self.centroid = 15
        self.exp_id = 12345
        self.extract_roi = ExtractROI(id=1,
                                      x=self.x0_y0_width_height,
                                      y=self.x0_y0_width_height,
                                      width=self.x0_y0_width_height,
                                      height=self.x0_y0_width_height,
                                      valid=True,
                                      mask=np.ones(
                                          (self.x0_y0_width_height,
                                           self.x0_y0_width_height)).tolist())

        self.data = self.rng.integers(low=0,
                                      high=2,
                                      size=(self.frames_image_size,
                                            self.frames_image_size,
                                            self.frames_image_size),
                                      dtype=int)
        self.output_path = tempfile.mkdtemp()
        _, self.video_path = tempfile.mkstemp(
            dir=self.output_path,
            prefix=f'{self.exp_id}_',
            suffix='.h5')

        with h5py.File(self.video_path, 'w') as h5_file:
            h5_file.create_dataset(name='data', data=self.data)

        self.args = {'video_path': self.video_path,
                     'roi_path': self.video_path,
                     'graph_path': self.video_path,
                     'channels': [
                         Channel.CORRELATION_PROJECTION.value,
                         Channel.MAX_PROJECTION.value,
                         Channel.MASK.value],
                     'out_dir': self.output_path,
                     'low_quantile': 0.2,
                     'high_quantile': 0.99,
                     'cutout_size': 128}

    def tearDown(self):
        shutil.rmtree(self.output_path)

    def test_run(self):
        """Test that run executes when no ROI list is specified.
        """
        self._run_wrapper(None)

    def test_run_selected_roi(self):
        """Test that when our ROI is specified, it is successfully written.
        """
        self._run_wrapper([self.extract_roi['id']])

    def _run_wrapper(self, selected_rois):
        """Test run method works and the images produced have the expected
        values.
        """
        self.args['selected_rois'] = selected_rois
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
                args=[], input_data=self.args)
            classArtifacts.run()

        output_file_list = np.sort(glob(f'{self.output_path}/*.png'))
        self.assertEqual(len(output_file_list), len(self.args['channels']))

        for output_file in output_file_list:
            for test_file in self.test_files:
                if pathlib.Path(test_file).name == \
                        pathlib.Path(output_file).name:
                    image = Image.open(output_file)
                    test = Image.open(test_file)
                    np.testing.assert_array_equal(np.array(image),
                                                  np.array(test))

    def test_no_selected_roi(self):
        """Test that an ROI is not written when its id is not specified.
        """
        # Select a roi that is not in the list if input ROIs
        self.args['selected_rois'] = [self.extract_roi['id'] + 1]
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
                args=[], input_data=self.args)
            classArtifacts.run()

        output_file_list = np.sort(glob(f'{self.output_path}/*.png'))
        self.assertEqual(len(output_file_list), 0)

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
            f'{channel_filename_prefix_map[Channel.MAX_ACTIVATION]}_{exp_id}_' \
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

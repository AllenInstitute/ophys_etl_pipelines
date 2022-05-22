from glob import glob
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from PIL import Image

from ophys_etl.types import ExtractROI
from \
    ophys_etl.modules.roi_cell_classifier.compute_classifier_artifacts import \
    ClassifierArtifactsGenerator


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
        self.expected_num_artifact_files = 5
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
        _, self.artifact_path = tempfile.mkstemp(
            dir=self.output_path,
            prefix=f'{self.exp_id}_',
            suffix='.h5'
        )

        with h5py.File(self.video_path, 'w') as h5_file:
            h5_file.create_dataset(name='data', data=self.data)

        with h5py.File(self.artifact_path, 'w') as h5_file:
            h5_file.create_dataset(
                name=f'traces/{self.extract_roi["id"]}',
                data=np.random.random(self.data.shape[0])
            )

        self.args = {'video_path': self.video_path,
                     'roi_path': self.video_path,
                     'artifact_path': self.artifact_path,
                     'graph_path': self.video_path,
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
        self.assertEqual(len(output_file_list),
                         self.expected_num_artifact_files)

        for output_file, test_file in zip(output_file_list, self.test_files):
            image = Image.open(output_file)
            test = Image.open(test_file)
            np.testing.assert_array_equal(image.__array__(),
                                          test.__array__())

    def test_no_selected_roi(self):
        """Test that and ROI is not written when its id is not specified.
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
        classArtifacts._write_thumbnails(
            extract_roi=self.extract_roi,
            max_img=max_img,
            avg_img=avg_img,
            corr_img=corr_img,
            exp_id=self.exp_id)

        output_file_list = np.sort(glob(f'{self.output_path}/*.png'))
        self.assertEqual(len(output_file_list),
                         self.expected_num_artifact_files)

    def test_get_cutout_indices(self):
        """Test the correct indices of the unpadded cutout are found in the
        image.
        """
        classArtifacts = ClassifierArtifactsGenerator(
            args=[], input_data=self.args)

        half_cutout = self.args['cutout_size'] // 2
        im_size = 512

        # Low is clipped.
        center = 10
        indices = classArtifacts._get_cutout_indices(center, im_size)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[1], center + half_cutout)

        # High is clipped.
        center = 510
        indices = classArtifacts._get_cutout_indices(center, im_size)
        self.assertEqual(indices[0], center - half_cutout)
        self.assertEqual(indices[1], im_size)

        # Both not clipped.
        center = 128
        indices = classArtifacts._get_cutout_indices(128, im_size)
        self.assertEqual(indices[0], center - half_cutout)
        self.assertEqual(indices[1], center + half_cutout)

    def test_get_padding(self):
        """Test that the correct amount of padding for the cuttout is found.
        """
        classArtifacts = ClassifierArtifactsGenerator(
            args=[], input_data=self.args)
        half_cutout = self.args['cutout_size'] // 2
        im_size = 512

        # Test pad lowside.
        center = 4
        pads = classArtifacts._get_padding(center, 512)
        self.assertEqual(pads[0], half_cutout - center)
        self.assertEqual(pads[1], 0)

        # Test pad highside.
        center = 512 - 4
        pads = classArtifacts._get_padding(center, 512)
        self.assertEqual(pads[0], 0)
        self.assertEqual(pads[1], center - im_size + half_cutout)

        # Test pad highside.
        center = 32
        im_size = 64
        pads = classArtifacts._get_padding(center, im_size)
        self.assertEqual(pads[0], half_cutout - center)
        self.assertEqual(pads[1], half_cutout - center)

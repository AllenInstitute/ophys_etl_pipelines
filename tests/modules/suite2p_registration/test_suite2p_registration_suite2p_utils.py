import h5py
import json
import numpy as np
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch

from ophys_etl.modules.suite2p_registration.suite2p_utils import (  # noqa: E402, E501
    compute_reference, load_initial_frames, compute_acutance,
    add_required_parameters, create_ave_image, optimize_motion_parameters,
    remove_extrema_frames)


# Mock function to return just the first frame as the reference. We
# only specifically mock the functions we absolutely need to return
# data or those that prevent testing.
def pick_initial_reference_mock(data):
    return data[0, :, :]


class TestRegistrationSuite2pUtils(unittest.TestCase):

    def setUp(self):
        file_loc = Path(__file__)
        self.resource_loc = file_loc.parent / 'resources'
        self.h5_file_loc = self.resource_loc \
            / '792757260_test_data.h5'
        self.mock_data_loc = self.resource_loc \
            / 'test_rand.npy'
        with h5py.File(self.h5_file_loc, 'r') as h5_file:
            self.frames = h5_file['input_frames'][:]
            self.ops = json.loads(h5_file['ops'][()].decode('utf-8'))
            self.original_reference = h5_file['reference_img'][:]
            self.org_ave_image = h5_file['ave_image'][:]
        self.n_frames = self.frames.shape[0]
        self.xy_shape = self.frames.shape[1]
        self.rng = np.random.default_rng(1234)

    def test_load_initial_frames(self):
        """Test loading specific frames from h5py."""
        n_frames_to_load = 20
        loaded_data = load_initial_frames(file_path=self.h5_file_loc,
                                          h5py_key='input_frames',
                                          n_frames=n_frames_to_load)
        expected_frames = np.linspace(
            0, self.n_frames, n_frames_to_load + 1, dtype=int)[:-1]
        self.assertTrue(np.all(np.equal(loaded_data,
                                        self.frames[expected_frames, :, :])))

    def test_compute_reference_with_mocks(self):
        """Test our version of compute_reference without relying on suite2p
        by mocking key functions.

        This is a separate test from the one using suite2p as we want to test
        for changes to our code and changes in suite2p separately. When
        updating the compute_reference method in ophys_etl, you should expect
        to have to update the data in "test_rand.npy".
        """
        # Create random data with shape
        # (self.n_frames, self.xy_shape, self.xy_shape) and with random
        # fluctuations from -self.n_frames to self.n_frames.
        mock_frames = self.rng.integers(-self.n_frames,
                                        self.n_frames,
                                        size=(self.n_frames,
                                              self.xy_shape,
                                              self.xy_shape))

        # Mock function to pretend that phase correlation ran and found
        # offsets. Return arrays with the correct types and a range of random
        # values.
        def phasecorr_mock(data, cfRefImg, maxregshift, smooth_sigma_time):
            return (self.rng.integers(-1, 1, size=self.n_frames),
                    self.rng.integers(-1, 1, size=self.n_frames),
                    self.rng.uniform(-1, 1, size=self.n_frames))

        # Where to insert a NaN value in the reference image.
        nan_insert = 12

        # Return the input frame with no shift.
        def shift_frame_mock(frame, dy, dx):
            # If this is an unpadded frame (and therefore is the reference
            # image) add in a NaN value to test the code's ablity to catch NaN.
            if frame.shape[0] == self.xy_shape \
               and frame.shape[1] == self.xy_shape:
                frame[nan_insert, nan_insert] = np.nan
            return frame

        # We want this test to always run on Mocks even when suit2p is
        # present. Hence we mock functions to return data when needed and
        # MagicMock certain functions that are called within the code.
        with patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'pick_initial_reference',
                   pick_initial_reference_mock), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'phasecorr',
                   new=phasecorr_mock), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'shift_frame',
                   new=shift_frame_mock), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'apply_masks',
                   new=MagicMock()), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'compute_masks',
                   new=MagicMock()):
            # Run the code with only two iterations as there is no need for
            # more. We are just testing that the data passes through,
            # correctly.
            output = compute_reference(
                input_frames=mock_frames,
                niter=2,
                maxregshift=self.ops['maxregshift'],
                smooth_sigma=self.ops['smooth_sigma'],
                smooth_sigma_time=self.ops['smooth_sigma_time'])

        # Test that our output data is the correct shape and that it is
        # identical to the expected output.
        self.assertEqual(output.shape[0], self.n_frames)
        self.assertEqual(output.shape[1], self.n_frames)
        self.assertTrue(np.all(np.equal(output, np.load(self.mock_data_loc))))

    def test_remove_extrema_frames(self):
        """Test that empty frames are properly removed from the data.
        """
        # Create a bunch of frames with the same values to have a high
        # versus low mean comparison.
        frame_value = 100
        frame_scatter = 10
        test_data = self.rng.integers(low=frame_value - frame_scatter,
                                      high=frame_value + frame_scatter,
                                      size=(self.n_frames,
                                            self.xy_shape,
                                            self.xy_shape))
        # Create a well separated frame.
        test_data[10, :, :] = test_data[10, :, :] // frame_value
        frames = remove_extrema_frames(test_data)
        # Test that our different frame is removed.
        self.assertEqual(len(test_data) - 1, len(frames))
        # Test that the frame we set to a value of 1 for each pixel is
        # removed.
        self.assertTrue(np.all(frames > frame_scatter))

    def test_compute_reference_replace_nans(self):
        """Test that the code properly replaces NaN values in a reference
        image where frames have been pathologically shifted in such a way as to
        not cover the full reference image area.
        """
        low_value = 0
        high_value = 2
        mean_value = 1
        frames = np.zeros((4, 100, 100), dtype=np.int16)
        frames[0, :, :] = np.full_like(frames[0], high_value)

        # Mock function to create a reference image made of two frames
        # that are shifted exactly opposite of one another by one pixel,
        # creating an empty pixel in either corner of the reference image.
        def phasecorr_mock(data, cfRefImg, maxregshift, smooth_sigma_time):
            return (np.array([1, -1, 0, 0], dtype=int),
                    np.array([1, -1, 0, 0], dtype=int),
                    np.array([1, 1, 0, 0], dtype=float))

        # Function to shift frames.
        def shift_frame_mock(frame, dy, dx):
            return np.roll(frame, (dy, dx), axis=(0, 1))

        # Patch out all suite2p functions.
        with patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'pick_initial_reference',
                   pick_initial_reference_mock), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'phasecorr',
                   new=phasecorr_mock), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'shift_frame',
                   new=shift_frame_mock), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'apply_masks',
                   new=MagicMock()), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'compute_masks',
                   new=MagicMock()):
            result_reference = compute_reference(
                input_frames=frames,
                niter=1,
                maxregshift=self.ops['maxregshift'],
                smooth_sigma=self.ops['smooth_sigma'],
                smooth_sigma_time=self.ops['smooth_sigma_time'])
        # Test that all the pixels in the overlapping region are the correct
        # value.
        np.testing.assert_equal(result_reference[1:-1, 1:-1], mean_value)
        # Test that the non overlapping corners that have values retain the
        # correct value.
        self.assertEqual(result_reference[0, 0], low_value)
        self.assertEqual(result_reference[-1, -1], high_value)
        # Test that the empty corners of the image are correctly replaced
        # with the mean.
        self.assertEqual(result_reference[0, -1], mean_value)
        self.assertEqual(result_reference[-1, 0], mean_value)

    def test_compute_reference(self):
        """Test that the method creates a reference image as expected using
        suite2p.

        If this test is failing and no changes have been made to the
        compute_reference function in ophys_etl, this likely means that
        suit2p's code for computing phase correlations has changed and should
        be investigated. Like test_compute_reference_with_mocks, changes to
        compute_reference or suite2p will result in expected failures and
        will require the updating of the data in 792757260_test_data.h5.
        """
        # These test data were created with 8 iterations and is the current
        # suite2p "magic number" setting.
        result_reference = compute_reference(
            input_frames=self.frames,
            niter=8,
            maxregshift=self.ops['maxregshift'],
            smooth_sigma=self.ops['smooth_sigma'],
            smooth_sigma_time=self.ops['smooth_sigma_time'])

        # These are integer arrays so they should be identical in value.
        self.assertTrue(np.all(np.equal(self.original_reference,
                                        result_reference)))

    def test_optimize_motion_parameters(self):
        """
        """
        # Set parameters for the pipeline.
        suite2p_args = {'h5py': self.h5_file_loc,
                        'h5py_key': 'input_frames',
                        'maxregshift': 0.2,
                        'smooth_sigma_min': 0.65,
                        'smooth_sigma_max': 1.15,
                        'smooth_sigma_steps': 2,
                        'smooth_sigma_time_min': 0.0,
                        'smooth_sigma_time_max': 1.0,
                        'smooth_sigma_time_steps': 2}
        smooth_sigmas = np.linspace(suite2p_args['smooth_sigma_min'],
                                    suite2p_args['smooth_sigma_max'],
                                    suite2p_args['smooth_sigma_steps'])
        smooth_sigma_times = np.linspace(
            suite2p_args['smooth_sigma_time_min'],
            suite2p_args['smooth_sigma_time_max'],
            suite2p_args['smooth_sigma_time_steps'])

        def create_ave_image_mock(ref_image,
                                  suite2p_args,
                                  trim_frames_start=0,
                                  trim_frames_end=0,
                                  batch_size=500):
            image = np.zeros((self.xy_shape, self.xy_shape), dtype=int)
            if suite2p_args['smooth_sigma'] > 0.65 and \
               suite2p_args['smooth_sigma_time'] > 0.0:
                image[:50, :] = 10
                return {'ave_image': image,
                        'dy_max': 0,
                        'dx_max': 0}
            else:
                return {'ave_image': image,
                        'dy_max': 10,
                        'dx_max': 10}

        with patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'compute_reference', Mock), \
             patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'create_ave_image', create_ave_image_mock):
            best_result = optimize_motion_parameters(
                initial_frames=self.frames,
                smooth_sigmas=smooth_sigmas,
                smooth_sigma_times=smooth_sigma_times,
                suite2p_args=suite2p_args,
                logger=print)
        self.assertAlmostEqual(best_result['acutance'], 1.0)
        self.assertAlmostEqual(best_result['smooth_sigma'], 1.15)
        self.assertAlmostEqual(best_result['smooth_sigma_time'], 1.0)

    def test_create_ave_image_mock(self):
        """Run test data through to assert that the average image is
        created as expected.
        """
        suite2p_args = {'h5py': self.h5_file_loc,
                        'h5py_key': 'input_frames',
                        'smooth_sigma': 1.15,
                        'smooth_sigma_time': 1.0}
        add_required_parameters(suite2p_args)
        mock_frames = np.ones((self.n_frames,
                               self.xy_shape,
                               self.xy_shape),
                              dtype=int)
        max_shfit = 10

        def register_frames_mock(refAndMasks, frames, ops):
            return (mock_frames,
                    self.rng.integers(-max_shfit,
                                      max_shfit,
                                      size=self.n_frames),
                    self.rng.integers(-max_shfit,
                                      max_shfit,
                                      size=self.n_frames),
                    np.arange(self.n_frames),
                    np.arange(self.n_frames),
                    np.arange(self.n_frames),
                    np.arange(self.n_frames))

        with patch('ophys_etl.modules.suite2p_registration.suite2p_utils.'
                   'register_frames', register_frames_mock):
            result = create_ave_image(
                ref_image=self.original_reference,
                suite2p_args=suite2p_args)
        self.assertTrue(np.allclose(result['ave_image'],
                                    np.ones((self.xy_shape,
                                             self.xy_shape))))
        self.assertEqual(result['dy_max'], 10)
        self.assertEqual(result['dx_max'], 10)

    def test_create_ave_image(self):
        """Run test data through to assert that the average image is
        created as expected.
        """
        suite2p_args = {'h5py': self.h5_file_loc,
                        'h5py_key': 'input_frames',
                        'maxregshift': 0.2,
                        'smooth_sigma': 1.15,
                        'smooth_sigma_time': 1.0}
        add_required_parameters(suite2p_args)
        result = create_ave_image(
            ref_image=self.original_reference,
            suite2p_args=suite2p_args)
        np.testing.assert_allclose(result['ave_image'], self.org_ave_image)
        self.assertEqual(result['dy_max'], 20)
        self.assertEqual(result['dx_max'], 20)

    def test_add_required_parameters(self):
        """Test adding to config parameters to the dict.
        """
        suite2p_args = {'smooth_sigma': 1.15}
        # Test that parameters are added correct.
        add_required_parameters(suite2p_args)
        self.assertFalse(suite2p_args['1Preg'])
        self.assertFalse(suite2p_args['bidiphase'])
        self.assertFalse(suite2p_args['nonrigid'])
        self.assertTrue(suite2p_args['norm_frames'])

        # Test that the code doesn't change values already there.
        suite2p_args['1Preg'] = True
        suite2p_args['bidiphase'] = True
        suite2p_args['nonrigid'] = True
        suite2p_args['norm_frames'] = False
        add_required_parameters(suite2p_args)
        self.assertTrue(suite2p_args['1Preg'])
        self.assertTrue(suite2p_args['bidiphase'])
        self.assertTrue(suite2p_args['nonrigid'])
        self.assertFalse(suite2p_args['norm_frames'])

    def test_acutance_calculation(self):
        """Test for consistent results from the acutance calculation.
        """
        # Fill an image with half zeros and half the value of 10 to give an
        # image with a strong edge.
        mock_image = np.zeros((100, 100), dtype=int)
        mock_image[:50, :] = 10
        acut = compute_acutance(mock_image)
        self.assertAlmostEqual(acut, 1.0)

        acut = compute_acutance(mock_image, 0, 10)
        self.assertAlmostEqual(acut, 1.0)

        acut = compute_acutance(mock_image, 10, 0)
        self.assertAlmostEqual(acut, 1.25)

        acut = compute_acutance(mock_image, 10, 10)
        self.assertAlmostEqual(acut, 1.25)

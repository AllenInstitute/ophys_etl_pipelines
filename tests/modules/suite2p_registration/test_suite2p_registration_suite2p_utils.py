import h5py
import json
import numpy as np
from pathlib import Path
import sys
import pytest
import unittest
from unittest.mock import MagicMock, Mock, patch

has_suite2p = True
try:
    import suite2p.registration  # noqa: F401
except ImportError:
    # only mock Suite2P if necessary; otherwise, the mock
    # makes it into the tests that actually rely on Suite2P
    has_suite2p = False
    sys.modules['suite2p.registration.rigid'] = Mock()
    sys.modules['suite2p.registration.register'] = Mock()


from ophys_etl.modules.suite2p_registration.suite2p_utils import (  # noqa: E402, E501
    compute_reference, load_initial_frames)


class TestRegistrationSuite2pUtils(unittest.TestCase):

    def setUp(self):
        file_loc = Path(__file__)
        self.resource_loc = file_loc.parent / "resources"
        self.h5_file_loc = self.resource_loc \
            / "792757260_test_data.h5"
        self.mock_data_loc = self.resource_loc \
            / "test_rand.npy"
        with h5py.File(self.h5_file_loc, 'r') as h5_file:
            self.frames = h5_file["input_frames"][:]
            self.ops = json.loads(h5_file['ops'][()].decode('utf-8'))
            self.original_reference = h5_file["reference_img"][:]
        self.n_frames = self.frames.shape[0]
        self.rng = np.random.default_rng(1234)

    def test_load_initial_frames(self):
        """Test loading specific frames from h5py."""
        n_frames_to_load = 20
        loaded_data = load_initial_frames(self.h5_file_loc,
                                          "input_frames",
                                          n_frames_to_load)
        expected_frames = np.linspace(
            0, self.n_frames, n_frames_to_load + 1, dtype=int)[:-1]
        self.assertTrue(np.all(np.equal(loaded_data,
                                        self.frames[expected_frames, :, :])))

    def test_compute_reference_with_mocks(self):
        """Test our version of compute_reference without relying on suite2p
        by mocking key functions.
        """
        # Create random data with shape
        # (self.n_frames, self.n_frames, self.n_frames) and with random
        # fluctuations from -self.n_frames to self.n_frames.
        mock_frames = self.rng.integers(-self.n_frames,
                                        self.n_frames,
                                        size=(self.n_frames,
                                              self.n_frames,
                                              self.n_frames))

        # Mock function to return just the first frame as the reference. We
        # only specifically mock the functions we absolutely need to return
        # data or those that prevent testing.
        def pick_initial_reference_mock(data):
            return data[0, :, :]

        # Mock function to pretend that phase correlation ran and found
        # offsets. Return arrays with the correct types and a range of random
        # values.
        def phasecorr_mock(data, cfRefImg, maxregshift, smooth_sigma_time):
            return (self.rng.integers(-1, 1, size=self.n_frames),
                    self.rng.integers(-1, 1, size=self.n_frames),
                    self.rng.uniform(-1, 1, size=self.n_frames))

        # Return the input frame with no shift.
        def shift_frame(frame, dy, dx):
            return frame

        # We want this test to always run on Mocks even when suit2p is
        # present. Hence we mock functions to return data when needed and
        # MagicMock certain functions that are called within the code.
        with patch("ophys_etl.modules.suite2p_registration.suite2p_utils."
                   "pick_initial_reference",
                   pick_initial_reference_mock), \
             patch("ophys_etl.modules.suite2p_registration.suite2p_utils."
                   "phasecorr",
                   new=phasecorr_mock), \
             patch("ophys_etl.modules.suite2p_registration.suite2p_utils."
                   "shift_frame",
                   new=shift_frame), \
             patch("ophys_etl.modules.suite2p_registration.suite2p_utils."
                   "apply_masks",
                   new=MagicMock()), \
             patch("ophys_etl.modules.suite2p_registration.suite2p_utils."
                   "compute_masks",
                   new=MagicMock()):
            # Run the code with only two iterations as there is no need for
            # more. We are just testing that the data passes through,
            # correctly.
            output = compute_reference(
                frames=mock_frames,
                niter=2,
                maxregshift=self.ops["maxregshift"],
                smooth_sigma=self.ops["smooth_sigma"],
                smooth_sigma_time=self.ops["smooth_sigma_time"])

        # Test that our output data is the correct shape and that it is
        # identical to the expected output.
        self.assertEqual(output.shape[0], self.n_frames)
        self.assertEqual(output.shape[1], self.n_frames)
        self.assertTrue(np.all(np.equal(output, np.load(self.mock_data_loc))))

    @pytest.mark.suite2p_only
    def test_compute_reference(self):
        """Test that the method creates a reference image as expected using
        suite2p."""
        # These test data were created with 8 iterations and is the current
        # suite2p "magic number" setting.
        result_reference = compute_reference(
            frames=self.frames,
            niter=8,
            maxregshift=self.ops["maxregshift"],
            smooth_sigma=self.ops["smooth_sigma"],
            smooth_sigma_time=self.ops["smooth_sigma_time"])

        # These are integer arrays so they should be identical in value.
        self.assertTrue(np.all(np.equal(self.original_reference,
                                        result_reference)))

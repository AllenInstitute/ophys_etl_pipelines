import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np

from ophys_etl.modules.demix.__main__ import DemixJob


class TestDemixJob(TestCase):
    def setUp(self):
        self.movie_data = np.random.rand(100, 512, 512)
        self.traces_data = np.random.rand(3, 100)
        self.roi_names = np.array([b"111", b"222", b"333"])
        test_path = Path(__file__).parent.absolute() / "test_data"
        with open(test_path / "roi_masks.json", "r") as f:
            self.roi_masks = json.load(f)["roi_masks"]

    def create_files(self, tmp_path):
        movie_h5 = os.path.join(tmp_path, "mock_movie.h5")
        traces_h5 = os.path.join(tmp_path, "mock_traces.h5")
        output_json = os.path.join(tmp_path, "mock_output.json")
        output_h5 = os.path.join(tmp_path, "mock_output.h5")

        with h5py.File(movie_h5, "w") as f:
            f.create_dataset("data", data=self.movie_data)

        with h5py.File(traces_h5, "w") as f:
            f.create_dataset("data", data=self.traces_data)
            f.create_dataset("roi_names", data=self.roi_names)

        return movie_h5, traces_h5, output_json, output_h5

    def delete_files(self, tmp_path):
        movie_h5 = os.path.join(tmp_path, "mock_movie.h5")
        traces_h5 = os.path.join(tmp_path, "mock_traces.h5")
        output_json = os.path.join(tmp_path, "mock_output.json")
        output_h5 = os.path.join(tmp_path, "mock_output.h5")
        os.remove(movie_h5)
        os.remove(traces_h5)
        if os.path.exists(output_json):
            os.remove(output_json)
        if os.path.exists(output_h5):
            os.remove(output_h5)

    def test_run(self):
        with TemporaryDirectory() as tmp_dir, patch(
            "ophys_etl.modules.demix.demixer.demix_time_dep_masks"
        ) as mock_demix_time_dep_masks, patch(
            "ophys_etl.modules.demix.demixer.plot_negative_transients"
        ) as mock_plot_negative_transients, patch(
            "ophys_etl.modules.demix.demixer.plot_negative_baselines"
        ) as mock_plot_negative_baselines:

            tmp_path = Path(tmp_dir)
            # Set up mock data and functions
            movie_h5, traces_h5, output_json, output_h5 = self.create_files(
                tmp_path
            )

            mock_demix_time_dep_masks.return_value = (self.traces_data * 2, [])
            mock_plot_negative_transients.return_value = []
            mock_plot_negative_baselines.return_value = []

            # Run the DemixJob
            args = {
                "movie_h5": movie_h5,
                "traces_h5": traces_h5,
                "output_file": output_h5,
                "output_json": output_json,
                "roi_masks": self.roi_masks,
                "exclude_labels": [],
            }
            demix_job = DemixJob(input_data=args, args=[])
            demix_job.run()

            # Check that the mock functions are called once
            mock_demix_time_dep_masks.assert_called_once()
            mock_plot_negative_transients.assert_called_once()
            mock_plot_negative_baselines.assert_called_once()

            # Check if the output_h5 file contains the correct datasets
            with h5py.File(output_h5, "r") as f:
                self.assertIn("data", f.keys())
                self.assertIn("roi_names", f.keys())

                data = f["data"][()]
                roi_names = f["roi_names"][()]

                np.testing.assert_array_equal(data, self.traces_data * 2)
                np.testing.assert_array_equal(roi_names, self.roi_names)

            # Check if the output_json file contains the correct data
            with open(output_json, "r") as f:
                output_data = json.load(f)

                self.assertIn("negative_transient_roi_ids", output_data)
                self.assertIn("negative_baseline_roi_ids", output_data)

                self.assertEqual(output_data["negative_transient_roi_ids"], [])
                self.assertEqual(output_data["negative_baseline_roi_ids"], [])

            self.delete_files

import argparse
import h5py
import json
from ophys_etl.modules.segmentation.graph_utils.conversion import graph_to_img
from pathlib import Path


if __name__ == "__main__":
    with open("/allen/aibs/informatics/chris.morrison/ticket-504/"
              "experiment_metadata.json", 'r') as jfile:
        experiment_meta = json.load(jfile)

    base_path = Path("/allen/programs/mindscope/workgroups/surround/"
                     f"denoising_labeling_2022/denoised_movies/")

    for experiment_id in experiment_meta.keys():
        print(f"Running {experiment_id}...")
        corr_graph_path = (base_path / f"{experiment_id}" /
                           f"{experiment_id}_correlation_graph.pkl")
        corr_image = graph_to_img(corr_graph_path)
        output_path = (base_path / f"{experiment_id}" /
                       f"{experiment_id}_corr.h5")
        if output_path.exists():
            print("File exists, continuing...")
        with h5py.File(output_path, 'w') as h5:
            h5.create_dataset(name='corr_projection', data=corr_image)
    print("Done!")

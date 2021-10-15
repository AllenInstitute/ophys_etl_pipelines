import os
from pathlib import Path

import pandas as pd


def get_experiment_genotype_map():
    root = os.path.dirname(os.path.abspath(__file__))
    path = Path(root) / 'deepcell_data/ophys_metadata_lookup.txt'
    experiment_metadata = pd.read_csv(path)
    experiment_metadata.columns = [c.strip() for c in experiment_metadata.columns]
    return experiment_metadata[['experiment_id', 'genotype']].set_index('experiment_id') \
        .to_dict()['genotype']

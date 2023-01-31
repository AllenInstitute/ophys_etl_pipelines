import tempfile
from pathlib import Path
from typing import Dict

import argschema
import h5py
import numpy as np
from deepinterpolation.cli.fine_tuning import FineTuning

from ophys_etl.modules.denoising.fine_tuning.schemas import \
    FineTuningInputSchemaPreDataSplit, DataSplitterOutputSchema
from ophys_etl.modules.denoising.fine_tuning.utils.data_splitter import \
    DataSplitter


class FinetuningRunner(argschema.ArgSchemaParser):
    """Runs finetuning of a base model on a single movie.

    Wrapper around `FineTuning` interface,
    this also splits the data into train/val splits before passing it to
    that interface"""
    default_schema = FineTuningInputSchemaPreDataSplit
    args: Dict

    def run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_out_path = Path(tmp_dir) / 'train.json'
            val_out_path = Path(tmp_dir) / 'val.json'

            self._write_train_val_datasets(
                train_out_path=train_out_path,
                val_out_path=val_out_path
            )
            self.args['generator_params']['data_path'] = str(train_out_path)
            self.args['test_generator_params']['data_path'] = str(val_out_path)

            del self.args['data_split_params']

            # Removes args added in post_load, since they are not expected in
            # the schema when `FineTuning` is called below
            del self.args['generator_params']['steps_per_epoch']
            del self.args['test_generator_params']['steps_per_epoch']

            fine_tuning_runner = FineTuning(input_data=self.args, args=[])
            fine_tuning_runner.run()

    def _write_train_val_datasets(
            self,
            train_out_path: Path,
            val_out_path: Path
    ):
        """Writes the train/val jsons to disk. Needed for the Finetuning CLI

        Parameters
        ----------
        train_out_path
            Where to write train dataset
        val_out_path
            Where to write val dataset
        """
        data_splitter = DataSplitter(
            movie_path=self.args['data_split_params']['movie_path'],
            seed=self.args['data_split_params']['seed']
        )
        train, val = data_splitter.get_train_val_split(
            train_frac=self.args['data_split_params']['train_frac'],
            window_size=(self.args['generator_params']['pre_frame'] +
                         self.args['generator_params']['post_frame'])
        )

        with h5py.File(self.args['data_split_params']['movie_path']) as f:
            # Only evaluating mean, std on train set to not leak any signal
            # to validation
            mean = f['data'][np.sort(train)].mean()
            std = f['data'][np.sort(train)].std()

        for ds_out_path, ds in zip((train_out_path, val_out_path),
                                   (train, val)):
            with open(ds_out_path, 'w') as f:
                f.write(DataSplitterOutputSchema().dumps({
                    'mean': mean,
                    'std': std,
                    'path': self.args['data_split_params']['movie_path'],
                    'frames': list(ds)
                }, indent=2))


def main():
    fine_tuning_runner = FinetuningRunner()
    fine_tuning_runner.run()


if __name__ == '__main__':
    main()

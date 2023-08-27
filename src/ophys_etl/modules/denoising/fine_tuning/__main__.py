import logging
import os
import signal
import sys
from pathlib import Path

import argschema
import h5py
import json
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
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
            kwargs to sent to `ArgSchemaParser` constructor
        """
        super().__init__(
            schema_type=FineTuningInputSchemaPreDataSplit,
            **kwargs
        )

        # this removes the logger set by `ArgSchemaParser`. We want to add
        # a timestamp to the logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=self.logger.level,
            stream=sys.stdout
        )
        logger = logging.getLogger(type(self).__name__)
        logger.setLevel(level=self.logger.level)
        self.logger = logger

    def run(self):
        train_out_path = Path(
            self.args['data_split_params']['dataset_output_dir']) / \
            'train.json'
        val_out_path = Path(
            self.args['data_split_params']['dataset_output_dir']) / 'val.json'

        self._write_train_val_datasets(
            train_out_path=train_out_path,
            val_out_path=val_out_path
        )
        self.args['generator_params']['data_path'] = str(train_out_path)
        self.args['test_generator_params']['data_path'] = str(val_out_path)

        del self.args['data_split_params']

        # Removes args added, since they are not expected in
        # the schema when `FineTuning` is called below
        del self.args['generator_params']['steps_per_epoch']
        del self.args['test_generator_params']['steps_per_epoch']
        manually_kill_process = self.args.pop('manually_kill_process')

        fine_tuning_runner = FineTuning(input_data=self.args, args=[])
        fine_tuning_runner.run()

        if manually_kill_process:
            # Manually killing process due to an issue with hanging processes
            os.kill(os.getpid(), signal.SIGTERM)

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
            seed=self.args['data_split_params']['seed'],
            downsample_frac=self.args['data_split_params']['downsample_frac']
        )
        train, val = data_splitter.get_train_val_split(
            train_frac=self.args['data_split_params']['train_frac'],
            window_size=(self.args['generator_params']['pre_frame'] +
                         self.args['generator_params']['post_frame'])
        )

        with h5py.File(self.args['data_split_params']['movie_path']) as f:
            # Only evaluating mean, std on train set to not leak any signal
            # to validation
            self.logger.info('Calculating train movie statistics')
            mov = f['data'][()]
            mean = mov[train].mean()
            std = mov[train].std()

        for ds_out_path, ds, ds_name in zip(
                (train_out_path, val_out_path),
                (train, val),
                ('train', 'val')
        ):
            out = {
                self.args['data_split_params']['ophys_experiment_id']:
                    DataSplitterOutputSchema().load({
                        'mean': mean,
                        'std': std,
                        'path': self.args['data_split_params']['movie_path'],
                        'frames': list(ds)
                    })
            }
            with open(ds_out_path, 'w') as f:
                f.write(json.dumps(out, indent=2))
            self.logger.info(f'Wrote {ds_name} set to {ds_out_path}')


def main():
    fine_tuning_runner = FinetuningRunner()
    fine_tuning_runner.run()


if __name__ == '__main__':
    main()

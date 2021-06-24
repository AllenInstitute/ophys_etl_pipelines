import argschema
import h5py
import numpy as np
from multiprocessing.pool import ThreadPool
from functools import partial

from ophys_etl.modules.segmentation.denoise.averaging import \
    temporal_filter1d
from ophys_etl.modules.segmentation.modules.schemas import \
    SimpleDenoiseInputSchema


class SimpleDenoise(argschema.ArgSchemaParser):
    default_schema = SimpleDenoiseInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        self.logger.info(f"filtering {self.args['video_path']}")
        with h5py.File(self.args["video_path"], "r") as f:
            data = f["data"][()]

        trace_filter = partial(temporal_filter1d,
                               size=self.args["size"],
                               filter_type=self.args["filter_type"])

        if self.args["n_parallel_workers"] == 1:
            output = trace_filter(data)
        else:
            self.logger.info("splitting movie across "
                             f"{self.args['n_parallel_workers']} workers")
            dshape = data.shape
            data = data.reshape(dshape[0], -1)
            indices = np.arange(data.shape[1])
            split_indices = np.array_split(indices,
                                           self.args["n_parallel_workers"])
            with ThreadPool(self.args["n_parallel_workers"]) as pool:
                results = pool.map(trace_filter,
                                   [data[:, i] for i in split_indices])
            output = np.concatenate(results, axis=1)
            output = output.reshape(*dshape)

        with h5py.File(self.args["video_output"], "w") as f:
            f.create_dataset("data",
                             data=output,
                             chunks=self.args["h5_chunk_shape"])
        self.logger.info(f"wrote {self.args['video_output']}")


if __name__ == "__main__":
    sd = SimpleDenoise()
    sd.run()

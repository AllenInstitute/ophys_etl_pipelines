import argschema
import functools
import math
import time
import h5py
import numpy as np
from multiprocessing import Pool

from ophys_etl.modules.sine_dewarp import utils
from ophys_etl.modules.sine_dewarp.schemas import SineDewarpInputSchema


class SineDewarp(argschema.ArgSchemaParser):
    default_schema = SineDewarpInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        # read the movie into memory
        with h5py.File(self.args["input_h5"], "r") as input_file:
            movie = input_file['data'][()]
        movie_shape = movie.shape
        movie_dtype = movie.dtype
        T, y_orig, x_orig = movie.shape
        self.logger.info(f"read {self.args['input_h5']} with shape "
                         f"{movie_shape}")

        # prepare for writing the output movie
        xtable = utils.create_xtable(movie,
                                     self.args["aL"],
                                     self.args["aR"],
                                     self.args["bL"],
                                     self.args["bR"],
                                     self.args["noise_reduction"])
        self.logger.info("created xtable")
        utils.make_output_file(self.args["output_h5"],
                               "data",
                               xtable,
                               self.args["FOV_width"],
                               movie_shape,
                               movie_dtype)

        fn = functools.partial(
                utils.xdewarp,
                FOVwidth=self.args["FOV_width"],
                xtable=xtable,
                noise_reduction=self.args["noise_reduction"])

        # write the output movie
        start_time = time.time()
        with Pool(self.args["n_parallel_workers"]) as pool, \
                h5py.File(self.args["output_h5"], "a") as f:
            # Chunk the movie up to prevent an error caused by multiprocessing
            # returning too much data. Use math.ceil in case there are fewer
            # than chunk_size frame, then we just make one chunk.
            chunk_size = self.args["chunk_size"]
            movie_chunks = np.array_split(
                movie, math.ceil(T / chunk_size), axis=0
            )

            # Use multiprocessing to dewarp one chunk of the movie at a time
            dewarped_so_far = 0
            for movie_chunk in movie_chunks:
                for frame_num, frame in enumerate(pool.map(fn, movie_chunk)):
                    f["data"][dewarped_so_far + frame_num, :, :] = \
                            frame
                dewarped_so_far = dewarped_so_far + len(movie_chunk)

        end_time = time.time()
        self.logger.info(f"Elapsed time (s): {end_time - start_time}")

        self.output(dict())


if __name__ == '__main__':
    smod = SineDewarp()
    smod.run()

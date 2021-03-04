import argparse
import functools
import logging
import math
import multiprocessing
import time
import h5py
import json
import numpy as np

from ophys_etl.modules.sine_dewarp import utils

valid_nr_methods = [0, 1, 2, 3]


def run_dewarping(FOVwidth: int,
                  noise_reduction: int,
                  threads: int,
                  input_file: str,
                  input_dataset: str,
                  output_file: str,
                  output_dataset: str,
                  aL: float,
                  aR: float,
                  bL: float,
                  bR: float) -> None:
    """
    Dewarp the movie frame by frame, utilizing multiprocessing (along with
    manual chunking of the movie beforehand) to increase efficiency.

    Parameters
    ----------
    FOVwidth: int
        Field of View width. Will be the width of the output image, unless 0,
        then it will be the width of the input image.
    noise_reduction: int
        The noise reduction method that will be used in dewarping.
    threads: int
        The number of worker threads to use in multiprocessing.
    input_file: str
        File path where the input h5 is saved.
    input_dataset: str
        The name of the dataset representing the movie within the
        input h5 file.
    output_file: str
        File path where the h5 containing the dewarped video will be saved.
    output_dataset: str,
        The name of the dataset representing the movie within the
        output h5 file.
    aL: float
        How far into the image (from the left side inward, in pixels) the
        warping occurs. This is specific to the experiment, and is known
        at the time of data collection.
    aR: float
        How far into the image (from the right side inward, in pixels) the
        warping occurs. This is specific to the experiment, and is known
        at the time of data collection.
    bL: float
        Roughly, a measurement of how strong the warping effect was on
        the left side for the given experiment.
    bR: float
        Roughly, a measurement of how strong the warping effect was on
        the right side for the given experiment.
    Returns
    -------
    """

    # Get statistics about the movie
    input_h5_file = h5py.File(input_file, 'r')
    movie = input_h5_file[input_dataset]

    movie_shape = movie.shape
    movie_dtype = movie.dtype
    T, y_orig, x_orig = movie.shape

    xtable = utils.create_xtable(movie, aL, aR, bL, bR, noise_reduction)

    utils.make_output_file(output_file, output_dataset, xtable,
                           FOVwidth, movie_shape, movie_dtype)

    start_time = time.time()
    with multiprocessing.Pool(threads) as pool, \
         h5py.File(output_file, "a") as f:

        fn = functools.partial(
            utils.xdewarp,
            FOVwidth=FOVwidth,
            xtable=xtable,
            noise_reduction=noise_reduction
        )

        # Chunk the movie up to prevent an error caused by multiprocessing
        # returning too much data. Use math.ceil in case there are fewer
        # than chunk_size frame, then we just make one chunk.
        chunk_size = 1000
        movie_chunks = np.array_split(
            movie, math.ceil(T / chunk_size), axis=0
        )

        # Use multiprocessing to dewarp one chunk of the movie at a time
        dewarped_so_far = 0
        for movie_chunk in movie_chunks:
            for frame_num, frame in enumerate(pool.map(fn, movie_chunk)):
                f[output_dataset][dewarped_so_far + frame_num, :, :] = frame

            dewarped_so_far = dewarped_so_far + len(movie_chunk)

    input_h5_file.close()

    end_time = time.time()
    logging.debug(f"Elapsed time (s): {end_time - start_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_json')
    parser.add_argument('--log_level', default=logging.DEBUG)
    parser.add_argument('--threads', default=4, type=int)
    parser.add_argument('--FOVwidth', default=0, type=int)
    parser.add_argument('--noise_reduction', default=0, type=int)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if args.noise_reduction not in valid_nr_methods:
        raise(
            f"{args.noise_reduction} is not a valid noise "
            f"reduction option. Must be one of {valid_nr_methods}. "
            f"Using default value noise_reduction = 0.\n",
            UserWarning
        )
    else:
        logging.debug(f"noise_reduction: {args.noise_reduction}")

    with open(args.input_json, 'r') as json_file:
        input_data = json.load(json_file)
    input_h5, output_h5, aL, aR, bL, bR = utils.parse_input(input_data)

    run_dewarping(
        FOVwidth=args.FOVwidth,
        noise_reduction=args.noise_reduction,
        threads=args.threads,
        input_file=input_h5,
        input_dataset="data",
        output_file=output_h5,
        output_dataset="data",
        aL=aL, aR=aR, bL=bL, bR=bR
    )

    with open(args.output_json, 'w') as outfile:
        json.dump({}, outfile)

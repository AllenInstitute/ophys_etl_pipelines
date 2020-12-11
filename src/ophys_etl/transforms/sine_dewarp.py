import argparse
import functools
import logging
import math
import multiprocessing
from pathlib import Path
import time
from typing_extensions import TypedDict

import h5py
import json
import numpy as np
from scipy.stats import mode

valid_nr_methods = [0, 1, 2, 3]


class XTable(TypedDict):
    mean_modevalue: float
    mean_maxcount: float
    mean_minvalue: float
    stdv_modevalue: float
    stdv_maxcount: float
    stdv_minvalue: float
    bgfactorL: float
    bgfactorR: float
    left_margin: int
    right_margin: int
    xindexL: np.ndarray
    xindexLB: np.ndarray
    xindexR: np.ndarray
    xindexRB: np.ndarray
    aL: float
    aR: float
    bL: float
    bR: float


def noise_reduce(data: np.ndarray,
                 xtable: XTable,
                 ind: int,
                 xindex: np.array,
                 noise_reduction: int) -> np.ndarray:
    """
    Noise reduce a numpy array by one of several methods, specified by
    the noise_reduction argument. The current acceptable values are:
    - 1: Reduce the data value by 2 * sigma
    - 2: Reduce the data value by sigma and also normalize it
    - 3: Normalize the data value
    - Else: No noise reduction, input data returned unchanged

    Parameters
    ----------
    data: np.ndarray
        The data that will be noise reduced.
    xtable: XTable
        A dict containing aggregate information about the input video
        to be used in the dewarping process.
    ind: int
        The column currently being noise reduced.
    xindex: np.array
        Array of indices of the input image to be used for the output image.
    noise_reduction: int
        The noise reduction method to use.
    Returns
    -------
    data: np.ndarray
        The data, of the sme shape as the input data, having had the
        noise reduction process applied to it
    """

    if 1 == noise_reduction:
        nf = (
            2 * xtable['stdv_modevalue']
            * (xindex[ind] - xindex[ind - 1] - 1)
        )

        data = data - nf

    elif 2 == noise_reduction:
        nf = xtable['stdv_modevalue'] * (xindex[ind] - xindex[ind - 1] - 1)
        data = (data / (xindex[ind] - xindex[ind - 1])) - nf

    elif 3 == noise_reduction:
        data = data / (xindex[ind] - xindex[ind - 1])

    else:
        pass

    return data


def xdewarp(imgin: np.ndarray,
            FOVwidth: int,
            xtable: XTable,
            noise_reduction: int) -> np.ndarray:
    """
    Dewarp a single numpy array based on the information
    specified in xtable and noise_reduction.

    Parameters
    ----------
    imgin: np.ndarray
        The image that wil have the dewarping process applied to it.
    FOVwidth: int
        Field of View width. Will be the width of the output image, unless 0,
        then it will be the width of the input image.
    xtable: XTable
        A dict containing aggregate information about the input video
        to be used in the dewarping process.
    noise_reduction: int
        The noise reduction method to use.
    Returns
    -------
    img: np.ndarray
        The data, of the same shape as the input data, having had the
        dewarping process applied to it
    """

    maxlevel = 65535

    # Grab a few commonly used values from xtable to make code cleaner
    xindexL = xtable['xindexL']
    xindexLB = xtable['xindexLB']
    xindexR = xtable['xindexR']
    xindexRB = xtable['xindexRB']

    # Prepare a blank image
    imgout = np.zeros(imgin.shape)
    imgout[:, (int(xtable['aL'])):(512-(int(xtable['aR'])))] = \
        imgin[:, (int(xtable['aL'])):(512-(int(xtable['aR'])))]

    col = np.zeros(imgin.shape[0], np.float)

    # Left side
    for j in range(0, int(xtable['aL'])):
        col[:] = 0.0  # reset
        if xindexL[j] >= 0:
            if xindexLB[j - 1] >= 0.0:
                s = int(math.floor(xindexLB[j - 1]))
                e = int(math.floor(xindexLB[j]))

                col[:] = (
                    (s + 1 - xindexLB[j - 1]) * imgin[:, s]
                    + (xindexLB[j] - e) * imgin[:, e]
                )

                if (e - s) > 1:   # have a middle pixel
                    col[:] = col[:] + imgin[:, s + 1]

                # Perform the desired noise reduction method
                col = noise_reduce(col, xtable, j, xindexLB, noise_reduction)

                low_mask = (col < 0)
                col[low_mask] = 0  # underflow?

                high_mask = (col > maxlevel)
                col[high_mask] = maxlevel  # saturated?  for max image==1.0

                imgout[:, j] = col
            else:
                imgout[:, j] = imgin[:, xindexL[j]] * xtable['bgfactorL']
        else:
            imgout[:, j] = xtable['mean_modevalue'] * xtable['bgfactorL']

    # Right side
    for j in range(0, int(xtable['aR'])):
        col[:] = 0.0
        if xindexR[j] >= 0:
            if xindexRB[j - 1] >= 0.0:
                s = int(math.floor(xindexRB[j - 1]))
                e = int(math.floor(xindexRB[j]))

                col[:] = (
                    (s + 1 - xindexRB[j - 1]) * imgin[:, 511 - s]
                    + (xindexRB[j] - e) * imgin[:, 511 - e]
                )

                if (e-s) > 1:  # have a middle pixel
                    col[:] = col[:] + imgin[:, 511 - (s + 1)]

                # Perform the desired noise reduction method
                col = noise_reduce(col, xtable, j, xindexRB, noise_reduction)

                low_mask = (col < 0)
                col[low_mask] = 0  # underflow?

                high_mask = (col > maxlevel)
                col[high_mask] = maxlevel  # saturated?  for max image==1.0

                imgout[:, 511 - j] = col
            else:
                imgout[:, 511 - j] = (
                    imgin[:, 511 - xindexR[j]]
                    * xtable['bgfactorR']
                )
        else:
            imgout[:, 511 - j] = xtable['mean_modevalue'] * xtable['bgfactorR']

    if FOVwidth == 512:
        img = imgout.astype(np.uint16)
    else:
        img = imgout[
            :, xtable['left_margin']:(512 - xtable['right_margin'])
        ].astype(np.uint16)

    return img


def get_xindex(warped_pixels: float, scale: float) -> np.ndarray:
    """
    Generate the xindex arrays that will be used in the dewarping process.

    Parameters
    ----------
    warped_pixels: float
        How far into the image (from the side inward, in pixels) the warping
        occurs. This is specific to the experiment, and is known at the
        time of data collection.
    scale: float
        Roughly, a measurement of how strong the warping effect was for
        the given experiment.
    Returns
    -------
    xindex: np.ndarray
        Array of indices of the input image to be used for the output image.
    xindexB: np.ndarray
        Array of indices of the input image to be used for the output image.
    """
    xindex = np.zeros(256, np.int)
    xindexB = np.zeros(256, np.float)  # between pixels

    for j in range(0, int(warped_pixels)):
        xindex[j] = (
            j - int(
                    (scale)
                    * (1.0 - math.sin(
                        (j/(warped_pixels * 3.0) + 1.0/6.0) * 3.14159265
                    ))
                    + 0.5
                )
        )

        xindexB[j] = (
            (j + 0.5) - (
                (scale) * (1.0 - math.sin(
                    ((j + 0.5)/(warped_pixels * 3.0) + 1.0 / 6.0) * 3.14159265
                ))
            )
        )

    return xindex, xindexB


def create_xtable(movie: np.ndarray,
                  aL: float,
                  aR: float,
                  bL: float,
                  bR: float,
                  noise_reduction: int) -> XTable:
    """
    Compute a number of statistics about the images to be dewarped.

    From comment: The bgfactor variables are only relevant in the case where
    no noise reduction will be performed. This is because they are only used
    on the first handful of columns (where the xindex arrays are negative).
    Also, once those arrays are not negative, dewarping works by creating
    a new column that is actually a combination of columns from the input
    image. This means that suddenly the new column will have higher values
    than the column next to it, and hence will be much brighter. When noise
    reduction is performed, the values of the new column will be reduced,
    causing the brightness of that column to fall back to a similar range as
    the previous column. But when noise reduction isn't performed, we need
    to take care of this problem in another way. The solution here is to
    brighten these early images by this particular factor.

    Parameters
    ----------
    movie: np.ndarray
        The movie that is being dewarped.
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
    noise_reduction: int
        The noise reduction method that will be used in dewarping.
    Returns
    -------
    xtable: XTable
        Various computed information about the video.
    """

    meanimg = np.mean(movie, axis=0)  # use avgimg to compute mode
    meanimg = meanimg.astype(int)

    # full movie got Memory Error
    stdvimg = np.std(movie[::8], axis=0)  # use stdvimg to compute mode
    stdvimg = stdvimg.astype(int)

    # Left side
    xindexL, xindexLB = get_xindex(aL, bL)

    # Right side
    xindexR, xindexRB = get_xindex(aR, bR)

    modelist, mean_maxcount = mode(meanimg)
    mean_modevalue = modelist[0]
    mean_minvalue = np.min(meanimg)

    modelist, stdv_maxcount = mode(stdvimg)
    stdv_modevalue = modelist[0]
    stdv_minvalue = np.min(stdvimg)

    j = 0
    while xindexLB[j] < 0.0 or xindexL[j] < 0:
        j = j + 1
    else:
        bgfactorL = xindexLB[j+1] - xindexLB[j]
        left_margin = j

    j = 0
    while xindexRB[j] < 0.0 or xindexR[j] < 0:
        j = j + 1
    else:
        bgfactorR = xindexRB[j+1] - xindexRB[j]
        right_margin = j

    # For an explanation, see the docstring
    if 0 != noise_reduction:
        bgfactorL = 1
        bgfactorR = 1

    table = XTable(
        mean_modevalue=mean_modevalue,
        mean_maxcount=mean_maxcount,
        mean_minvalue=mean_minvalue,
        stdv_modevalue=stdv_modevalue,
        stdv_maxcount=stdv_maxcount,
        stdv_minvalue=stdv_minvalue,
        bgfactorL=bgfactorL,
        bgfactorR=bgfactorR,
        left_margin=left_margin,
        right_margin=right_margin,
        xindexL=xindexL,
        xindexLB=xindexLB,
        xindexR=xindexR,
        xindexRB=xindexRB,
        aL=aL,
        aR=aR,
        bL=bL,
        bR=bR
    )
    logging.debug(table)

    return table


def parse_input(data: json) -> (str, str, float, float, float, float):
    """
    Compute a number of statistics about the images to be dewarped.

    Parameters
    ----------
    data: json
        Data from the input json file.
    Returns
    -------
    input_h5: str
        File path where the input h5 is saved.
    output_h5: str
        File path where the output h5 is saved.
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
    """

    input_h5 = data.get('input_h5', None)
    if input_h5 is None:
        raise KeyError("input JSON does not have required field: 'input_h5'")
    if not Path(input_h5).is_file():
        raise IOError(f"{input_h5} does not exist")

    output_h5 = data.get('output_h5', None)
    if output_h5 is None:
        raise KeyError("input JSON does not have required field: 'output_h5'")

    equipment_name = data.get("equipment_name", None)
    if equipment_name is None:
        raise KeyError("input JSON does not have"
                       "required field: 'equipment_name'")

    aLstr = data.get("aL", None)
    if aLstr is None:
        raise KeyError(f"parameter aL not determined for {equipment_name}")
    aL = float(aLstr)

    aRstr = data.get("aR", None)
    if aRstr is None:
        raise KeyError(f"parameter aR not determined for {equipment_name}")
    aR = float(aRstr)

    bLstr = data.get("bL", None)
    if bLstr is None:
        raise KeyError(f"parameter bL not determined for {equipment_name}")
    bL = float(bLstr)

    bRstr = data.get("bR", None)
    if bRstr is None:
        raise KeyError(f"parameter bR not determined for {equipment_name}")
    bR = float(bRstr)

    logging.debug(f"Equipment name: {str(equipment_name)}")
    logging.debug(f"aL: {aL}   aR: {aR}")
    logging.debug(f"bL: {bL}   bR: {bR}")

    return input_h5, output_h5, aL, aR, bL, bR


def make_output_file(output_file: str,
                     output_dataset: str,
                     xtable: XTable,
                     FOVwidth: int,
                     movie_shape: (int, int),
                     movie_dtype) -> None:
    """
    Creates the output h5 file, and ensures that it has a Dataset
    of the correct size for the dewarped movie.

    Parameters
    ----------
    output_file: str
        File path where the h5 containing the dewarped video will be saved.
    output_dataset: str,
        The name of the dataset representing the movie within the
        output h5 file.
    xtable: XTable
        A dict containing aggregate information about the input video
        to be used in the dewarping process.
    FOVwidth: int
        Field of View width. Will be the width of the output image, unless 0,
        then it will be the width of the input image.
    movie_shape: tuple[int, int],
        The shape of the numpy array containing the movie.
    movie_dtype
        The data type of the values in the numpy array representing the movie.
    Returns
    -------
    """

    # Remove old output if it exists and create new output file
    output_path = Path(output_file)
    if output_path.is_file():
        output_path.unlink()

    dewarped_file = h5py.File(output_file, 'w')
    if FOVwidth == 512:
        dewarped_file.create_dataset(
            output_dataset, shape=movie_shape, dtype=movie_dtype
        )
    else:
        out_shape = [
            movie_shape[0],
            movie_shape[1],
            512 - xtable['right_margin'] - xtable['left_margin']
        ]

        dewarped_file.create_dataset(
            output_dataset, shape=out_shape, dtype=movie_dtype
        )
    dewarped_file.close()


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

    xtable = create_xtable(movie, aL, aR, bL, bR, noise_reduction)

    make_output_file(output_file, output_dataset, xtable,
                     FOVwidth, movie_shape, movie_dtype)

    start_time = time.time()
    with multiprocessing.Pool(threads) as pool, \
         h5py.File(output_file, "a") as f:

        fn = functools.partial(
            xdewarp,
            FOVwidth=FOVwidth,
            xtable=xtable,
            noise_reduction=noise_reduction
        )

        # Chunk the movie up to prevent an error caused by multiprocessing
        # returning too much data
        chunk_size = 1000
        movie_chunks = np.array_split(
            movie, int(T / chunk_size), axis=0
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
    input_h5, output_h5, aL, aR, bL, bR = parse_input(input_data)

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

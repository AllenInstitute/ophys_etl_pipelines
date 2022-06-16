import argparse
import json
import logging
import os
import pathlib
from typing import Tuple
from subprocess import Popen


MOTION_DATA_BASE_PATH = pathlib.Path(
    '/allen/programs/mindscope/workgroups/surround/'
    'motion_correction_labeling_2022')
ROI_DATA_BASE_PATH = pathlib.Path(
    '/allen/programs/mindscope/workgroups/surround/'
    'denoising_labeling_2022/segmentations')


def create_trace_input_json(output_dir: pathlib.Path,
                            experiment_id: int) -> Tuple[pathlib.Path,
                                                         pathlib.Path]:
    """
    """
    # Setup output dir.
    trace_output_dir = output_dir / "traces_2022" / str(experiment_id)
    if not trace_output_dir.exists():
        logging.info(f'Creating Trace output dir {str(trace_output_dir)}')
        os.makedirs(trace_output_dir)
    else:
        logging.info(f'Using Trace output dir {str(trace_output_dir)}')
    # Put input and output into json.
    input_json = {
        "storage_directory": str(trace_output_dir),
        "motion_corrected_stack": str(
            MOTION_DATA_BASE_PATH
            / str(experiment_id)
            / f"{experiment_id}_motion_corrected_video.h5"),
        "motion_border": {"y1": 0.0,
                          "y0": 0.0,
                          "x0": 0.0,
                          "x1": 0.0},
        "log_0": str(MOTION_DATA_BASE_PATH
                     / str(experiment_id)
                     / f"{experiment_id}_rigid_motion_transform.csv"),
    }

    # Copy over ROI data and rename some columns.
    with open(ROI_DATA_BASE_PATH
              / str(experiment_id)
              / f"{experiment_id}_rois.json", 'r') as jfile:
        roi_data = json.load(jfile)
    modified_rois = []
    for roi in roi_data:
        modified_rois.append({"id": roi["id"],
                              "x": roi["x"],
                              "y": roi["y"],
                              "width": roi["width"],
                              "height": roi["height"],
                              "valid": roi["valid_roi"],
                              "mask": roi["mask_matrix"],
                              })
    input_json['rois'] = modified_rois

    input_json_path = (trace_output_dir
                       / f"{experiment_id}_traces_input.json")
    output_json_path = (trace_output_dir
                        / f"{experiment_id}_traces_output.json")
    if input_json_path.exists():
        logging.info('Trace input json exists. Using already created...')
    else:
        logging.info('Writing trace input json...')
        with open(input_json_path, 'w') as jfile:
            json.dump(input_json, jfile, indent=2)
    return input_json_path, output_json_path


def create_demix_input_json(
        output_dir: pathlib.Path,
        experiment_id: int,
        trace_output_json_path: pathlib.Path) -> Tuple[pathlib.Path,
                                                       pathlib.Path]:
    """
    """
    # Setup output dir.
    demix_output_dir = output_dir / "demix_2022" / str(experiment_id)
    if not demix_output_dir.exists():
        logging.info(f'Creating Demix output dir {str(demix_output_dir)}')
        os.makedirs(demix_output_dir)
    else:
        logging.info(f'Using Demix output dir {str(demix_output_dir)}')

    with open(trace_output_json_path, 'r') as jfile:
        trace_output_json = json.load(jfile)
    input_json = {
        "movie_h5": str(
            MOTION_DATA_BASE_PATH
            / str(experiment_id)
            / f"{experiment_id}_motion_corrected_video.h5"),
        "traces_h5": trace_output_json["roi_trace_file"],
        "output_file": str(demix_output_dir
                           / f"{experiment_id}_demixed_traces.h5"),
        "roi_masks": trace_output_json["input_parameters"]["rois"]
    }

    input_json_path = (demix_output_dir
                       / f"{experiment_id}_demix_input.json")
    output_json_path = (demix_output_dir
                        / f"{experiment_id}_demix_output.json")
    if input_json_path.exists():
        logging.info('Demix input json exists. Using already created...')
    else:
        logging.info('Writing demix input json...')
        with open(input_json_path, 'w') as jfile:
            json.dump(input_json, jfile, indent=2)
    return input_json_path, output_json_path


def create_neuropil_input_json(
        output_dir: pathlib.Path,
        experiment_id: int,
        trace_output_json_path: pathlib.Path) -> Tuple[pathlib.Path,
                                                       pathlib.Path]:
    """
    """
    # Setup output dir.
    neuropil_output_dir = output_dir / "neuropil_2022" / str(experiment_id)
    if not neuropil_output_dir.exists():
        logging.info(
            f'Creating neuropil output dir {str(neuropil_output_dir)}')
        os.makedirs(neuropil_output_dir)
    else:
        logging.info(f'Using neuropil output dir {str(neuropil_output_dir)}')

    with open(trace_output_json_path, 'r') as jfile:
        trace_output_json = json.load(jfile)
    input_json = {
        "neuropil_trace_file": str(trace_output_json["neuropil_trace_file"]),
        "storage_directory": str(neuropil_output_dir),
        "motion_corrected_stack": str(
            MOTION_DATA_BASE_PATH
            / str(experiment_id)
            / f"{experiment_id}_motion_corrected_video.h5"),
        "roi_trace_file": trace_output_json["roi_trace_file"],
    }

    input_json_path = (neuropil_output_dir
                       / f"{experiment_id}_neuropil_input.json")
    output_json_path = (neuropil_output_dir
                        / f"{experiment_id}_neuropil_output.json")
    if input_json_path.exists():
        logging.info('Neuropil input json exists. Using already created...')
    else:
        logging.info('Writing neuropil input json...')
        with open(input_json_path, 'w') as jfile:
            json.dump(input_json, jfile, indent=2)
    return input_json_path, output_json_path


def create_dff_input_json(
        output_dir: pathlib.Path,
        experiment_id: int,
        movie_frame_rate_hz: float,
        neuropil_output_json_path: pathlib.Path) -> Tuple[pathlib.Path,
                                                          pathlib.Path]:
    """
    """
    ddf_output_dir = output_dir / "dff_2022" / str(experiment_id)
    if not ddf_output_dir.exists():
        logging.info(f'Creating DF/F output dir {str(ddf_output_dir)}')
        os.makedirs(ddf_output_dir)
    else:
        logging.info(f'Using DF/F output dir {str(ddf_output_dir)}')

    with open(neuropil_output_json_path, "r") as jfile:
        neuropil_output_json = json.load(jfile)
    input_json = {
        "input_file": str(neuropil_output_json["neuropil_correction"]),
        "output_file": str(ddf_output_dir / f"{str(experiment_id)}_dff.h5"),
        "movie_frame_rate_hz": movie_frame_rate_hz,
    }

    input_json_path = (ddf_output_dir
                       / f"{experiment_id}_dff_input.json")
    output_json_path = (ddf_output_dir
                        / f"{experiment_id}_dff_output.json")
    if input_json_path.exists():
        logging.info('DF/F input json exists. Using already created...')
    else:
        logging.info('Writing DF/F input json...')
        with open(input_json_path, 'w') as jfile:
            json.dump(input_json, jfile, indent=2)
    return input_json_path, output_json_path



def get_motion_border(motion_data):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process labeled data.')
    parser.add_argument('--experiment_id',
                        type=str,
                        help='Path to write data to. Script will create '
                             'sub-directories for each queue.')
    parser.add_argument('--output_path',
                        type=str,
                        help='Path to write data to. Script will create '
                             'sub-directories for each queue.')
    parser.add_argument('--movie_frame_rate_hz',
                        type=float,
                        help='Framerate of the experiment.')
    args = parser.parse_args()
    base_dir_path = pathlib.Path(args.output_path)

    # Extract traces
    trace_input_json_path, trace_output_json_path = create_trace_input_json(
        base_dir_path, args.experiment_id)
    job = Popen("python -m allensdk.brain_observatory.ophys.trace_extraction "
                f"--input_json={str(trace_input_json_path)} "
                f"--output_json={str(trace_output_json_path)}",
                shell=True)
    job.wait()

    # Demix
    demix_input_json_path, demix_output_json_path = create_demix_input_json(
        base_dir_path, args.experiment_id, trace_output_json_path)
    job = Popen("python -m allensdk.internal.pipeline_modules.run_demixing "
                f"{str(demix_input_json_path)} "
                f"{str(demix_output_json_path)}",
                shell=True)
    job.wait()

    # Neuropil extraction
    npil_input_json_path, npil_output_json_path = create_neuropil_input_json(
        base_dir_path, args.experiment_id, trace_output_json_path)
    job = Popen(
        "python -m allensdk.internal.pipeline_modules.run_neuropil_correction "
        f"{str(npil_input_json_path)} "
        f"{str(npil_output_json_path)}",
        shell=True)
    job.wait()

    # DF/F calculation
    dff_input_json_path, dff_output_json_path = create_dff_input_json(
        base_dir_path,
        args.experiment_id,
        args.movie_frame_rate_hz,
        npil_output_json_path)
    job = Popen(
        "python -m ophys_etl.modules.dff --n_parallel_workers 24 "
        f"--input_json={str(dff_input_json_path)} "
        f"--output_json={str(dff_output_json_path)}",
        shell=True)
    job.wait()


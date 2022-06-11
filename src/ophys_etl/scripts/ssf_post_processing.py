import json
import logging
import os
import pathlib
import shulti
from subprocess import Popen


MOTION_DATA_BASE_PATH = pathlib.Path(
    '/allen/programs/mindscope/workgroups/surround/'
    'motion_correction_labeling_2022')
ROI_DATA_BASE_PATH = pathlib.Path(
    '/allen/programs/mindscope/workgroups/surround/'
    'denoising_labeling_2022/segmentations')
TRACE_DATA_BASE_PATH = pathlib.Path(
    '/allen/programs/mindscope/workgroups/surround/'
    'traces_2022/')


def create_trace_input_json(experiment_id: int) -> Path:
    """
    """
    # Setup output and input files
    trace_output_dir = str(TRACE_DATA_BASE_PATH / str(experiment_id))
    if not trace_output_dir.exists():
        logging.info(f'Creating Trace output dir {str(trace_output_dir)}')
        os.mkdir(trace_output_dir)
    else:
        logging.info(f'Using Trace output dir {str(trace_output_dir)}')
        os.mkdir(trace_output_dir)
    input_json = {
        "storage_directory": str(TRACE_DATA_BASE_PATH / str(experiment_id)),
        "motion_corrected_stack": str(
            MOTION_DATA_BASE_PATH /
            str(experiment_id) /
            f"{experiment_id}_motion_corrected_video.h5"),
        "motion_border": {"y1": 0.0,
                          "y0": 0.0,
                          "x0": 0.0,
                          "x1": 0.0},
        "log_0": str(MOTION_DATA_BASE_PATH /
                     str(experiment_id) /
                     f"{experiment_id}rigid_motion_transform.csv"),
    }

    # Copy over ROI data and rename some columns.
    with open(ROI_DATA_BASE_PATH /
              str(experiment_id) /
              f"{experiment_id}_rois.json", 'r') as jfile:
        roi_data = json.load(jfile)
    modified_rois = []
    for roi in roi_data:
        modified_rois.append({"id": roi["id"],
                              "x": roi["x"],
                              "y": roi["y"],
                              "width": roi["width"],
                              "height": roi["height"],
                              "valid": roi["valid_roi"],
                              "mask": roi["mask"],
                              })
    input_json['rois'] = modified_rois
    input_json_path = (TRACE_DATA_BASE_PATH /
                       str(experiment_id) /
                       f"{experiment_id}_traces_input.json")
    if input_json_path.exists():
        logging.info('Trace input json exists. Using already created...')
    else:
        logging.info('Trace input json exists. Using already created...')
        with open(input_json_path, 'w') as jfile:
            json.dump(input_json)
    return input_json_path
  

def get_motion_border(motion_data):
    pass


if __name__ == "__main__":
    pass

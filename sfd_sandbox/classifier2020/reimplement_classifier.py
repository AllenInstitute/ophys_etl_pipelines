"""
Just going to naively copy code from the Sagemaker notebook and see
if I can run it locally
"""

#import sys
#sys.path.append('/Users/scott.daniel/Pika/DeepCell')

from croissant.utils import read_jsonlines
import pandas as pd

import pathlib
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from imgaug import augmenters as iaa
from Classifier import Classifier
from deepcell_models.VggBackbone import VggBackbone
from DataSplitter import DataSplitter
import datetime
from PIL import Image
from Transform import Transform
from Inference import inference
from RoiDataset import RoiDataset
import random
import time
import tempfile

from ophys_etl.modules.cell_classifier_2020.generate_artifacts import (
    run_artifacts)


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

def run(cnn, experiment_dir, use_cuda=False, test_transform=None):
    res = []
    times = []
    num_rois = []
   
    #print(f'Downloading artifacts for experiment {experiment_id}')
    #download_artifacts_for_experiment(experiment_id=experiment_id)
    #print(f'Done Downloading artifacts for experiment {experiment_id}')

    experiment_dir = pathlib.Path(experiment_dir)

    roi_ids = []
    for name in experiment_dir.rglob('avg*png'):
        roi_id = int(str(name).split('avg_')[-1].replace('.png',''))
        roi_ids.append(roi_id)

    num_rois.append(len(roi_ids))

    test = RoiDataset(manifest_path=None,
                      project_name=None,
                      data_dir=experiment_dir,
                      roi_ids=roi_ids, 
                      transform=test_transform,
                      has_labels=False,
                      parse_from_manifest=False)

    test_dataloader = DataLoader(dataset=test, shuffle=False, batch_size=64)
        
    print(f'Inference for experiment {experiment_dir}')
        
    start = time.time()
    _, inference_res = inference(model=cnn,
                                test_loader=test_dataloader,
                                has_labels=False,
                                checkpoint_path=pathlib.Path('models').resolve().absolute(),
                                use_cuda=use_cuda)
    end = time.time()
        
    times.append(end - start)
        
    inference_res['experiment_id'] = str(experiment_dir)
    res.append(inference_res)

    return pd.concat(res), np.array(times), np.array(num_rois)

def classify(artifact_dir):

    all_transform = transforms.Compose([
        iaa.Sequential([
            iaa.CenterCropToFixedSize(height=60, width=60)
        ]).augment_image,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Transform(all_transform=all_transform)

    model = torchvision.models.vgg11_bn(pretrained=True, progress=False)
    cnn = VggBackbone(model=model, truncate_to_layer=15, classifier_cfg=[512])

    t0 = time.time()
    (pd_result,
     time_result,
     num_result) = run(cnn.cpu(),
                       artifact_dir,
                       test_transform=test_transform)

    duration = time.time()-t0

    roi_id_to_score = dict()
    roi_id_to_label = dict()
    for roi_id, label, score in zip(pd_result['roi-id'],
                                    pd_result['y_pred'],
                                    pd_result['y_score']):
        roi_id_to_label[roi_id] = label
        roi_id_to_score[roi_id] = score
    return roi_id_to_label, roi_id_to_score

import argparse
import json

def full_classification(roi_path=None,
                        video_path=None,
                        scratch_dir=None,
                        out_file_path=None):

    scratch_dir = pathlib.Path(tempfile.mkdtemp(dir=scratch_dir))
    roi_list = run_artifacts(roi_path=roi_path,
                      video_path=video_path,
                      n_roi=-1,
                      out_dir=scratch_dir)

    (result,
     score) = classify(pathlib.Path(scratch_dir))

    assert len(roi_list) == len(result)
    for roi in roi_list:
        roi['valid'] = result[roi['id']]
        roi['classifier_score'] = score[roi['id']]

    with open(out_file_path, 'w') as out_file:
        out_file.write(json.dumps(roi_list, indent=2))

    print(f'cleaning up {scratch_dir}')
    t0 = time.time()
    for fname in scratch_dir.iterdir():
        fname.unlink()
    scratch_dir.rmdir()
    duration = time.time()-t0
    print(f'clean up took {duration:.2e}')

    #print(result)
    #print(scratch_dir)
    #duration = time.time()-t0
    #print(f'that took {duration:.2e}')


if __name__ == "__main__":

    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_path', type=str, default=None)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--scratch_dir', type=str, default=None)
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    assert args.out_file is not None
    full_classification(roi_path=args.roi_path,
                        video_path=args.video_path,
                        scratch_dir=args.scratch_dir,
                        out_file_path=args.out_file)

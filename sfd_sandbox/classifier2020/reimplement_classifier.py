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


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)


all_transform = transforms.Compose([
    iaa.Sequential([
        iaa.CenterCropToFixedSize(height=30, width=30)
    ]).augment_image,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = Transform(all_transform=all_transform)

model = torchvision.models.vgg11_bn(pretrained=True, progress=False)
cnn = VggBackbone(model=model, truncate_to_layer=15, classifier_cfg=[512])

def run(cnn, experiment_dir, use_cuda=False):
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

if __name__ == "__main__":

    (pd_result,
     time_result,
     num_result) = run(cnn.cpu(),
                       pathlib.Path('785569423_artifacts'))

    print(time_result)
    print(num_result)
    print(pd_result)
    print(pd_result.y_pred.sum())
    import Inference
    print(Inference.__file__)

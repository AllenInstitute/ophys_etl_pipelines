import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from Metrics import Metrics
from RoiDataset import RoiDataset


def inference(model: torch.nn.Module, test_loader: DataLoader, checkpoint_path, has_labels=True, threshold=0.5, ensemble=True,
              cv_fold=None, use_cuda=True):
    if not ensemble and cv_fold is None:
        raise ValueError('If not using ensemble, need to give cv_fold')
    if cv_fold is not None and ensemble:
        raise ValueError('Ensemble should be false if passing in cv_fold')

    dataset: RoiDataset = test_loader.dataset
    metrics = Metrics()

    if ensemble:
        models = os.listdir(checkpoint_path)
        models = [model for model in models if model != 'model_init.pt']
    else:
        models = [f'{cv_fold}_model.pt']

    y_scores = np.zeros((len(models), len(dataset)))
    y_preds = np.zeros((len(models), len(dataset)))

    if len(models) == 0:
        raise RuntimeError("There are no models to "
                           f"apply in {checkpoint_path}")

    for i, model_checkpoint in enumerate(models):
        state_dict = torch.load(f'{checkpoint_path}/{model_checkpoint}',
                                map_location=torch.device('cpu'))
                                # map_location invocation is just for running
                                # on CPU
        model.load_state_dict(state_dict)

        model.eval()
        prev_start = 0

        for data, _ in test_loader:
            if use_cuda:
                raise RuntimeError("should not be using cuda")
                data = data.cuda()

            with torch.no_grad():
                output = model(data)
                output = output.squeeze()
                y_score = torch.sigmoid(output).cpu().numpy()
                start = prev_start
                end = start + data.shape[0]
                y_scores[i][start:end] = y_score
                y_preds[i][start:end] = y_score > threshold
                prev_start = end

        if has_labels:
            TP = ((dataset.y == 1) & (y_preds[i] == 1)).sum().item()
            FP = ((dataset.y == 0) & (y_preds[i] == 1)).sum().item()
            FN = ((dataset.y == 1) & (y_preds[i] == 0)).sum().item()
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f1 = 2 * p * r / (p + r)
            print(f'{model_checkpoint} precision: {p}')
            print(f'{model_checkpoint} recall: {r}')
            print(f'{model_checkpoint} f1: {f1}')

    y_scores = y_scores.mean(axis=0)
    y_preds = y_scores > threshold
    if has_labels:
        metrics.update_accuracies(y_true=dataset.y, y_score=y_scores, threshold=threshold)

    df = pd.DataFrame({'roi-id': dataset.roi_ids, 'y_score': y_scores, 'y_pred': y_preds})

    return metrics, df

import numpy as np
import torch
from sklearn.metrics import average_precision_score


class TrainingMetrics:
    def __init__(self, n_epochs):
        self.losses = np.zeros(n_epochs)
        self.auprs = np.zeros(n_epochs)
        self.best_epoch = None

    def update(self, epoch, loss, aupr):
        self.losses[epoch] = loss
        self.auprs[epoch] = aupr


class Metrics:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.loss = 0.0
        self.y_scores = []
        self.y_trues = []

    @property
    def precision(self):
        try:
            res = self.TP / (self.TP + self.FP)
        except ZeroDivisionError:
            res = 0.0
        return res

    @property
    def recall(self):
        try:
            res = self.TP / (self.TP + self.FN)
        except ZeroDivisionError:
            res = 0.0
        return res

    @property
    def F1(self):
        precision = self.precision
        recall = self.recall

        try:
            res = 2 * precision * recall / (recall + precision)
        except ZeroDivisionError:
            res = 0.0
        return res

    @property
    def AUPR(self):
        return average_precision_score(y_true=self.y_trues, y_score=self.y_scores)

    def update_accuracies(self, y_true, y_out=None, y_score=None, threshold=0.5):
        if y_out is not None and y_score is not None:
            raise ValueError('Supply either logits or score')
        if y_out is None and y_score is None:
            raise ValueError('Supply either logits or score')

        if y_score is None:
            y_score = torch.sigmoid(y_out)
        y_pred = y_score > threshold

        self._increment_TP(y_pred=y_pred, y_true=y_true)
        self._increment_FP(y_pred=y_pred, y_true=y_true)
        self._increment_FN(y_pred=y_pred, y_true=y_true)

    def update_outputs(self, y_out, y_true):
        preds = torch.sigmoid(y_out).detach().cpu().numpy().tolist()
        y_true = y_true.detach().cpu().numpy().tolist()
        self.y_scores += preds
        self.y_trues += y_true

    def update_loss(self, loss, num_batches):
        self.loss += loss / num_batches  # loss*num_batches/N = (Sum of all l)/N

    def _increment_TP(self, y_pred, y_true):
        self.TP += ((y_true == 1) & (y_pred == 1)).sum().item()

    def _increment_FP(self, y_pred, y_true):
        self.FP += ((y_true == 0) & (y_pred == 1)).sum().item()

    def _increment_FN(self, y_pred, y_true):
        self.FN += ((y_true == 1) & (y_pred == 0)).sum().item()


class CVMetrics:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.train_metrics = []
        self.valid_metrics = []

    def update(self, train_metrics, valid_metrics):
        self.train_metrics.append(train_metrics)
        self.valid_metrics.append(valid_metrics)

    @property
    def metrics(self):
        train_loss = sum([x.losses[x.best_epoch] for x in self.train_metrics]) / self.n_splits
        valid_loss = sum([x.losses[x.best_epoch] for x in self.valid_metrics]) / self.n_splits

        return {
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }





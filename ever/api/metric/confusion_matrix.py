import torch
import numpy as np
from scipy import sparse


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

    def forward(self, y_true, y_pred):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        y_pred = y_pred.reshape((-1,))
        y_true = y_true.reshape((-1,))

        v = np.ones_like(y_pred)
        cm = sparse.coo_matrix((v, (y_true, y_pred)), shape=(self.num_classes, self.num_classes), dtype=np.float32)
        self._total += cm

        return cm

    @property
    def dense_cm(self):
        return self._total.toarray()

    @property
    def sparse_cm(self):
        return self._total

    def reset(self):
        num_classes = self.num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

    @staticmethod
    def plot(confusion_matrix):
        return NotImplementedError

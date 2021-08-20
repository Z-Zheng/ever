import logging
import os
import time
import numpy as np
import prettytable as pt
import pandas as pd

from ever.api.metric.confusion_matrix import ConfusionMatrix
from ever.core.logger import get_console_file_logger

EPS = 1e-7


class PixelMetric(ConfusionMatrix):
    def __init__(self, num_classes, logdir=None, logger=None, class_names=None):
        super(PixelMetric, self).__init__(num_classes)
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        if logdir is not None and logger is None:
            self._logger = get_console_file_logger('PixelMetric', logging.INFO, self.logdir)
        elif logger is not None:
            self._logger = logger
        else:
            self._logger = None
        self._class_names = class_names
        if class_names:
            assert num_classes == len(class_names)

    @property
    def logger(self):
        return self._logger

    @staticmethod
    def compute_iou_per_class(confusion_matrix):
        """
        Args:
            confusion_matrix: numpy array [num_classes, num_classes] row - gt, col - pred
        Returns:
            iou_per_class: float32 [num_classes, ]
        """
        sum_over_row = np.sum(confusion_matrix, axis=0)
        sum_over_col = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        denominator = sum_over_row + sum_over_col - diag

        iou_per_class = diag / (denominator + EPS)

        return iou_per_class

    @staticmethod
    def compute_recall_per_class(confusion_matrix):
        sum_over_row = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        recall_per_class = diag / (sum_over_row + EPS)
        return recall_per_class

    @staticmethod
    def compute_precision_per_class(confusion_matrix):
        sum_over_col = np.sum(confusion_matrix, axis=0)
        diag = np.diag(confusion_matrix)
        precision_per_class = diag / (sum_over_col + EPS)
        return precision_per_class

    @staticmethod
    def compute_overall_accuracy(confusion_matrix):
        diag = np.diag(confusion_matrix)
        return np.sum(diag) / (np.sum(confusion_matrix) + EPS)

    @staticmethod
    def compute_F_measure_per_class(confusion_matrix, beta=1.0):
        precision_per_class = PixelMetric.compute_precision_per_class(confusion_matrix)
        recall_per_class = PixelMetric.compute_recall_per_class(confusion_matrix)
        F1_per_class = (1 + beta ** 2) * precision_per_class * recall_per_class / (
                (beta ** 2) * precision_per_class + recall_per_class + EPS)

        return F1_per_class

    @staticmethod
    def cohen_kappa_score(cm_th):
        cm_th = cm_th.astype(np.float32)
        n_classes = cm_th.shape[0]
        sum0 = cm_th.sum(axis=0)
        sum1 = cm_th.sum(axis=1)
        expected = np.outer(sum0, sum1) / (np.sum(sum0) + EPS)
        w_mat = np.ones([n_classes, n_classes])
        w_mat.flat[:: n_classes + 1] = 0
        k = np.sum(w_mat * cm_th) / (np.sum(w_mat * expected) + EPS)
        return 1. - k

    def _log_summary(self, table, dense_cm):
        if self.logger is not None:
            self.logger.info('\n' + table.get_string())
            if self.logdir is not None:
                np.save(os.path.join(self.logdir, 'confusion_matrix-{time}.npy'.format(time=time.time())), dense_cm)
        else:
            print(table)

    def summary_iou(self):
        dense_cm = self._total.toarray()
        iou_per_class = PixelMetric.compute_iou_per_class(dense_cm)
        miou = iou_per_class.mean()

        tb = pt.PrettyTable()
        tb.field_names = ['class', 'iou']
        for idx, iou in enumerate(iou_per_class):
            tb.add_row([idx, iou])
        tb.add_row(['mIoU', miou])

        self._log_summary(tb, dense_cm)

        return tb

    def summary_all(self, dec=5):
        dense_cm = self._total.toarray()

        iou_per_class = np.round(PixelMetric.compute_iou_per_class(dense_cm), dec)
        miou = np.round(iou_per_class.mean(), dec)
        F1_per_class = np.round(PixelMetric.compute_F_measure_per_class(dense_cm, beta=1.0), dec)
        mF1 = np.round(F1_per_class.mean(), dec)
        overall_accuracy = np.round(PixelMetric.compute_overall_accuracy(dense_cm), dec)
        kappa = np.round(PixelMetric.cohen_kappa_score(dense_cm), dec)

        precision_per_class = np.round(PixelMetric.compute_precision_per_class(dense_cm), dec)
        mprec = np.round(precision_per_class.mean(), dec)
        recall_per_class = np.round(PixelMetric.compute_recall_per_class(dense_cm), dec)
        mrecall = np.round(recall_per_class.mean(), dec)

        tb = pt.PrettyTable()
        if self._class_names:
            tb.field_names = ['name', 'class', 'iou', 'f1', 'precision', 'recall']
            for idx, (iou, f1, precision, recall) in enumerate(
                    zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
                tb.add_row([self._class_names[idx], idx, iou, f1, precision, recall])

            tb.add_row(['', 'mean', miou, mF1, mprec, mrecall])
            tb.add_row(['', 'OA', overall_accuracy, '-', '-', '-'])
            tb.add_row(['', 'Kappa', kappa, '-', '-', '-'])

        else:
            tb.field_names = ['class', 'iou', 'f1', 'precision', 'recall']
            for idx, (iou, f1, precision, recall) in enumerate(
                    zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
                tb.add_row([idx, iou, f1, precision, recall])

            tb.add_row(['mean', miou, mF1, mprec, mrecall])
            tb.add_row(['OA', overall_accuracy, '-', '-', '-'])
            tb.add_row(['Kappa', kappa, '-', '-', '-'])

        self._log_summary(tb, dense_cm)

        return tb


def prettytable_to_dataframe(tb: pt.PrettyTable):
    heads = tb.field_names
    data = tb._rows
    df = pd.DataFrame(data, columns=heads)
    return df


def prettytable_to_csv(tb: pt.PrettyTable, csv_file: str):
    df = prettytable_to_dataframe(tb)
    df.to_csv(csv_file)

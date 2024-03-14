import torch
import torch.nn as nn
from .pixel import PixelMetric
from tqdm import tqdm
from ever.core.device import auto_device
from ever.core.dist import all_gather, get_rank
from ever.data.distributed import DistributedNonOverlapSeqSampler


def _data_parse_fn(data):
    x, y_blob = data
    return x, y_blob, {}


def _tune_model_fn(model): return model


def _process_prediction_fn(y_true, y_pred, data_info, model_dir, checkpoint):
    return y_true, y_pred


def evaluate_pixel_prediction_task(num_classes,
                                   data_parse_fn=_data_parse_fn,
                                   tune_model_fn=_tune_model_fn,
                                   prediction_fn=_process_prediction_fn,
                                   desc='',
                                   acc_table_based_callback=None,
                                   cuda_empty_cache=True):
    def _evaluate_fn(self, test_dataloader, config=None):
        pm = PixelMetric(num_classes,
                         self.model_dir,
                         logger=self.logger)
        self.model.eval()
        model: nn.Module = tune_model_fn(self.model)

        device = auto_device()
        model.to(device)

        with torch.no_grad():
            for data in tqdm(test_dataloader, desc=desc):
                x, y_true, other_info = data_parse_fn(data)
                x = x.to(device)

                y_pred = model(x)

                y_true, y_pred = prediction_fn(y_true, y_pred, other_info, self.model_dir, self.checkpoint)

                pm.forward(y_true, y_pred)

        acc_tb = pm.summary_all()
        if acc_table_based_callback is not None:
            launcher = self
            acc_table_based_callback(launcher, acc_tb)

        if cuda_empty_cache:
            torch.cuda.empty_cache()
        return acc_tb

    return _evaluate_fn


def distributed_evaluate_pixel_prediction_task(num_classes,
                                               data_parse_fn=_data_parse_fn,
                                               tune_model_fn=_tune_model_fn,
                                               prediction_fn=_process_prediction_fn,
                                               desc='',
                                               acc_table_based_callback=None,
                                               cuda_empty_cache=True
                                               ):
    def _evaluate_fn(self, test_dataloader, config=None):
        if not isinstance(test_dataloader.sampler, DistributedNonOverlapSeqSampler):
            from torch.utils.data import DataLoader
            dataloder = DataLoader(
                dataset=test_dataloader.dataset,
                batch_size=test_dataloader.batch_size,
                sampler=DistributedNonOverlapSeqSampler(test_dataloader.dataset),
                num_workers=test_dataloader.num_workers,
                pin_memory=test_dataloader.pin_memory,
                drop_last=test_dataloader.drop_last,
                worker_init_fn=test_dataloader.worker_init_fn
            )
        else:
            dataloder = test_dataloader

        pm = PixelMetric(num_classes,
                         self.model_dir,
                         logger=self.logger)
        self.model.eval()
        model: nn.Module = tune_model_fn(self.model)

        device = auto_device()
        model.to(device)

        if get_rank() == 0:
            dataloder = tqdm(dataloder, desc='Distributed Rank:0' + desc)
        else:
            dataloder = dataloder

        with torch.no_grad():
            for data in dataloder:
                x, y_true, other_info = data_parse_fn(data)
                x = x.to(device)

                y_pred = model(x)

                y_true, y_pred = prediction_fn(y_true, y_pred, other_info, self.model_dir, self.checkpoint)

                pm.forward(y_true, y_pred)

        cm = pm.dense_cm
        cm_list = all_gather(cm)

        total_cm = sum(cm_list)

        acc_tb = pm.summary_all(dense_cm=total_cm)

        torch.cuda.empty_cache()

        if acc_table_based_callback is not None:
            launcher = self
            acc_table_based_callback(launcher, acc_tb)

        if cuda_empty_cache:
            torch.cuda.empty_cache()
        return acc_tb

    return _evaluate_fn

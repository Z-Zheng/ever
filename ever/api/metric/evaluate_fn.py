import torch
import torch.nn as nn
from .pixel import PixelMetric
from tqdm import tqdm
from ever.core.device import auto_device


def _data_parse_fn(data):
    x, y_blob = data
    return x, y_blob, {}


def _tune_model_fn(model): return model


def _process_prediction_fn(y_true, y_pred, data_info, model_dir, checkpoint): return y_true, y_pred


def evaluate_pixel_prediction_task(num_classes,
                                   data_parse_fn=_data_parse_fn,
                                   tune_model_fn=_tune_model_fn,
                                   prediction_fn=_process_prediction_fn,
                                   desc=''):
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

        torch.cuda.empty_cache()
        return pm.summary_all()

    return _evaluate_fn

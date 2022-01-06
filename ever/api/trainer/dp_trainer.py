import torch
import torch.nn as nn

from ever.api.trainer import trainer


class DPTrainer(trainer.Trainer):
    def make_model(self):
        model = super().make_model()
        model = nn.DataParallel(model,
                                device_ids=list(range(torch.cuda.device_count())))
        return model

import torch
import torch.nn as nn

from ever.api.trainer import trainer

class DPTrainer(trainer.Trainer):
    def __init__(self):
        super(DPTrainer, self).__init__()

    def make_model(self):
        model = super(DPTrainer, self).make_model()
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        return model

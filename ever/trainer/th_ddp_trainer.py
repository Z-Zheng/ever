import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast

from ever.trainer import trainer
from ever.core.launcher import Launcher


class THDDPTrainer(trainer.Trainer):
    def __init__(self, args):
        super().__init__(args)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )

    def make_model(self):
        model = super(THDDPTrainer, self).make_model()
        if self.config.train.get('sync_bn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = self.torch_compile(model)
        model = model.to(self.device)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=self.args.find_unused_parameters,
        )
        return model

    def build_launcher(self):
        kwargs = dict(model_dir=self.args.model_dir, mixed_precision=self.args.mixed_precision)
        kwargs.update(dict(model=self.make_model()))
        kwargs.update(
            self.make_lr_optimizer(kwargs['model'].module.custom_param_groups()))
        tl = Launcher(**kwargs)

        return dict(config=self.config, launcher=tl)


class GANLauncher(Launcher):
    def compute_loss_gradient(self, data, forward_times=1):
        with autocast('cuda', enabled=self._amp, dtype=self._mixed_precision):
            msg_dict = self.model.forward_backward(data, optimizer=self.optimizer, scaler=self.scaler)
        return msg_dict


class THDDPGANTrainer(trainer.Trainer):
    def __init__(self, args):
        super().__init__(args)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )

    def make_model(self):
        model = super().make_model()
        if self.config.train.get('sync_bn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = self.torch_compile(model)
        model = model.to(self.device)

        model.G = nn.parallel.DistributedDataParallel(
            model.G,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=self.args.find_unused_parameters,
            broadcast_buffers=False,
        )
        model.D = nn.parallel.DistributedDataParallel(
            model.D,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=self.args.find_unused_parameters,
            broadcast_buffers=False,
        )
        return model

    def build_launcher(self):
        kwargs = dict(model_dir=self.args.model_dir, mixed_precision=self.args.mixed_precision)
        kwargs.update(dict(model=self.make_model()))
        kwargs.update(
            self.make_lr_optimizer(kwargs['model'].custom_param_groups()))
        tl = GANLauncher(**kwargs)

        return dict(config=self.config, launcher=tl)

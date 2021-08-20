import torch
import torch.distributed as dist

from ever.api.trainer import trainer
from ever.core.launcher import Launcher, scale_dict, average_dict
from ever.util import param_util
from torch.cuda.amp import GradScaler, autocast
from ever.core import to
import torch.nn as nn


class THAmpLauncher(Launcher):
    def __init__(self,
                 model_dir,
                 model,
                 optimizer,
                 lr_schedule):
        super(THAmpLauncher, self).__init__(model_dir, model, optimizer, lr_schedule)

        self.scaler = GradScaler()

    def compute_loss_gradient(self, data):
        """

        Args:
            data:

        Returns:

        """
        if not isinstance(data, list):
            data = [data]

        loss_dict = {'total_loss': 0.0}

        for d in data:
            d = to.to_device(d, self._device)

            with autocast():
                msg_dict = self._model(*d)

                losses = {k: v for k, v in msg_dict.items() if k.endswith('loss')}
                # scale losses by 1. / forward times
                if len(data) != 1:
                    losses = scale_dict(losses, 1. / len(data))
                losses = average_dict(losses)
                total_loss = sum([e for e in losses.values()])

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            self.scaler.scale(total_loss).backward()

            self.scaler.unscale_(self.optimizer)

            self.log_info_dict(data, losses, msg_dict, loss_dict)

        return loss_dict

    def apply_gradient(self):
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        self.scaler.step(self._optimizer)
        # Updates the scale for next iteration.
        self.scaler.update()
        self._optimizer.zero_grad()

        self._update_lr()
        self._ckpt.step()


class THAMPDDPTrainer(trainer.Trainer):
    def __init__(self):
        super(THAMPDDPTrainer, self).__init__()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )

    def make_model(self):
        model = super(THAMPDDPTrainer, self).make_model()
        if self.config.train.get('sync_bn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(self.device)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True,
        )

        return model

    def run(self, after_construct_launcher_callbacks=None):
        kwargs = dict(model_dir=self.args.model_dir)
        kwargs.update(dict(model=self.make_model()))
        kwargs.update(self.make_lr_optimizer(kwargs['model'].parameters()))

        kw_dataloader = self.make_dataloader()
        tl = THAmpLauncher(**kwargs)

        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)

        tl.logger.info(
            'th sync bn: {}'.format('on' if self.config.train.get('sync_bn', False) else 'off'))
        tl.logger.info('external parameter: {}'.format(self.args.opts))

        tl.train_by_config(kw_dataloader['traindata_loader'],
                           config=trainer.merge_dict(self.config.train, self.config.test),
                           test_data_loader=kw_dataloader['testdata_loader'])

        return dict(config=self.config, launcher=tl)

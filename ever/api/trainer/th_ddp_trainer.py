import torch
import torch.distributed as dist
import torch.nn as nn

from ever.api.trainer import trainer
from ever.core.launcher import Launcher
from ever.util import param_util


class THDDPTrainer(trainer.Trainer):
    def __init__(self):
        super(THDDPTrainer, self).__init__()

        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )

    def make_model(self):
        model = super(THDDPTrainer, self).make_model()
        if self.config.train.get('sync_bn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(self.device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
            find_unused_parameters=True,
        )
        return model

    def run(self, after_construct_launcher_callbacks=None):
        kwargs = dict(model_dir=self.args.model_dir)
        kwargs.update(dict(model=self.make_model()))
        kwargs.update(self.make_lr_optimizer(kwargs['model'].parameters()))

        kw_dataloader = self.make_dataloader()
        tl = Launcher(**kwargs)

        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)

        tl.logger.info('th sync bn: {}'.format('True' if self.config.train.get('sync_bn', False) else 'False'))
        tl.logger.info('external parameter: {}'.format(self.args.opts))
        tl.train_by_config(kw_dataloader['traindata_loader'],
                           config=trainer.merge_dict(self.config.train, self.config.test),
                           test_data_loader=kw_dataloader['testdata_loader'])

        return dict(config=self.config, launcher=tl)

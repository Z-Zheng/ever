import torch
import torch.distributed as dist

from ever.api.trainer import trainer
from ever.core.launcher import Launcher
from ever.util import param_util
from ever.core import default_backward

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex")


class ApexDDPTrainer(trainer.Trainer):
    def __init__(self):
        super(ApexDDPTrainer, self).__init__()
        self.parser.add_argument('--opt_level', type=str, default='O0', help='O0, O1, O2, O3')
        self.parser.add_argument('--keep_batchnorm_fp32', type=bool, default=None, help='')

        self.OPT_LEVELS = ['O0', 'O1', 'O2', 'O3']

    def make_model(self):
        model = super(ApexDDPTrainer, self).make_model()
        if self.config.train.get('apex_sync_bn', False):
            model = apex.parallel.convert_syncbn_model(model)

        return model

    def run(self, after_construct_launcher_callbacks=None):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )
        kwargs = dict(model_dir=self.args.model_dir)
        kwargs.update(dict(model=self.make_model().to(self.device)))
        kwargs.update(self.make_lr_optimizer(kwargs['model'].parameters()))
        if dist.is_available():
            model, optimizer = amp.initialize(kwargs['model'], kwargs['optimizer'],
                                              opt_level=self.args.opt_level)

            # half bn for https://github.com/NVIDIA/apex/issues/122, when frozen bn
            if self.config.train.get('half_bn', False):
                model.apply(trainer.half_bn)

            model = DDP(model, delay_allreduce=True)
            kwargs['model'] = model
            kwargs['optimizer'] = optimizer

        kw_dataloader = self.make_dataloader()
        tl = Launcher(**kwargs)

        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)

        tl.logger.info('[NVIDIA/apex] amp optimizer. opt_level = {}'.format(self.args.opt_level))
        tl.logger.info(
            '[NVIDIA/apex] sync bn: {}'.format('on' if self.config.train.get('apex_sync_bn', False) else 'off'))
        tl.logger.info('external parameter: {}'.format(self.args.opts))
        tl.override_backward(default_backward.amp_backward)
        tl.train_by_config(kw_dataloader['traindata_loader'],
                           config=trainer.merge_dict(self.config.train, self.config.test),
                           test_data_loader=kw_dataloader['testdata_loader'])

        return dict(config=self.config, launcher=tl)

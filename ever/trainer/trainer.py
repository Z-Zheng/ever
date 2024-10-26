import os
import torch

from ever.core import config
from ever.core.dist import main_process_only
from ever.core.builder import make_dataloader
from ever.core.builder import make_learningrate
from ever.core.builder import make_model
from ever.core.builder import make_optimizer
from ever.core.builder import make_callback
from ever.core.launcher import Launcher

from ever.util import param_util

__all__ = ['merge_dict', 'Trainer', 'half_bn']


def merge_dict(dict1: dict, dict2: dict):
    # check whether redundant key
    redundant_keys = [key for key in dict1 if key in dict2]
    if len(redundant_keys) > 0:
        raise ValueError('Duplicate keys: {}'.format(redundant_keys))

    merged = dict1.copy()
    merged.update(dict2)

    if isinstance(dict1, config.AttrDict):
        return config.AttrDict.from_dict(merged)
    return merged


def half_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.half()


class Trainer(object):
    def __init__(self, args):
        self._args = args
        self._cfg = config.import_config(self.args.config_path)
        if self._args.opts:
            self._cfg.update_from_list(self._args.opts)

        self.initialize_workspace()

        self._callbacks = []

    def __call__(self):
        return self

    @main_process_only
    def initialize_workspace(self):
        os.makedirs(self.args.model_dir, exist_ok=True)
        self.config.to_pickle(os.path.join(self.args.model_dir, 'config.pkl'))

    @property
    def device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @property
    def args(self):
        return self._args

    @property
    def config(self):
        return self._cfg

    @property
    def cfg(self):
        return self._cfg

    def make_model(self):
        model = make_model(self.config.model)
        return model

    def make_dataloader(self):
        traindata_loader = make_dataloader(self.config.data.train)
        testdata_loader = make_dataloader(self.config.data.test) if 'test' in self.config.data else None
        return dict(traindata_loader=traindata_loader, testdata_loader=testdata_loader)

    def make_lr_optimizer(self, params):
        # case1: single lr, single opt
        if hasattr(self.config.learning_rate, 'type') and hasattr(self.config.optimizer, 'type'):
            lr_schedule = make_learningrate(self.config.learning_rate)
            self.config.optimizer.params['lr'] = lr_schedule.base_lr
            optimizer = make_optimizer(self.config.optimizer, params=params)
            return dict(lr_schedule=lr_schedule, optimizer=optimizer)

        # case2: multiple lr, multiple opt
        if ~hasattr(self.config.learning_rate, 'type') and ~hasattr(self.config.optimizer, 'type'):
            assert isinstance(params, dict)
            keys = list(self.config.learning_rate.keys())
            assert all([k in keys for k in params.keys()])
            assert all([k in keys for k in self.config.optimizer.keys()])

            ret = dict(lr_schedule={}, optimizer={})

            for k in keys:
                opt_cfg = self.config.optimizer[k]
                lr_cfg = self.config.learning_rate[k]
                sub_params = params[k]

                lr_schedule = make_learningrate(lr_cfg)
                opt_cfg.params['lr'] = lr_schedule.base_lr
                optimizer = make_optimizer(opt_cfg, params=sub_params)

                ret['lr_schedule'][k] = lr_schedule
                ret['optimizer'][k] = optimizer
            return ret

        raise ValueError('Only support (single lr, single opt) and (multiple lr, multiple opt)')

    def evaluate(self, test_config=None, after_construct_launcher_callbacks=None):
        tl = self.build_launcher()['launcher']

        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        if test_config:
            if isinstance(test_config, config.AttrDict):
                pass
            elif isinstance(test_config, dict):
                test_config = config.AttrDict.from_dict(test_config)
            else:
                raise ValueError()
            dataloader = make_dataloader(test_config)
        else:
            dataloader = make_dataloader(self.config.data.test)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)

        tl.evaluate(dataloader, merge_dict(self.config.train, self.config.test))

        return dict(config=self.config, launcher=tl)

    def build_launcher(self):
        kwargs = dict(model_dir=self.args.model_dir)
        kwargs.update(dict(model=self.make_model().to(self.device)))
        kwargs.update(self.make_lr_optimizer(kwargs['model'].custom_param_groups()))
        tl = Launcher(**kwargs)

        return dict(config=self.config, launcher=tl)

    def build_callbacks(self):
        cbs = []
        if hasattr(self.config.train, 'callbacks'):
            for cfg in self.config.train.callbacks:
                cbs.append(make_callback(cfg))
        return cbs

    def run(self, after_construct_launcher_callbacks=None):
        if self.args.opts:
            self.config.update_from_list(self.args.opts)

        tl = self.build_launcher()['launcher']

        if self.args.use_wandb:
            name = self.args.model_dir
            tl.logger.init_wandb(project=self.args.project, name=name, wandb_dir=self.args.model_dir)

        kw_dataloader = self.make_dataloader()

        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        for c in self.build_callbacks():
            self.register_callback(c)

        for c in self._callbacks:
            tl.info(f'callback: {c}')
            tl.register_callback(c)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)

        tl.info('th sync bn: {}'.format(
            'True' if self.config.train.get('sync_bn', False) else 'False'))
        tl.info('external parameter: {}'.format(self.args.opts))

        tl.info(f'config: {self.config}')
        # start training
        tl.train_by_config(kw_dataloader['traindata_loader'],
                           config=merge_dict(self.config.train, self.config.test),
                           test_data_loader=kw_dataloader['testdata_loader'])

        return dict(config=self.config, launcher=tl)

    def run_with_dataloader(self,
                            train_dataloader,
                            test_dataloader=None,
                            after_construct_launcher_callbacks=None):
        if self.args.opts:
            self.config.update_from_list(self.args.opts)

        tl = self.build_launcher()['launcher']

        if self.args.use_wandb:
            name = self.args.model_dir
            tl.logger.init_wandb(project=self.args.project, name=name, wandb_dir=self.args.model_dir)

        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)

        tl.logger.info('th sync bn: {}'.format('on' if self.config.train.get('sync_bn', False) else 'off'))
        tl.logger.info('external parameter: {}'.format(self.args.opts))

        tl.train_by_config(train_dataloader,
                           config=merge_dict(self.config.train,
                                             self.config.test),
                           test_data_loader=test_dataloader
                           )

        return dict(config=self.config, launcher=tl)

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def reset_callbacks(self):
        self._callbacks.clear()

    def torch_compile(self, model):
        if 'torch_compile' in self.config.train:
            model = torch.compile(model, **self.config.train.torch_compile)
        return model

import argparse

import torch
import shutil
import os
from ever.core import config
from ever.core.builder import make_dataloader
from ever.core.builder import make_learningrate
from ever.core.builder import make_model
from ever.core.builder import make_optimizer
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
        self._cfg = None

    def __call__(self):
        return self

    @property
    def device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @property
    def args(self):
        return self._args

    @property
    def config(self):
        if self._cfg:
            return self._cfg
        cfg = config.import_config(self.args.config_path)
        self._cfg = config.AttrDict.from_dict(cfg)
        return self._cfg

    def make_model(self):
        if self.args.opts:
            self.config.update_from_list(self.args.opts)
        model = make_model(self.config.model)
        return model

    def make_dataloader(self):
        traindata_loader = make_dataloader(self.config.data.train)
        testdata_loader = make_dataloader(self.config.data.test) if 'test' in self.config.data else None
        return dict(traindata_loader=traindata_loader, testdata_loader=testdata_loader)

    def make_lr_optimizer(self, params):
        lr_schedule = make_learningrate(self.config.learning_rate)
        self.config.optimizer.params['lr'] = lr_schedule.base_lr
        optimizer = make_optimizer(self.config.optimizer, params=params)
        return dict(lr_schedule=lr_schedule, optimizer=optimizer)

    def run(self, after_construct_launcher_callbacks=None):
        tl = self.build_launcher()['launcher']

        kw_dataloader = self.make_dataloader()

        param_util.trainable_parameters(tl.model, tl.logger)
        param_util.count_model_parameters(tl.model, tl.logger)

        if after_construct_launcher_callbacks is not None:
            for f in after_construct_launcher_callbacks:
                f(tl)
        tl.logger.info('external parameter: {}'.format(self.args.opts))
        tl.train_by_config(kw_dataloader['traindata_loader'], config=merge_dict(self.config.train, self.config.test),
                           test_data_loader=kw_dataloader['testdata_loader'])

        return dict(config=self.config, launcher=tl)

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
        kwargs.update(self.make_lr_optimizer(kwargs['model'].parameters()))
        tl = Launcher(**kwargs)

        return dict(config=self.config, launcher=tl)

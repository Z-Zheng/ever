import functools
import os
import time
import types

import torch
from ever.interface.learning_rate import LearningRateBase
from ever.interface.module import ERModule
from torch.nn.utils import clip_grad

from . import to
from .checkpoint import CheckPoint
from .config import AttrDict
from .dist import reduce_loss_dict, get_rank
from .iterator import get_iterator
from .logger import Logger

__all__ = ['Launcher', 'scale_dict', 'average_dict', 'reduce_loss_dict']


class Launcher(object):
    def __init__(self,
                 model_dir,
                 model,
                 optimizer,
                 lr_schedule):
        self._model_dir = model_dir
        self._model = model
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._master = get_rank() == 0
        if self._master:
            self.init_model_dir()
            self._logger = Logger('EVER', use_tensorboard=self._master, tensorboard_logdir=model_dir)
            self._logger.on()
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._ckpt = CheckPoint(self)
        self._training = False

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def checkpoint(self):
        return self._ckpt

    @property
    def lr(self):
        return self._optimizer.param_groups[0]['lr']

    @property
    def logger(self):
        class _FakeLogger(object):
            def info(self, value):
                pass

            def register_train_log_hook(self, hook):
                pass

        if self._master:
            return self._logger
        else:
            return _FakeLogger()

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
            msg_dict = self._model(*d)

            losses = {k: v for k, v in msg_dict.items() if k.endswith('loss')}

            # scale losses by 1. / forward times
            if len(data) != 1:
                losses = scale_dict(losses, 1. / len(data))

            losses = average_dict(losses)
            total_loss = sum([e for e in losses.values()])

            self.backward(total_loss, self.optimizer)

            self.log_info_dict(data, losses, msg_dict, loss_dict)

        return loss_dict

    @staticmethod
    def log_info_dict(data, loss_tensor_dict, msg_dict, output_info_dict):
        # log losses
        with torch.no_grad():
            losses = reduce_loss_dict(loss_tensor_dict)
            for name, value in losses.items():
                if name not in output_info_dict:
                    output_info_dict[name] = 0.0
                output_info_dict[name] += value.item()
            output_info_dict['total_loss'] += sum(list(output_info_dict.values()))
        # extra log message
        log_dict = {k: v for k, v in msg_dict.items() if not k.endswith('loss')}
        with torch.no_grad():
            if len(log_dict) != 0:
                if len(data) != 1:
                    log_dict = scale_dict(log_dict, 1. / len(data))
                avg_log_dict = average_dict(log_dict)
                for name, value in avg_log_dict.items():
                    if name not in output_info_dict:
                        output_info_dict[name] = 0.0
                    output_info_dict[name] += value.item() if isinstance(value, torch.Tensor) else value

        return output_info_dict

    def apply_gradient(self):
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._update_lr()
        self._ckpt.step()

    def _update_lr(self):
        if isinstance(self._lr_schedule, LearningRateBase):
            self._lr_schedule.step(self._ckpt.global_step, self._optimizer)
        else:
            raise NotImplementedError()

    def train_iters(self,
                    train_data_loader,
                    test_data_loader=None,
                    **kwargs):
        num_iters = kwargs.get('num_iters', -1)
        forward_times = kwargs.get('forward_times', 1)
        eval_per_epoch = kwargs.get('eval_per_epoch', False)
        tensorboard_interval_step = kwargs.get('tensorboard_interval_step', 100)
        log_interval_step = kwargs.get('log_interval_step', 1)
        distributed = kwargs.get('distributed', False)
        summary_grads = kwargs.get('summary_grads', False)
        summary_weights = kwargs.get('summary_weights', False)
        iterator_type = kwargs.get('iterator_type', 'normal')
        save_ckpt_interval_epoch = kwargs.get('save_ckpt_interval_epoch', 1)
        eval_interval_epoch = kwargs.get('eval_interval_epoch', 1)

        iterator = get_iterator(iterator_type)(train_data_loader)

        call_backs = [(self._ckpt.save, save_ckpt_interval_epoch)]
        signal_loss_dict = dict()
        if eval_per_epoch:
            call_backs.append(
                (functools.partial(self.evaluate, test_data_loader, AttrDict.from_dict(kwargs)), eval_interval_epoch))
        while self._ckpt.global_step < num_iters:
            start = time.time()
            if distributed:
                iterator.set_seed_for_dist_sampler(self._ckpt.global_step)
            data_list = iterator.next(forward_times,
                                      call_backs=call_backs,
                                      is_master=self._master)
            data_time = time.time() - start
            self._model.train()
            loss_dict = self.compute_loss_gradient(data_list)
            signal_loss_dict = loss_dict.copy()
            # clip gradient
            if 'grad_clip' in self._optimizer.er_config:
                grad_clip_config = self._optimizer.er_config.get('grad_clip', dict(max_norm=35, norm_type=2))
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.module.parameters()),
                                          **grad_clip_config)

            if self._master:
                if summary_grads and self._ckpt.global_step % tensorboard_interval_step == 0:
                    self._logger.summary_grads(module=self.model.module, step=self._ckpt.global_step)

            self.apply_gradient()

            if self._master:
                time_cost = time.time() - start
                self._logger.train_log(step=self._ckpt.global_step,
                                       loss_dict=loss_dict,
                                       data_time=data_time,
                                       time_cost=time_cost,
                                       lr=self.lr,
                                       num_iters=num_iters,
                                       tensorboard_interval_step=tensorboard_interval_step,
                                       log_interval_step=log_interval_step)

                if summary_weights and self._ckpt.global_step % tensorboard_interval_step == 0:
                    self._logger.summary_weights(module=self.model.module, step=self._ckpt.global_step)

        return signal_loss_dict

    def train_by_config(self, train_data_loader, config, test_data_loader=None, ):
        self._training = True
        if config.get('resume_from_last', True):
            self.init()
        self.model.train()
        forward_times = config['forward_times'] if 'forward_times' in config else 1

        if self._master:
            self._logger.equation('batch_size_per_gpu', train_data_loader.batch_sampler.batch_size)
            self._logger.forward_times(forward_times)
            self._logger.approx_equation('num_epochs',
                                         round(config['num_iters'] * forward_times / len(train_data_loader), 1))
            self._logger.equation('num_iters', config['num_iters'])
            self._logger.equation('optimizer', self.optimizer)

            if isinstance(self.model, ERModule):
                model_extra_info = self.model.log_info()
                model_extra_info['model.type'] = self.model.__class__.__name__
            else:
                model_extra_info = self.model.module.log_info()
                model_extra_info['model.type'] = self.model.module.__class__.__name__

            for k, v in model_extra_info.items():
                self._logger.equation(k, v)

        signal_loss_dict = self.train_iters(train_data_loader, test_data_loader=test_data_loader, **config)

        if self._master:
            self._ckpt.save()
            if config.get('eval_after_train', True):
                self.evaluate(test_data_loader, config)
        return signal_loss_dict

    def init(self):
        if self._master:
            self.init_model_dir()
        self._ckpt.try_resume()

    def init_model_dir(self):
        os.makedirs(self._model_dir, exist_ok=True)

    def evaluate(self, data_loader, config=None):
        if not self._training:
            self.init()
        self._evaluate_fn(data_loader, config)

    def evaluate_last_ckpt(self, data_loader):
        self.init()
        self._evaluate_fn(data_loader)

    def _evaluate_fn(self, data_loader, config=None):
        raise NotImplementedError

    def backward(self, total_loss, optimizer, **kwargs):
        total_loss.backward()

    def override_evaluate(self, fn):
        self._evaluate_fn = types.MethodType(fn, self)

    def override_backward(self, fn):
        self.backward = types.MethodType(fn, self)


def scale_dict(input_dict, scale):
    for k, v in input_dict.items():
        input_dict[k] = v * scale
    return input_dict


def average_dict(input_dict):
    for k, v in input_dict.items():
        input_dict[k] = v.mean() if v.ndimension() != 0 else v
    return input_dict

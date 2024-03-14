import logging
import os
import time
from abc import abstractmethod, ABCMeta
from collections import deque

import wandb
from ever.core.dist import main_process_only
import numpy as np

logging.basicConfig(level=logging.INFO)


@main_process_only
def info(msg):
    if _logger is not None:
        _logger.info(msg)
    else:
        _default_logger.info(msg)


def get_logger(name=__name__, file_path=None, create_global=False):
    logger = logging.Logger(name)
    logger.setLevel(level=logging.INFO)

    logger.handlers = []
    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=logging.INFO)
    logger.addHandler(chlr)
    if file_path:
        fhlr = logging.FileHandler(file_path)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    if create_global:
        global _logger
        if _logger is None:
            _logger = logger
        return _logger
    else:
        return logger


_default_logger = get_logger('EVER')
_logger: logging.Logger = None


def get_console_file_logger(name, level, logdir):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []
    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=level)

    fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    global _logger
    if _logger is None:
        _logger = logger

    return _logger


class TrainLogHook(metaclass=ABCMeta):
    def __init__(self, interval_step=1):
        self.interval_step = interval_step

    def __call__(self,
                 current_step,
                 loss_dict,
                 learning_rate,
                 num_iters,
                 ):
        if current_step % self.interval_step == 0:
            self.after_iter(current_step, loss_dict, learning_rate, num_iters)

        if current_step == num_iters:
            self.after_train(current_step, loss_dict, learning_rate, num_iters)

    @abstractmethod
    def after_iter(self,
                   current_step,
                   loss_dict,
                   learning_rate,
                   num_iters,
                   ):
        return NotImplementedError

    @abstractmethod
    def after_train(self,
                    current_step,
                    loss_dict,
                    learning_rate,
                    num_iters,
                    ):
        return NotImplementedError


class Logger(object):
    def __init__(self,
                 name,
                 level=logging.INFO,
                 use_tensorboard=False,
                 tensorboard_logdir=None,
                 ):
        self._level = level
        self._logger = get_console_file_logger(name, level, tensorboard_logdir)
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and tensorboard_logdir is None:
            raise ValueError('logdir is not None if you use tensorboard')
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_w = SummaryWriter(tensorboard_logdir)
        self.smoothvalues = dict()

        self._train_log_hooks = []

    def finish(self):
        if self.use_wandb:
            wandb.finish()

    @property
    def use_wandb(self):
        return wandb.run is not None

    @main_process_only
    def init_wandb(self, project, name, wandb_dir, config=None):
        wandb.login()
        wandb.init(
            project=project,
            dir=wandb_dir,
            name=name,
            config=config,
        )

    @main_process_only
    def wandb_summary(self, loss_dict: dict, lr, step: int):
        assert self.use_wandb
        log_dict = {}
        for name, value in loss_dict.items():
            log_dict[name] = value

        if isinstance(lr, dict):
            for name, value in lr.items():
                log_dict[f'lr/{name}'] = value
        else:
            log_dict['lr'] = lr
        wandb.log(log_dict, step=step)

    def create_or_get_smoothvalues(self, value_dict: dict):
        for key, value in value_dict.items():
            if key not in self.smoothvalues:
                self.smoothvalues[key] = SmoothedValue(100)
            self.smoothvalues[key].add_value(value)

        return {key: self.smoothvalues[key].get_average_value() for key, _ in value_dict.items()}

    def info(self, value):
        self._logger.info(value)

    def on(self):
        self._logger.setLevel(self._level)
        self.use_tensorboard = True

    def off(self):
        self._logger.setLevel(100)
        self.use_tensorboard = False

    def summary_weights(self, module, step):
        if step % 100 == 0:
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                self.summary_w.add_histogram('weights/{}'.format(name), p.cpu().data.numpy(), step)

    def summary_grads(self, module, step):
        if step % 100 == 0:
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                self.summary_w.add_histogram('grads/{}'.format(name), p.grad.cpu().data.numpy(), step)

    def train_log(self,
                  step,
                  epoch,
                  loss_dict,
                  time_cost,
                  data_time,
                  lr,
                  num_iters,
                  tensorboard_interval_step=100,
                  log_interval_step=1):
        smooth_loss_dict = self.create_or_get_smoothvalues(loss_dict)
        loss_info = ''.join(
            ['{name} = {value}, '.format(name=name, value=str(round(value, 6)).ljust(6, '0')) for name, value in
             smooth_loss_dict.items()])
        step_info = f'step: {int(step)}({epoch}), '
        # eta
        smooth_time_cost = self.create_or_get_smoothvalues({'time_cost': time_cost})['time_cost']
        smooth_data_time = self.create_or_get_smoothvalues({'data_time': data_time})['data_time']
        if num_iters is not None:
            eta = (num_iters - step) * smooth_time_cost
            m, s = divmod(eta, 60)
            h, m = divmod(m, 60)
            eta_str = "%02d:%02d:%02d" % (h, m, s)
            time_cost_info = '({} sec / step, data: {} sec, eta: {})'.format(round(smooth_time_cost, 3),
                                                                             round(smooth_data_time, 3),
                                                                             eta_str)
        else:
            time_cost_info = '({} sec / step, data: {} sec)'.format(round(smooth_time_cost, 3),
                                                                    round(smooth_data_time, 3))

        if isinstance(lr, dict):
            lr_info = ''
            for k, v in lr.items():
                lr_info += f'{k}_lr = {str(round(v, 6))}, '
        else:
            lr_info = 'lr = {}, '.format(str(round(lr, 6)))
        msg = '{loss}{lr}{step}{time}'.format(loss=loss_info,
                                              step=step_info,
                                              lr=lr_info,
                                              time=time_cost_info)
        if step % log_interval_step == 0:
            self._logger.info(msg)
            if self.use_wandb:
                self.wandb_summary(smooth_loss_dict, lr, step)

        if self.use_tensorboard and step % tensorboard_interval_step == 0:
            self.train_summary(step, smooth_loss_dict, time_cost, lr)

        if len(self._train_log_hooks) != 0:
            for hook in self._train_log_hooks:
                hook(current_step=step,
                     loss_dict=smooth_loss_dict,
                     learning_rate=lr,
                     num_iters=num_iters)

    def train_summary(self, step, loss_dict, time_cost, lr):
        for name, value in loss_dict.items():
            self.summary_w.add_scalar('loss/{}'.format(name), float(value), global_step=step)

        self.summary_w.add_scalar('sec_per_step', float(time_cost), global_step=step)

        if isinstance(lr, dict):
            for name, v in lr.items():
                self.summary_w.add_scalar(f'learning_rate/{name}', float(v), global_step=step)
        else:
            self.summary_w.add_scalar('learning_rate', float(lr), global_step=step)

    def eval_log(self, metric_dict, step=None):
        for name, value in metric_dict.items():
            self._logger.info('[Eval] {name} = {value}'.format(name=name, value=np.round(value, 6)))
        if self.use_tensorboard:
            self.eval_summary(metric_dict, step)

    def eval_summary(self, metric_dict, step):
        if step is None:
            step = 1
        for name, value in metric_dict.items():
            if isinstance(value, float):
                self.summary_w.add_scalar('eval/{}'.format(name), value, global_step=step)
            elif isinstance(value, np.ndarray):
                for idx, nd_v in enumerate(value):
                    self.summary_w.add_scalar('eval/{}_{}'.format(name, idx), float(nd_v), global_step=step)
        self.summary_w.file_writer.flush()

    def forward_times(self, forward_times):
        self._logger.info('use {} forward and {} backward mode.'.format(forward_times, forward_times))

    def equation(self, name, value):
        self._logger.info('{name} = {value}'.format(name=name, value=value))

    def approx_equation(self, name, value):
        self._logger.info('{name} ~= {value}'.format(name=name, value=value))

    def register_train_log_hook(self, hook: TrainLogHook):
        assert isinstance(hook, TrainLogHook)
        self._train_log_hooks.append(hook)


def save_log(logger, checkpoint_name):
    logger.info('{} has been saved.'.format(checkpoint_name))


def restore_log(logger, checkpoint_name):
    logger.info('{} has been restored.'.format(checkpoint_name))


def eval_start(logger):
    logger.info('Start evaluation at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


def eval_progress(logger, cur, total):
    logger.info('[Eval] {}/{}'.format(cur, total))


def speed(logger, sec, unit='im'):
    logger.info('[Speed] {} s/{}'.format(sec, unit))


# ref to:
# https://github.com/facebookresearch/Detectron/blob/7c0ad88fc0d33cf0f698a3554ee842262d27babf/detectron/utils/logging.py#L41
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def get_median_value(self):
        return np.median(self.deque)

    def get_average_value(self):
        return np.mean(self.deque)

    def get_global_average_value(self):
        return self.total / self.count

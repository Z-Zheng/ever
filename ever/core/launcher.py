import os
import time
import types
from packaging import version

import torch

if version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.amp import GradScaler
else:
    from torch.cuda.amp import GradScaler

from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from ever.interface.callback import Callback
from ever.interface.callback import SaveCheckpointCallback
from ever.interface.callback import EvaluationCallback
from ever.interface.learning_rate import LearningRateBase

from . import to
from .checkpoint import CheckPoint
from .config import AttrDict
from .dist import reduce_loss_dict, is_main_process
from .iterator import get_iterator
from .logger import Logger
from .device import auto_device

__all__ = ['Launcher', ]


class Launcher(object):
    def __init__(self,
                 model_dir,
                 model,
                 optimizer,
                 lr_schedule,
                 mixed_precision,
                 ):
        if mixed_precision == 'fp32':
            self._mixed_precision = torch.float32
            self._amp = False
        elif mixed_precision == 'fp16':
            self._mixed_precision = torch.float16
            self._amp = True
        elif mixed_precision == 'bf16':
            self._mixed_precision = torch.bfloat16
            self._amp = True
        else:
            raise ValueError('unrecognized datatype, it should be one of [fp32, fp16, bf16].')

        self._model_dir = model_dir
        self._model = model
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._master = is_main_process()
        if self._master:
            self.init_model_dir()
            self._logger = Logger(
                'EVER',
                use_tensorboard=self._master,
                tensorboard_logdir=model_dir
            )
            self._logger.on()
        self._device = auto_device()

        self._ckpt = CheckPoint(self)
        self._training = False

        self._buffer = dict()

        self._callbacks = []

        if self._amp:
            if isinstance(optimizer, dict):
                self.scaler = {name: GradScaler() for name, _ in optimizer.items()}
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

    @property
    def is_main_process(self):
        return self._master

    def info(self, msg):
        if self._master:
            self._logger.info(msg)

    @property
    def use_wandb(self):
        if self._master:
            return self._logger.use_wandb
        else:
            return False

    @property
    def buffer(self):
        return self._buffer

    @property
    def model(self):
        return self._model

    @property
    def unwrapped_model(self):
        model = self._model
        if isinstance(self._model, DistributedDataParallel):
            model = model.module
        if version.parse(torch.__version__) < version.parse("2.0") or not hasattr(torch, "_dynamo"):
            return model
        if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
            model = model._orig_mod

        return model

    @property
    def model_without_ddp(self):
        if isinstance(self._model, DistributedDataParallel):
            return self._model.module
        else:
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
    def global_step(self):
        return self._ckpt.global_step

    def save_model(self, filename=None):
        if self._master:
            weights = self.model_without_ddp.state_dict()
            if filename is None:
                filename = self._ckpt.get_checkpoint_name(self.global_step)
            torch.save(weights, os.path.join(self.model_dir, filename))
        self.info(f'{filename} has been saved.')

    @property
    def lr(self):
        if isinstance(self._optimizer, dict):
            return {k: opt.param_groups[0]['lr'] for k, opt in self._optimizer.items()}
        else:
            return self._optimizer.param_groups[0]['lr']

    @property
    def logger(self):
        class _FakeLogger(object):
            def info(self, value):
                pass

            def register_train_log_hook(self, hook):
                pass

            def init_wandb(self, *args, **kwargs):
                pass

            def wandb_summary(self, *args, **kwargs):
                pass

            def finish(self):
                pass

        if self._master:
            return self._logger
        else:
            return _FakeLogger()

    def reset_callback(self):
        self._callbacks.clear()

    def register_callback(self, callback):
        assert isinstance(callback, Callback), f'{type(callback)} is not Callback'
        callback.set_launcher(self)
        self._callbacks.append(
            callback
        )

    def compute_loss_gradient(self, data, forward_times):
        with autocast(device_type='cuda', enabled=self._amp, dtype=self._mixed_precision):
            msg_dict = self._model(*data)
            losses = {k: v / forward_times for k, v in msg_dict.items() if k.endswith('loss')}

        self.unwrapped_model.backward(loss_dict=losses, amp=self._amp, scaler=self.scaler)

        return msg_dict

    @torch.no_grad()
    def log_info_dict(self, msg_dict):
        output_info_dict = {'total_loss': 0.0}
        # log losses
        loss_tensor_dict = {k: v for k, v in msg_dict.items() if k.endswith('loss')}
        losses = reduce_loss_dict(loss_tensor_dict)
        for name, value in losses.items():
            if name not in output_info_dict:
                output_info_dict[name] = 0.0
            output_info_dict[name] += value.item()
        output_info_dict['total_loss'] += sum(list(output_info_dict.values()))
        # extra log message
        log_dict = {k: v for k, v in msg_dict.items() if not k.endswith('loss')}
        if len(log_dict) != 0:
            avg_log_dict = average_dict(log_dict)
            for name, value in avg_log_dict.items():
                if name not in output_info_dict:
                    output_info_dict[name] = 0.0
                output_info_dict[name] += value.item() if isinstance(value, torch.Tensor) else value

        return output_info_dict

    def update_training_status(self):
        self._update_lr()
        self._ckpt.step()

    def _update_lr(self):
        if isinstance(self._lr_schedule, LearningRateBase):
            self._lr_schedule.step(self._ckpt.global_step, self._optimizer)
        elif isinstance(self._lr_schedule, dict):
            assert isinstance(self._optimizer, dict)
            for k, lr_s in self._lr_schedule.items():
                assert isinstance(lr_s, LearningRateBase)
                lr_s.step(self._ckpt.global_step, self._optimizer[k])
        else:
            raise NotImplementedError()

    def run_callbacks(self, stage_name):
        for f in self._callbacks:
            if getattr(f, stage_name):
                if f.only_master:
                    if self._master:
                        f.func()
                else:
                    f.func()

    def train_iters(self,
                    train_data_loader,
                    test_data_loader=None,
                    **kwargs):
        distributed = kwargs.get('distributed', False)

        num_iters = kwargs.get('num_iters', -1)
        assert num_iters > 0

        forward_times = kwargs.get('forward_times', 1)

        eval_per_epoch = kwargs.get('eval_per_epoch', False)
        eval_interval_epoch = kwargs.get('eval_interval_epoch', -1)
        eval_after_train = kwargs.get('eval_after_train', False)

        tensorboard_interval_step = kwargs.get('tensorboard_interval_step', 100)
        log_interval_step = kwargs.get('log_interval_step', 1)
        log_model_dir_interval_step = kwargs.get('task_log_interval_step', 500)

        summary_grads = kwargs.get('summary_grads', False)
        summary_weights = kwargs.get('summary_weights', False)

        iterator_type = kwargs.get('iterator_type', 'normal')

        save_ckpt_interval_epoch = kwargs.get('save_ckpt_interval_epoch', 1)

        dist_eval = kwargs.get('distributed_evaluate', False)

        iterator = get_iterator(iterator_type)(train_data_loader)

        self.register_callback(SaveCheckpointCallback(save_ckpt_interval_epoch))

        if eval_per_epoch or eval_after_train:
            if eval_per_epoch and eval_interval_epoch < 0:
                raise ValueError(
                    'eval_interval_epoch should be a positive number when eval_per_epoch = True')
            if not eval_per_epoch and eval_interval_epoch > 0:
                raise ValueError(
                    'eval_per_epoch should be True when eval_interval_epoch > 0')

            self.register_callback(
                EvaluationCallback(test_data_loader, eval_interval_epoch, not dist_eval,
                                   config=AttrDict.from_dict(kwargs),
                                   after_train=eval_after_train)
            )
        self._callbacks.sort(key=lambda callback: callback.prior)

        self.run_callbacks('before_train')

        signal_loss_dict = dict()
        while self._ckpt.global_step < num_iters:
            start = time.time()
            if distributed:
                iterator.set_seed_for_dist_sampler(self._ckpt.global_step)

            with torch.autograd.profiler.record_function('load_data'):
                data_list = iterator.next(forward_times,
                                          call_backs=self._callbacks,
                                          is_master=self._master)
            data_time = time.time() - start
            self._model.train()

            data = to.to_device(data_list, self._device)

            with torch.autograd.profiler.record_function('forward_backward'):
                if len(data) == 1:
                    msg_dict = self.compute_loss_gradient(data[0], 1)
                else:
                    for sub_data in data:
                        msg_dict = self.compute_loss_gradient(sub_data, len(data))

                self.unwrapped_model.apply_gradients(self.optimizer, self._amp, scaler=self.scaler)

            msg_dict = self.log_info_dict(msg_dict)
            signal_loss_dict = msg_dict.copy()

            if self._master:
                if summary_grads and self._ckpt.global_step % tensorboard_interval_step == 0:
                    self._logger.summary_grads(module=self.unwrapped_model,
                                               step=self._ckpt.global_step)

            with torch.autograd.profiler.record_function('update_lr_params'):
                self.update_training_status()

            if self._master:
                time_cost = time.time() - start
                epoch = iterator.epoch(forward_times)

                self._logger.train_log(
                    step=self._ckpt.global_step,
                    epoch=epoch,
                    loss_dict=msg_dict,
                    data_time=data_time,
                    time_cost=time_cost,
                    lr=self.lr,
                    num_iters=num_iters,
                    tensorboard_interval_step=tensorboard_interval_step,
                    log_interval_step=log_interval_step
                )
                if (log_model_dir_interval_step > 0) and (
                        self._ckpt.global_step % log_model_dir_interval_step == 0):
                    self._logger.info(self.model_dir)

                if summary_weights and self._ckpt.global_step % tensorboard_interval_step == 0:
                    self._logger.summary_weights(module=self.unwrapped_model,
                                                 step=self._ckpt.global_step)

        del iterator
        self.run_callbacks('after_train')
        self.logger.finish()
        return signal_loss_dict

    def train_by_config(self, train_data_loader, config, test_data_loader=None, ):
        self._training = True
        if config.get('resume_from_last', True):
            self.init()
        self._model.train()

        if hasattr(train_data_loader.sampler, 'indices'):
            # Subset sampler
            num_samples = len(train_data_loader.sampler.indices)
        else:
            num_samples = len(train_data_loader.dataset)

        if self._master:
            self._logger.info(f'mixed precision type: {self._mixed_precision}')
            self._logger.equation('num_samples', num_samples)
            self._logger.equation('batch_size_per_gpu', train_data_loader.batch_sampler.batch_size)
            self._logger.forward_times(config['forward_times'])
            self._logger.approx_equation('num_epochs',
                                         round(config['forward_times'] * config['num_iters'] / len(train_data_loader), 1))
            self._logger.equation('num_iters', config['num_iters'])
            self._logger.equation('optimizer', self.optimizer)

            model_extra_info = self.unwrapped_model.log_info()
            model_extra_info['model.type'] = self.unwrapped_model.__class__.__name__

            for k, v in model_extra_info.items():
                self._logger.equation(k, v)

        signal_loss_dict = self.train_iters(train_data_loader,
                                            test_data_loader=test_data_loader, **config)

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
        return self._evaluate_fn(data_loader, config)

    def evaluate_last_ckpt(self, data_loader):
        self.init()
        return self._evaluate_fn(data_loader)

    def _evaluate_fn(self, data_loader, config=None):
        raise NotImplementedError

    def override_evaluate(self, fn):
        self._evaluate_fn = types.MethodType(fn, self)


def scale_dict(input_dict, scale):
    for k, v in input_dict.items():
        input_dict[k] = v * scale
    return input_dict


def average_dict(input_dict):
    for k, v in input_dict.items():
        input_dict[k] = v.mean() if v.ndimension() != 0 else v
    return input_dict

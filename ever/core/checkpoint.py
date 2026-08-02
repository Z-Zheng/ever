import json
import os
from collections import OrderedDict

import torch
from ever.core.logger import save_log, restore_log


def is_checkpoint(obj):
    if isinstance(obj, CheckPoint):
        return True
    if isinstance(obj, OrderedDict) and all([
        CheckPoint.MODEL in obj,
        CheckPoint.OPTIMIZER in obj,
        CheckPoint.GLOBALSTEP in obj
    ]):
        return True
    return False


class CheckPoint:
    MODEL = 'model'
    OPTIMIZER = 'opt'
    SCALER = 'scaler'
    GLOBALSTEP = 'global_step'
    LASTCHECKPOINT = 'last'
    CHECKPOINT_NAME = 'checkpoint_info.json'

    def __init__(self, train_context=None):
        self._train_context = train_context
        self._global_step = 0
        self._json_log = {CheckPoint.LASTCHECKPOINT: dict(step=0, name='')}

        self._init_checkpoint_info_from_context()

    def set_global_step(self, value):
        if value >= 0:
            self._global_step = value
        else:
            raise ValueError('The global step must be larger than zero.')

    @property
    def global_step(self):
        return self._global_step

    def step(self):
        self._global_step += 1

    def set_context(self, train_context):
        self._train_context = train_context
        self._init_checkpoint_info_from_context()

    def save(self, filename=None):
        ckpt = OrderedDict({
            CheckPoint.MODEL: self._train_context.unwrapped_model.state_dict(),
            CheckPoint.GLOBALSTEP: self.global_step
        })
        if isinstance(self._train_context.optimizer, dict):
            ckpt[CheckPoint.OPTIMIZER] = {name: opt.state_dict() for name, opt in self._train_context.optimizer.items()}
        else:
            ckpt[CheckPoint.OPTIMIZER] = self._train_context.optimizer.state_dict()
        if self._train_context.scaler is not None:
            if isinstance(self._train_context.scaler, dict):
                ckpt[CheckPoint.SCALER] = {
                    name: scaler.state_dict() for name, scaler in self._train_context.scaler.items()
                }
            else:
                ckpt[CheckPoint.SCALER] = self._train_context.scaler.state_dict()

        if filename is None:
            filename = self.get_checkpoint_name(self.global_step)
        filepath = os.path.join(self._train_context.model_dir, filename)

        torch.save(ckpt, filepath)

        self._json_log[self.global_step] = filename
        if self.global_step > self._json_log[CheckPoint.LASTCHECKPOINT]['step']:
            self._json_log[CheckPoint.LASTCHECKPOINT]['step'] = self.global_step
            self._json_log[CheckPoint.LASTCHECKPOINT]['name'] = filename

        self.save_checkpoint_info(self._train_context.model_dir, self._train_context.logger)
        # log
        save_log(self._train_context.logger, filename)

    @staticmethod
    def load(filepath):
        ckpt = torch.load(filepath, map_location=torch.device("cpu"))

        return ckpt

    def save_checkpoint_info(self, model_dir, logger):
        with open(os.path.join(model_dir, CheckPoint.CHECKPOINT_NAME), 'w') as f:
            json.dump(self._json_log, f)
        save_log(logger, CheckPoint.CHECKPOINT_NAME)

    def cleanup_checkpoints(self, max_keep):
        if max_keep is None:
            return
        entries = []
        for key, filename in self._json_log.items():
            if key == CheckPoint.LASTCHECKPOINT:
                continue
            try:
                step = int(key)
            except (TypeError, ValueError):
                continue
            entries.append((step, key, filename))
        if len(entries) <= max_keep:
            return
        entries.sort(key=lambda item: item[0])
        to_remove = entries[:-max_keep]
        for _, key, filename in to_remove:
            filepath = os.path.join(self._train_context.model_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            if key in self._json_log:
                self._json_log.pop(key, None)
            else:
                self._json_log.pop(str(key), None)

    def try_resume(self):
        """ json -> ckpt_path -> ckpt -> launcher

        Returns:

        """
        if self._train_context is None:
            return
        # 1. json
        model_dir = self._train_context.model_dir
        json_log = self.load_checkpoint_info(model_dir)
        if json_log is None:
            return
        # 2. ckpt path
        last_path = os.path.join(self._train_context.model_dir, json_log[CheckPoint.LASTCHECKPOINT]['name'])
        # 3. load ckpt
        ckpt = self.load(last_path)
        # 4. resume
        self._train_context.unwrapped_model.load_state_dict(ckpt[CheckPoint.MODEL])

        if self._train_context.optimizer is not None:
            if isinstance(self._train_context.optimizer, dict):
                for name, opt in self._train_context.optimizer.items():
                    opt.load_state_dict(ckpt[CheckPoint.OPTIMIZER][name])
            else:
                self._train_context.optimizer.load_state_dict(ckpt[CheckPoint.OPTIMIZER])
        if CheckPoint.SCALER in ckpt and self._train_context.scaler is not None:
            if isinstance(self._train_context.scaler, dict):
                for name, scaler in self._train_context.scaler.items():
                    if name in ckpt[CheckPoint.SCALER]:
                        scaler.load_state_dict(ckpt[CheckPoint.SCALER][name])
            else:
                self._train_context.scaler.load_state_dict(ckpt[CheckPoint.SCALER])
        if self._train_context.checkpoint is not None:
            self._train_context.checkpoint.set_global_step(ckpt[CheckPoint.GLOBALSTEP])
        # log
        restore_log(self._train_context.logger, last_path)

    def _init_checkpoint_info_from_context(self):
        if self._train_context is None:
            return

        model_dir = self._train_context.model_dir
        json_file = self.load_checkpoint_info(model_dir)

        if json_file is None:
            return
        self._json_log = json_file

    @staticmethod
    def load_checkpoint_info(model_dir):
        json_path = os.path.join(model_dir, CheckPoint.CHECKPOINT_NAME)
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        return json_file

    @staticmethod
    def get_checkpoint_name(global_step):
        return 'checkpoint-{}.pth'.format(global_step)


def remove_module_prefix(model_state_dict):
    ret = {}
    safe_flag = False
    for k, v in model_state_dict.items():
        if ('module.' not in k) and ('_orig_mod.' not in k):
            safe_flag = True
            break
        if k.find('module.') == 0:
            k = k.replace('module.', '', 1)

        if k.find('_orig_mod.') == 0:
            k = k.replace('_orig_mod.', '', 1)

        ret[k] = v
    if safe_flag:
        return model_state_dict
    else:
        return ret


def load_model_state_dict_from_ckpt(filepath):
    try:
        ckpt = torch.load(filepath)
    except RuntimeError:
        ckpt = torch.load(filepath, map_location=lambda storage, loc: storage)
    statedict = ckpt[CheckPoint.MODEL]

    ret = remove_module_prefix(statedict)

    return ret


def remove_optimizer_in_ckpt(fp, new_fp=None):
    ckpt = CheckPoint.load(fp)
    if CheckPoint.OPTIMIZER in ckpt:
        ckpt.pop(CheckPoint.OPTIMIZER)
    torch.save(ckpt, fp if new_fp is None else new_fp)

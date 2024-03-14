import torch

from ever.core import config, checkpoint
from ever.core.builder import make_model
from ever.core.logger import get_logger
import os
from pathlib import Path

logger = get_logger(__name__)

__all__ = [
    'build_from_file',
    'build_and_load_from_file',
    'export_model'
]


def build_from_file(config_path):
    cfg = config.import_config(config_path)
    model = make_model(cfg['model'])
    return model


def build_and_load_from_file(config_path, checkpoint_path):
    model = build_from_file(config_path)
    model_state_dict = checkpoint.load_model_state_dict_from_ckpt(checkpoint_path)
    global_step = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)[
        checkpoint.CheckPoint.GLOBALSTEP]
    model.eval()
    model.load_state_dict(model_state_dict)
    logger.info('[Load params] from {}'.format(checkpoint_path))
    return model, global_step


def build_from_model_dir(model_dir, checkpoint_name=None):
    pkl_cfg = os.path.join(model_dir, 'config.pkl')
    py_cfg = os.path.join(model_dir, 'config.py')
    if os.path.exists(pkl_cfg):
        cfg_path = pkl_cfg
    elif os.path.exists(py_cfg):
        cfg_path = py_cfg
    else:
        raise FileNotFoundError('The config file is not found in model_dir.')

    if checkpoint_name is None:  # try the best and then the last
        if os.path.exists(os.path.join(model_dir, 'model-best.pth')):
            checkpoint_name = 'model-best.pth'
        else:
            fps = [str(fp) for fp in Path(model_dir).glob('checkpoint-*.pth')]

            def _key_fn(e):
                return int(os.path.basename(e).replace('checkpoint-', '').replace('.pth', ''))

            checkpoint_name = os.path.basename(sorted(fps, key=_key_fn)[-1])

    ckpt_path = os.path.join(model_dir, checkpoint_name)
    return build_and_load_from_file(cfg_path, ckpt_path)


def export_model(config_path, checkpoint_path, input_shape, output_path):
    model, gs = build_and_load_from_file(config_path, checkpoint_path)
    traced = torch.jit.trace(model, torch.ones(input_shape))
    torch.jit.save(traced, output_path)
    logger.info('[export model] to {}'.format(output_path))

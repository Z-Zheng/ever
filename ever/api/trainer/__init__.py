import argparse
import os
import shutil

from .dp_trainer import DPTrainer
from .th_ddp_trainer import THDDPTrainer
from .th_amp_ddp_trainer import THAMPDDPTrainer
from .trainer import Trainer

TRAINER = dict(
    th_ddp=THDDPTrainer,
    th_amp_ddp=THAMPDDPTrainer,
    dp=DPTrainer,
    base=Trainer,
)


def get_default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', default=None, type=str,
                        help='path to config file')
    parser.add_argument('--model_dir', default=None, type=str,
                        help='path to model directory')
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument('--trainer', default='th_ddp', type=str,
                        help='path to model directory')
    # apex
    parser.add_argument('--opt_level', type=str, default='O0', help='O0, O1, O2, O3')
    parser.add_argument('--keep_batchnorm_fp32', type=bool, default=None, help='')
    # command line options
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def initialize_workspace(config_path, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    if config_path.endswith('.py'):
        shutil.copy(config_path,
                    os.path.join(model_dir, 'config.py'))
    else:
        cfg_path_segs = ['configs'] + config_path.split('.')
        cfg_path_segs[-1] = cfg_path_segs[-1] + '.py'
        shutil.copy(os.path.join(os.path.curdir, *cfg_path_segs),
                    os.path.join(model_dir, 'config.py'))


def get_trainer(trainer_name=None, parser=None):
    if parser is None:
        parser = get_default_parser()
    args = parser.parse_args()
    # check args
    assert args.config_path is not None, 'The `config_path` is needed'
    assert args.model_dir is not None, 'The `model_dir` is needed'

    # initialize directory
    initialize_workspace(args.config_path, args.model_dir)

    # compatible with torchrun and torch.distributed.launch
    if args.local_rank is None:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if trainer_name is None:
        trainer_name = args.trainer

    if trainer_name == 'apex_ddp':
        from .apex_ddp_trainer import ApexDDPTrainer
        TRAINER.update(dict(apex_ddp=ApexDDPTrainer))

    return TRAINER[trainer_name](args)

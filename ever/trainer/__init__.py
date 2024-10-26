import argparse
import os

from .th_ddp_trainer import THDDPTrainer, THDDPGANTrainer
from .trainer import Trainer

TRAINER = dict(
    gan_th_ddp=THDDPGANTrainer,
    th_ddp=THDDPTrainer,
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
                        help='type of trainer')
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='whether to find unused parameters')
    parser.add_argument('--mixed_precision', default='fp32', type=str,
                        help='datatype', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--use_wandb', action='store_true',
                        help='whether to use wandb for logging')
    parser.add_argument('--project', default=None, type=str,
                        help='Project name for init wandb')

    # command line options
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def get_trainer(trainer_name=None, parser=None, return_args=False):
    if parser is None:
        parser = get_default_parser()
    args = parser.parse_args()
    # check args
    assert args.config_path is not None, 'The `config_path` is needed'
    assert args.model_dir is not None, 'The `model_dir` is needed'

    if args.use_wandb:
        assert args.project is not None, '`project` is needed if you use wandb'

    # compatible with torchrun and torch.distributed.launch
    if args.local_rank is None:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if trainer_name is None:
        trainer_name = args.trainer

    if return_args:
        return TRAINER[trainer_name](args), args
    else:
        return TRAINER[trainer_name](args)

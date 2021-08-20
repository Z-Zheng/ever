from .dp_trainer import DPTrainer
from .th_ddp_trainer import THDDPTrainer
from .trainer import Trainer

TRAINER = dict(
    th_ddp=THDDPTrainer,
    dp=DPTrainer,
    base=Trainer,
)


def get_trainer(trainer_name):
    if trainer_name == 'apex_ddp':
        from .apex_ddp_trainer import ApexDDPTrainer
        TRAINER.update(dict(apex_ddp=ApexDDPTrainer))
    if trainer_name =='th_amp_ddp':
        from .th_amp_ddp_trainer import THAMPDDPTrainer
        TRAINER.update(dict(th_amp_ddp=THAMPDDPTrainer))

    return TRAINER[trainer_name]

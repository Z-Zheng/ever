from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

from ..core import registry

registry.OPT.register('sgd', SGD)
registry.OPT.register('adam', Adam)
registry.OPT.register('adamw', AdamW)
try:
    from apex.optimizers import FusedAdam

    registry.OPT.register('fused_adam', FusedAdam)
except ImportError:
    pass

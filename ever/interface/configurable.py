from ever.core.config import AttrDict


class ConfigurableMixin(object):
    """
    Usage 1: for torch.nn.Module
    >>> import torch.nn as nn
    >>> class Custom(nn.Module, ConfigurableMixin):
    >>>     def __init__(self, config:AttrDict):
    >>>         super(Custom,self).__init__()
    >>>         ConfigurableMixin.__init__(self, config)
    >>>     def forward(self, *input):
    >>>         pass
    >>>     def set_default_config(self):
    >>>         self.config.update(dict())
    """

    def __init__(self, config):
        self._cfg = AttrDict(

        )
        self.set_default_config()
        self._cfg.update(config)

    def set_default_config(self):
        raise NotImplementedError

    @property
    def config(self):
        return self._cfg

    @property
    def cfg(self):
        # alias of config
        return self._cfg

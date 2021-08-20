from . import registry

__all__ = ['make_dataloader', 'make_optimizer', 'make_learningrate', 'make_model']


def make_optimizer(config, params):
    opt_type = config['type']
    if opt_type in registry.OPT:
        opt = registry.OPT[opt_type](params=params, **config['params'])
        opt.er_config = config
    else:
        raise ValueError('{} is not support now.'.format(opt_type))
    return opt


def make_learningrate(config):
    lr_type = config['type']
    if lr_type in registry.LR:
        lr_module = registry.LR[lr_type]
        return lr_module(**config['params'])
    else:
        raise ValueError('{} is not support now.'.format(lr_type))


def make_dataloader(config):
    dataloader_type = config['type']
    if dataloader_type in registry.DATALOADER:
        data_loader = registry.DATALOADER[dataloader_type](config['params'])
    else:
        raise ValueError('{} is not support now.'.format(dataloader_type))

    return data_loader


def make_model(config):
    model_type = config['type']
    if model_type in registry.MODEL:
        model = registry.MODEL[model_type](config['params'])
    else:
        raise ValueError('{} is not support now.'.format(model_type))
    return model

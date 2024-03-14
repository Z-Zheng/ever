from . import registry

__all__ = ['make_dataloader', 'make_optimizer', 'make_learningrate', 'make_model', 'make_callback']


def make_callback(config):
    callback_type = config['type']
    if callback_type in registry.CALLBACK:
        callback = registry.CALLBACK[callback_type](**config['params'])
    else:
        raise ValueError('{} is not support now.'.format(callback_type))
    return callback


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
    elif dataloader_type in registry.DATASET:
        dataset = registry.DATASET[dataloader_type](config['params'])
        data_loader = dataset.to_dataloader()
    else:
        raise ValueError('{} is not support now.'.format(dataloader_type))

    return data_loader


def make_model(config):
    from ever.interface import ERModule
    from torch.nn import Module
    model_type = config['type']
    if model_type in registry.MODEL:
        if issubclass(registry.MODEL[model_type], ERModule):
            model = registry.MODEL[model_type](config['params'])
        elif issubclass(registry.MODEL[model_type], Module):
            model = registry.MODEL[model_type](**config['params'])
        else:
            raise ValueError(f'unsupported model class: {registry.MODEL[model_type]}')
    else:
        raise ValueError(
            '{} is not support now. This model seems not to be registered via @er.registry.MODEL.register()'.format(model_type))
    return model

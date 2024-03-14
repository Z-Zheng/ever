# modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/registry.py
import logging
import importlib
import sys

logging.basicConfig(level=logging.INFO)

__all__ = ['Registry', 'LR', 'OPT', 'DATALOADER', 'MODEL', 'LOSS', 'OP']


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _register_generic(module_dict, module_name, module, override=False):
    module_name = module_name if module_name else module.__name__
    if not override:
        if module_name in module_dict:
            logging.warning('{} has been in module_dict.'.format(module_name))
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    3): used as decorator when declaring the module named via __name__:
        @some_registry.register()
        def foo():
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name=None, module=None, override=False):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module, override)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn, override)
            return fn

        return register_fn


def register_dir(dir_name):
    import importlib
    import os
    for root, dirs, files in os.walk(os.path.join(os.path.curdir, dir_name)):
        if os.path.basename(root).startswith('_'):
            continue
        x = root.split(os.path.sep)
        prefix = '.'.join(x[1:])
        py_files = [f for f in files if f.endswith('.py') and not f.startswith('_')]
        for pyf in py_files:
            importlib.import_module('{}.{}'.format(prefix, pyf.replace('.py', '')))


def register_file(file_path):
    return _import_file('ever.custom', file_path)


def register_callbacks():
    register_dir('callback')


def register_modules():
    register_dir('module')


def register_dataloaders():
    register_dir('data')


def register_all():
    register_dataloaders()
    register_modules()
    register_callbacks()


LR = Registry()
OPT = Registry()
DATALOADER = Registry()
MODEL = Registry()
LOSS = Registry()
OP = Registry()
CALLBACK = Registry()
DATASET = Registry()

import importlib


class ConfigStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def config_from_file(filepath):
    module_path=filepath.split('.')[0]
    module_path=module_path.replace('/','.')
    mod = importlib.import_module(module_path)
    cfg_dict = {}
    for name, val in mod.__dict__.items():
        if not name.startswith('__'):
            cfg_dict[name] = val

    cfg = ConfigStruct(**cfg_dict)
    return cfg

from typing import Dict, Any
from copy import deepcopy
import importlib


def module_from_config(config: Dict[str, Any]):
    """This function is used to instantiate a module from a config dict.
    The config dict should contain a key "class" which specifies the
    full import path of the module to be instantiated. The remaining
    keys in the config dict are passed as kwargs to the constructor of
    the module. If a key in the config dict is itself a config dict,
    it is assumed to be a submodule and is recursively instantiated
    using this function.
    If the config dict contains a key "constructor", the value of this
    key is used as the name of the constructor to be called. This is
    useful if the module is to be instantiated with a classmethod.

    Args:
        config (Dict[str, Any]): The config dict.

    Returns:
        _type_: The instantiated module.
    """
    config = deepcopy(config)

    import_path = config.pop("class").split(".")
    module_cls = getattr(
        importlib.import_module(".".join(import_path[:-1])), import_path[-1]
    )

    constructor_name = config.pop("constructor", None)
    if constructor_name is not None:
        module_cls = getattr(module_cls, constructor_name)

    module_kwargs = {}

    for arg_name, arg_value in config.items():
        if isinstance(arg_value, dict) and "class" in arg_value:
            submodule_config = arg_value
            submodule = module_from_config(submodule_config)
            module_kwargs[arg_name] = submodule
        else:
            module_kwargs[arg_name] = arg_value

    return module_cls(**module_kwargs)

import importlib


def dynamic_import(object_path: str):
    """Import an object from an import path dynamically.

    Args:
        object_path (str): fully classified path to the object to be imported
    """
    module_name, func_name = object_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

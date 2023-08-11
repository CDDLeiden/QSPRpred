import importlib


def import_class(class_path):
    """Import a class from a string path."""
    class_name = class_path.split(".")[-1]
    module_name = class_path.replace(f".{class_name}", "")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

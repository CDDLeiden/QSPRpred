from _pytest.doctest import DoctestItem, DoctestModule


# pylint: disable=unused-argument
def pytest_collection_modifyitems(config, items):
    """pytest hook which gets called after collection is completed"""
    for item in items:
        if (
            isinstance(item, DoctestItem) and isinstance(item.parent, DoctestModule) and
            hasattr(item.parent.module, "gpu")
        ):

            item.own_markers = [item.parent.module.gpu.mark]

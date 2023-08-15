import os

from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.warning")

__version__ = "2.1.0"
if os.path.exists(os.path.join(os.path.dirname(__file__), '_version.py')):
    from ._version import version
    __version__ = version

VERSION = __version__
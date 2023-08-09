from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.warning")

__version__ = "2.0.1"
VERSION = __version__

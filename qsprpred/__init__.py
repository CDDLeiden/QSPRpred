from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.warning")

__version__ = "2.1.0"
VERSION = __version__

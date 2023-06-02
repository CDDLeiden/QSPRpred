"""
__init__.py

Created by: Martin Sicho
On: 06.04.22, 16:51
"""
from rdkit import rdBase

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')

__version__ = "2.0.0.dev2"
VERSION = __version__

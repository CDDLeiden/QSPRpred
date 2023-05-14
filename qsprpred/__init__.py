"""
__init__.py

Created by: Martin Sicho
On: 06.04.22, 16:51
"""
from rdkit import rdBase

from .about import VERSION

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')

__version__ = VERSION

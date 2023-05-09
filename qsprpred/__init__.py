"""
__init__.py

Created by: Martin Sicho
On: 06.04.22, 16:51
"""
from rdkit import rdBase
from importlib.util import find_spec

from .about import VERSION

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')

if find_spec('torch') is not None:
    import torch
    torch.set_num_threads(1)

    DEFAULT_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DEFAULT_GPUS = (0,)

__version__ = VERSION

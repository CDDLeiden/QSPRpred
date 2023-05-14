"""
__init__.py

Created by: Martin Sicho
On: 12.05.23, 16:38
"""

import torch
torch.set_num_threads(1)

DEFAULT_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEFAULT_GPUS = (0,) if torch.cuda.is_available() else -1

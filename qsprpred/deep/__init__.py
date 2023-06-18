"""
__init__.py

Created by: Martin Sicho
On: 12.05.23, 16:38
"""

import pkg_resources
import torch

# set default number of threads to 1
torch.set_num_threads(1)
# set default device to GPU if available
DEFAULT_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
DEFAULT_GPUS = (0, )
# set default parameter search space path
SSPACE = pkg_resources.resource_filename("qsprpred.deep", "models/search_space.json")

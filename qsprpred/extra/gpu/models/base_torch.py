from abc import ABC, abstractmethod

import torch

from .base import QSPRModelGPU

# set default number of threads to 1
torch.set_num_threads(1)
# set default device to GPU if available
DEFAULT_TORCH_DEVICE = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)
DEFAULT_TORCH_GPUS = (0,)


class QSPRModelPyTorchGPU(QSPRModelGPU, ABC):
    @abstractmethod
    def getGPUs(self):
        pass

    @abstractmethod
    def setGPUs(self, gpus: list[int]):
        pass

    @abstractmethod
    def getDevice(self) -> torch.device:
        pass

    @abstractmethod
    def setDevice(self, device: str):
        pass

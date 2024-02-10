from abc import ABC, abstractmethod

from qsprpred.models import QSPRModel


class QSPRModelGPU(QSPRModel, ABC):
    @abstractmethod
    def getGPUs(self):
        pass

    @abstractmethod
    def setGPUs(self, gpus: list[int]):
        pass

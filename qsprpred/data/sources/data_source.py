from abc import ABC, abstractmethod

from qsprpred import TargetProperty
from qsprpred.data import MoleculeTable, QSPRDataset


class DataSource(ABC):
    @abstractmethod
    def getData(self, name: str | None = None, **kwargs) -> MoleculeTable:
        pass

    def getDataSet(
        self,
        target_props: list[TargetProperty | dict],
        name: str | None = None,
        **kwargs
    ) -> QSPRDataset:
        mt = self.getData(name, **kwargs)
        name = name or mt.name
        return QSPRDataset.fromMolTable(mt, target_props, name, **kwargs)

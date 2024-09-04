from abc import ABC, abstractmethod

from qsprpred import TargetProperty
from qsprpred.data import QSPRDataset, MoleculeTable


class DataSource(ABC):
    """General definition of a data source. It is essentially a factory
    for creating `MoleculeTable` and `QSPRDataset` instances.
    """

    @abstractmethod
    def getData(self, name: str | None = None, **kwargs) -> MoleculeTable:
        """Get the molecule data from the source as a `MoleculeTable`.

        Args:
            name (str, optional): The name of the dataset.
            kwargs: Additional keyword arguments to pass to the method.

        Returns:
            MoleculeTable: The molecule data.
        """

    def getDataSet(
            self,
            target_props: list[TargetProperty | dict],
            name: str | None = None,
            **kwargs
    ) -> QSPRDataset:
        """Get the dataset from the source as a `QSPRDataset`. Essentially
        just creates a `QSPRDataset` from the `MoleculeTable` obtained from
        the `getData` method.

        Args:
            target_props (list[TargetProperty | dict]):
                The target properties to add.
            name (str, optional):
                The name of the dataset.
            kwargs:
                Additional keyword arguments to pass to the `getData`
                method and the `QSPRDataset` constructor.

        Returns:
            QSPRDataset: The dataset.
        """
        mt = self.getData(name, **kwargs)
        name = name or mt.name
        return QSPRDataset.fromMolTable(mt, target_props, name=name, path=mt.rootDir,
                                        **kwargs)

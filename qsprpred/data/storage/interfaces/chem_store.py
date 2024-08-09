from abc import ABC, abstractmethod
from typing import Iterable, Generator, Any

from qsprpred.data.chem.identifiers import ChemIdentifier
from qsprpred.data.chem.standardizers import ChemStandardizer
from qsprpred.data.processing.mol_processor import MolProcessor
from qsprpred.data.storage.interfaces.mol_processable import MolProcessable
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class ChemStore(PropertyStorage, MolProcessable, ABC):

    @property
    @abstractmethod
    def smilesProp(self) -> str:
        """Get the name of the property that contains the SMILES strings."""

    @property
    @abstractmethod
    def standardizer(self) -> ChemStandardizer:
        """
        Get the standardizer used by the store.

        :return: `ChemStandardizer` instance
        """

    @property
    @abstractmethod
    def identifier(self) -> ChemIdentifier:
        """
        Get the identifier used by the store.

        :return: `ChemIdentifier` instance
        """

    @property
    def n_mols(self) -> int:
        """Number of molecules in storage."""
        return self.get_mol_count()

    @property
    def smiles(self) -> Generator[str, None, None]:
        return (x.smiles for x in self)

    @abstractmethod
    def get_mol(self, mol_id: str) -> StoredMol:
        """
        Get a molecule from the store using its ID.

        :param mol_id: identifier of the molecule to search
        :return: instance of `StoredMol`
        """

    @abstractmethod
    def add_mols(self, smiles: Iterable[str], props: dict[str, list] | None = None,
                 *args, **kwargs) -> list[StoredMol]:
        """
        Add a molecule to the store. This method should not perform any standardization or identifier calculation. The `add_mol_from_smiles` method should be used instead if automatic standardization and identification should be performed before storage.

        :param smiles: molecule to add as SMILES
        :param mol_id: identifier of the molecule to add as determined by `self.identifier`
        :param metadata: additional metadata to store with the molecule
        :return: `StoredMol` instance of the added molecule

        :raises: `ValueError` if the molecule cannot be added
        """

    @abstractmethod
    def remove_mol(self, mol_id):
        """
        Remove a molecule from the store.

        :param mol_id: identifier of the molecule to remove
        :return:
        """

    @abstractmethod
    def get_mol_ids(self) -> tuple[str]:
        """
        Get all molecule IDs in the store.

        :return: list of molecule IDs
        """

    @abstractmethod
    def get_mol_count(self):
        """
        Get the number of molecules in the store.

        :return: number of molecules
        """

    @abstractmethod
    def iter_mols(self) -> Generator[StoredMol, None, None]:
        """
        Iterate over all molecules in the store.

        :return: iterator over `StoredMol` instances
        """

    @abstractmethod
    def iterChunks(
            self,
            size: int | None = None,
            on_props: list | None = None
    ) -> Generator[list[StoredMol], None, None]:
        """
        Iterate over chunks of molecules across the store.

        :return: an iterable of lists of stored molecules
        """

    def __len__(self):
        return self.get_mol_count()

    def __contains__(self, item):
        return item in self.get_mol_ids()

    def __iter__(self):
        return self.iter_mols()

    def __getitem__(self, item):
        return self.get_mol(item)

    def __delitem__(self, key):
        return self.remove_mol(key)

    def __bool__(self):
        return len(self) > 0

    def processMols(
            self,
            processor: MolProcessor,
            proc_args: Iterable[Any] | None = None,
            proc_kwargs: dict[str, Any] | None = None,
            add_props: Iterable[str] | None = None,
    ) -> Generator:
        pass

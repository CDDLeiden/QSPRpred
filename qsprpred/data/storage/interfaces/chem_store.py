from abc import ABC, abstractmethod
from typing import Iterable, Generator, Literal

import pandas as pd
from rdkit import Chem

from qsprpred.data.chem.identifiers import Identifiable
from qsprpred.data.chem.standardizers.base import Standardizable
from qsprpred.data.storage.interfaces.mol_processable import MolProcessable
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class ChemStore(PropertyStorage, MolProcessable, Identifiable, Standardizable, ABC):
    """Interface for storing and managing chemical data."""

    @property
    @abstractmethod
    def smilesProp(self) -> str:
        """Get the name of the property that contains the SMILES strings."""

    @property
    def n_mols(self) -> int:
        """Number of molecules in storage."""
        return self.get_mol_count()

    @property
    def smiles(self) -> Generator[str, None, None]:
        """Generator of SMILES strings of all molecules in storage."""
        return (x.smiles for x in self)

    @abstractmethod
    def get_mol(self, mol_id: str) -> StoredMol:
        """Get a molecule from the store using its ID.

        Args:
            mol_id (str): identifier of the molecule to search
        
        Returns:
            StoredMol: instance of the molecule
        """

    @abstractmethod
    def add_mols(self, smiles: Iterable[str], props: dict[str, list] | None = None,
                 *args, **kwargs) -> list[StoredMol]:
        """Add a molecule to the store. 
        
        This method should not perform any standardization or identifier calculation.
        The `add_mol_from_smiles` method should be used instead if automatic 
        standardization and identification should be performed before storage.

        Args:
            smiles (Iterable[str]): molecules to add as SMILES
            props (dict, optional): additional metadata to store with the molecules
            args: Additional positional arguments to be passed to each molecule.
            kwargs: Additional keyword arguments to be passed to each molecule.

        Returns:
            list[StoredMol]: instances of the added molecules
            
        Raises:
            ValueError: if the molecules cannot be added
        """

    @abstractmethod
    def remove_mol(self, mol_id: str):
        """Remove a molecule from the store.
        
        Args:
            mol_id (str): identifier of the molecule to remove
        """

    @abstractmethod
    def get_mol_ids(self) -> tuple[str]:
        """Get all molecule IDs in the store.

        Returns:
            tuple[str]: molecule IDs
        """

    @abstractmethod
    def get_mol_count(self):
        """Get the number of molecules in the store.

        Returns:
            (int) number of molecules
        """

    @abstractmethod
    def iter_mols(self) -> Generator[StoredMol, None, None]:
        """Iterate over all molecules in the store.

        Returns:
            iterator over `StoredMol` instances
        """

    @abstractmethod
    def iterChunks(
            self,
            size: int | None = None,
            on_props: list | None = None,
            chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
    ) -> Generator[list[StoredMol | str | Chem.Mol | pd.DataFrame], None, None]:
        """Iterate over chunks of molecules across the store.

        Args:
            size (int, optional): The size of the chunks.
            on_props (list, optional): The properties to include in the chunks.
            chunk_type (str, optional): The type of chunks to yield.
            
        Returns:
            an iterable of lists of stored molecules
        """

    @abstractmethod
    def apply(
            self,
            func: callable,
            func_args: list | None = None,
            func_kwargs: dict | None = None,
            on_props: tuple[str, ...] | None = None,
            chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
    ) -> Generator[Iterable[StoredMol | str | Chem.Mol | pd.DataFrame], None, None]:
        """Apply a function on all or selected properties of the chunks of data.
        The requested chunk type is supplied as the first positional argument
        to the function. Properties are attached to it as appropriate.
        The format of the properties is up to the downstream implementation, but they
        should be attached to the objects in chunks somehow.

        Args:
            func (callable): The function to apply.
            func_args (list, optional): The positional arguments of the function.
            func_kwargs (dict, optional): The keyword arguments of the function.
            on_props (list, optional): The properties to apply the function on.
            chunk_type (str, optional): The type of chunks to yield.
            
        Returns:
            A generator that yields the results of the function applied to each chunk.
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

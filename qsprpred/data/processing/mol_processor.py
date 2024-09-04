"""Abstract class that defines a simple callback interface to process molecules."""

from abc import ABC, abstractmethod
from typing import Any

from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class MolProcessor(ABC):
    """A callable that processes a list of molecules either specified as strings, RDKit
    molecules, or `StoredMol` instances. The processor can also accept additional
    properties related to the molecules if specified by the caller.
    """

    @abstractmethod
    def __call__(
            self,
            mols: list[str | Chem.Mol | StoredMol],
            *args,
            props: dict[str, list] | None = None,
            **kwargs
    ) -> Any:
        """Process molecules.

        Args:
            mols (list[str | Mol | StoredMol]):
                A list of SMILES or RDKit molecules to process.
            props (dict):
                A dictionary of properties related to the molecules to process. The
                dictionary uses property names as keys and lists of values as values.
                Each value in the list corresponds to a molecule in the list of
                molecules. Thus, the length of the list of values for each property
                can be expected to be the same as the length of the list of molecules.
                However, depending on the context, the properties may not be present
                and instead can be accessed from the `StoredMol` instances passed in
                the `mols` argument.
            args:
                Additional positional arguments.
            kwargs:
                Additional keyword arguments.

        Returns:
            Any: The result of the processing.
        """

    @property
    @abstractmethod
    def supportsParallel(self) -> bool:
        """Whether the processor supports parallel processing."""

    @property
    def requiredProps(self) -> list[str]:
        """The properties required by the processor. This is to inform the caller
        that the processor requires certain properties to be passed to the
        `__call__` method or via the `props` attribute of `StoredMol` instances.
        """
        return []


class MolProcessorWithID(MolProcessor, ABC):
    """A processor that requires a unique identifier for each molecule. Callers are
    instructed to pass this property with the `requiredProps` attribute.

    Attributes:
        idProp (str):
            The name of the passed property that contains
            the molecule's unique identifier.
    """

    def __init__(self, id_prop: str | None = None):
        """
        Initialize the processor with the name of the property that contains the
        molecule's unique identifier.

        Args:
            id_prop (str):
                Name of the property that contains the molecule's unique identifier.
                Defaults to "QSPRID".
        """
        self.idProp = id_prop if id_prop else "ID"

    def iterMolsAndIDs(self, mols, props: dict[str, list] | None):
        """Iterate over molecules and their corresponding IDs regardless of the input
        molecule format. This is just a helper function that will detect the input
        and yield the molecule and its ID.

        Args:
            mols (list[str | Mol | StoredMol]):
                A list of SMILES or RDKit molecules to process.
            props (dict):
                An optional dictionary of properties
                related to the molecules to process.
        Returns:
            tuple[Mol, str]: A tuple of the molecules and their IDs.
        """
        for idx, mol in enumerate(mols):
            if isinstance(mol, StoredMol):
                yield mol.as_rd_mol(), mol.id
            else:
                mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
                yield mol, props[self.idProp][idx]

    @property
    def requiredProps(self) -> list[str]:
        return [self.idProp]

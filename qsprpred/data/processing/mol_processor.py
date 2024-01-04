"""Abstract class that defines a simple callback interface to process molecules.
"""

from abc import ABC, abstractmethod
from typing import Any

from rdkit.Chem import Mol


class MolProcessor(ABC):
    """Processes either SMILES or RDKit molecules and  returns any result."""

    @abstractmethod
    def __call__(
        self, mols: list[str | Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> Any:
        """Process molecules.

        Args:
            mols (list[str | Mol]):
                A list of SMILES or RDKit molecules to process.
            props (dict):
                A dictionary of properties related to the molecules to process.
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
        """The properties required by the processor."""
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
        Initialize the processor.

        Args:
            id_prop (str):
                Name of the property that contains the molecule's unique identifier.
                Defaults to "QSPRID".
        """
        self.idProp = id_prop if id_prop else "QSPRID"

    @property
    def requiredProps(self) -> list[str]:
        return [self.idProp]

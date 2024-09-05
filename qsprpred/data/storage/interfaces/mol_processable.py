from abc import ABC, abstractmethod
from typing import Any, Generator, Iterable, Literal

from qsprpred.data.processing.mol_processor import MolProcessor


class MolProcessable(ABC):
    """Interface for processing molecules."""
    @abstractmethod
    def processMols(
        self,
        processor: MolProcessor,
        proc_args: tuple[Any, ...] | None = None,
        proc_kwargs: dict[str, Any] | None = None,
        mol_type: Literal["smiles", "mol", "rdkit"] = "mol",
        add_props: Iterable[str] | None = None,
    ) -> Generator[Any, None, None]:
        """Process the molecules in this instance with a given `MolProcessor`.

        Args:
            processor (MolProcessor):
                The processor to use.
            proc_args (tuple, optional):
                Additional arguments to pass to the processor.
            proc_kwargs (dict, optional):
                Additional keyword arguments to pass to the processor.
            mol_type (str, optional):
                The type of molecule to process.
            add_props (list, optional):
                Additional properties to add to the dataset.

        Returns:
            Generator: A generator that yields the processed molecules.
        """

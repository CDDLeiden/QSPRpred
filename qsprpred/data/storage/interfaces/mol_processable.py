from abc import ABC
from typing import Any, Generator

from qsprpred.data.processing.mol_processor import MolProcessor


class MolProcessable(ABC):

    def processMols(
            self,
            processor: MolProcessor,
            proc_args: tuple[Any, ...] | None = None,
            proc_kwargs: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        pass

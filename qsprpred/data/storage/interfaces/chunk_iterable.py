from abc import ABC, abstractmethod
from typing import Generator, Any, Iterable


class ChunkIterable(ABC):
    """Objects that can be iterated over and processed in chunks."""

    @abstractmethod
    def iterChunks(self, size: int | None = None) -> Generator[list[Any], None, None]:
        """Iterate over chunks of the storage.

        Args:
            size (int): The size of each chunk.

        Returns:
            A generator that yields chunks of the storage in any format.
        """

    @abstractmethod
    def apply(
            self,
            func: callable,
            func_args: list | None = None,
            func_kwargs: dict | None = None,
    ) -> Generator[Iterable[Any], None, None]:
        """Apply a function on chunks of data.
        The chunks are supplied as the first positional argument to the function.
        The format of the chunks is up to the downstream implementation, but it
        should always be a single object supplied as the first parameter.

        Args:
            func (callable): The function to apply.
            func_args (list, optional): The positional arguments of the function.
            func_kwargs (dict, optional): The keyword arguments of the function.
            *args: Additional positional arguments to the function.
            **kwargs: Additional keyword arguments to the function.
        """
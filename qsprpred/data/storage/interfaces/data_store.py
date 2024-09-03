from abc import ABC, abstractmethod

from qsprpred.utils.serialization import JSONSerializable


class DataStorage(JSONSerializable, ABC):
    """Abstract base class defining an API to interact with persistent data storage.

    Attributes:
        _notJSON (list):
            list of attributes that should not be serialized to JSON explicitly
    """

    @abstractmethod
    def save(self) -> str:
        """Save current state to storage and return the path to the serialized file.
        
        Returns:
            str: The path to the serialized file.
        """

    @abstractmethod
    def reload(self):
        """Reset the current state by reloading from storage."""

    @abstractmethod
    def clear(self):
        """Delete entries in the persistent storage."""

    @property
    @abstractmethod
    def metaFile(self) -> str:
        """Get the absolute path to the metadata file that describes how the
        persisted data can be accessed. This can be used to load the object
        back from storage using the `fromFile` class method.
        
        Returns:
            str: The absolute path to the metadata file.
        """

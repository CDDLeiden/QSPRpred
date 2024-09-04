from abc import ABC, abstractmethod

from qsprpred.utils.serialization import JSONSerializable


class DataStorage(JSONSerializable, ABC):
    """Abstract base class defining an API to interact with persistent data storage.
    This does not mean that the data is all stored locally, but only database or
    REST API connection details can be saved into this file as well. It assumes
    existence of `metaFile` attribute that points to a metadata file that describes
    this instance and it should be possible to initialize it from this file.
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

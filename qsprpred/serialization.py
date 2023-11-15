import copy
import os
from abc import ABC, abstractmethod

import jsonpickle
jsonpickle.set_encoder_options('json', indent=4)


class FileSerializable(ABC):
    """
    A class that can be serialized to a file and reconstructed from a file."""

    @abstractmethod
    def toFile(self, filename: str) -> str:
        """Serialize object to a metafile. This metafile should contain all
        data necessary to reconstruct the object.

        Args:
            filename (str): filename to save object to

        Returns:
            filename (str): absolute path to the saved metafile of the object
        """

    @classmethod
    @abstractmethod
    def fromFile(cls, filename: str) -> object:
        """Reconstruct object from a metafile.

        Args:
            filename (str): filename of the metafile to load object from

        Returns:
            obj (object): reconstructed object
        """


class JSONSerializable(FileSerializable):
    """A class that can be serialized to JSON and reconstructed from JSON.

    Attributes:
        _notJSON (list):
            list of attributes that should not be serialized to JSON explicitly
    """

    _notJSON = []

    def __getstate__(self) -> dict:
        """Get state of object for JSON serialization. Whatever
        is returned should be serializable to JSON.

        Returns:
            o_dict (dict): dictionary of object attributes serializable to JSON
        """
        o_dict = dict()
        for key in self.__dict__:
            if key in self._notJSON:
                continue
            o_dict[key] = copy.deepcopy(self.__dict__[key])
        return o_dict

    def __setstate__(self, state: dict):
        """Set state of object from a JSON serialization.

        Args:
            state (dict): dictionary of object attributes serializable to JSON
        """
        self.__dict__.update(state)

    def toFile(self, filename: str) -> str:
        """Serialize object to a JSON file. This JSON file should
        contain all  data necessary to reconstruct the object.

        Args:
            filename (str): filename to save object to

        Returns:
            filename (str): absolute path to the saved JSON file of the object
        """
        json = self.toJSON()
        with open(filename, 'w') as f:
            f.write(json)
        return os.path.abspath(filename)

    @classmethod
    def fromFile(cls, filename: str) -> object:
        """Initialize a new instance from a JSON file.

        Args:
            filename (str): path to the JSON file

        Returns:
            instance (object): new instance of the class
        """
        with open(filename, 'r') as f:
            json = f.read()
        return cls.fromJSON(json)

    def toJSON(self) -> str:
        """Serialize object to a JSON string. This JSON string should
         contain all data necessary to reconstruct the object.

        Returns:
            json (str): JSON string of the object
        """
        return jsonpickle.encode(self, unpicklable=True)

    @classmethod
    def fromJSON(cls, json: str) -> object:
        """Reconstruct object from a JSON string.

        Args:
            json (str): JSON string of the object

        Returns:
            obj (object): reconstructed object
        """
        return jsonpickle.decode(json)

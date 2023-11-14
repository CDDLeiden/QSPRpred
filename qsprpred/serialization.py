import copy
from abc import ABC, abstractmethod

import jsonpickle
jsonpickle.set_encoder_options('json', indent=4)


class FileSerializable(ABC):

    @abstractmethod
    def toFile(self, filename: str):
        """Serialize object to a metafile. This metafile should contain all
        data necessary to reconstruct the object.

        Args:
            filename (str): filename to save object to

        Returns:
            filename (str): absolute path to the saved metafile of the object
        """
        pass

    @classmethod
    @abstractmethod
    def fromFile(cls, filename: str):
        """Reconstruct object from a metafile.

        Args:
            filename (str): filename of the metafile to load object from

        Returns:
            obj (object): reconstructed object
        """
        pass


class JSONSerializable(FileSerializable):
    """A class that can be serialized to JSON and reconstructed from JSON.

    Attributes:
        _notJSON (list):
            list of attributes that should not be serialized to JSON explicitly
    """

    _notJSON = []

    def __getstate__(self):
        o_dict = dict()
        for key in self.__dict__:
            if key in self._notJSON:
                continue
            o_dict[key] = copy.deepcopy(self.__dict__[key])
        return o_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def toFile(self, filename: str):
        json = self.toJSON()
        with open(filename, 'w') as f:
            f.write(json)

    @classmethod
    def fromFile(cls, filename: str):
        with open(filename, 'r') as f:
            json = f.read()
        return cls.fromJSON(json)

    def toJSON(self):
        """Serialize object to a JSON string. This JSON string should contain
        all data necessary to reconstruct the object.

        Returns:
            json (str): JSON string of the object
        """
        return jsonpickle.encode(self, unpicklable=True)

    @classmethod
    def fromJSON(cls, json: str):
        """Reconstruct object from a JSON string.

        Args:
            json (str): JSON string of the object

        Returns:
            obj (object): reconstructed object
        """
        return jsonpickle.decode(json)

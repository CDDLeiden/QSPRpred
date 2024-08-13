import base64
import copy
import json
import marshal
import os
import types
from abc import ABC, abstractmethod
from typing import Any, Callable

import jsonpickle

from ..logs import logger
from ..utils.inspect import dynamic_import

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
            try:
                o_dict[key] = copy.deepcopy(self.__dict__[key])
            except Exception as exp:
                logger.error(f"Could not deepcopy '{key}' because of: {exp}")
                raise exp
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
    def fromFile(cls, filename: str) -> Any:
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
    def fromJSON(cls, json: str) -> Any:
        """Reconstruct object from a JSON string.

        Args:
            json (str): JSON string of the object

        Returns:
            obj (object): reconstructed object
        """
        return jsonpickle.decode(json)


def function_as_string(func: Callable) -> str:
    """Convert a function to a string.

    Args:
        func (Callable): function to convert
    """
    json_form = json.loads(jsonpickle.encode(func, unpicklable=True))
    if not json_form or "__main__" in json_form:
        return base64.b64encode(
            marshal.dumps(
                func.__code__
            )
        ).decode("ascii")
    elif "py/function" in json_form:
        return json_form["py/function"]
    elif "py/object" in json_form:
        return json_form["py/object"]


def function_from_string(func_str: str) -> Callable:
    """Convert a function from a string. Conversion from encoded bytecode is
    attempted first. If that fails, the string is assumed to be a fully
    qualified name of the function.

    Args:
        func_str (str): string representation of the function

    Returns:
        processor (Callable): function
    """
    try:
        return types.FunctionType(
            marshal.loads(
                base64.b64decode(func_str)
            ),
            globals()
        )
    except:
        return dynamic_import(func_str)

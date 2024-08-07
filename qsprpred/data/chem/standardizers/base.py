from abc import ABC, abstractmethod


class ChemStandardizer(ABC):

    def __call__(self, smiles):
        return self.convert_smiles(smiles)

    @abstractmethod
    def convert_smiles(self, smiles):
        """
        Convert the SMILES to a standardized form.

        :param smiles: SMILES to be converted
        :return: standardized SMILES
        """
        pass

    @property
    @abstractmethod
    def settings(self):
        pass

    @abstractmethod
    def get_id(self):
        pass

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: dict):
        pass

    @classmethod
    def from_settings_file(cls, path: str):
        """
        Load the standardizer from a settings file in JSON format.

        :param path:
        :return:
        """
        import json

        with open(path, "r") as f:
            settings = json.load(f)
        return cls.from_settings(settings)

    def get_hash_id(self):
        import hashlib

        return hashlib.md5(self.get_id()).hexdigest()

"""Module containing various extra descriptor calculators."""

import json

import numpy as np
import pandas as pd

from ....data.utils.descriptorcalculator import DescriptorsCalculator
from ....utils.inspect import import_class
from .descriptor_utils.msa_calculator import ClustalMSA, MSAProvider
from .descriptorsets import ProteinDescriptorSet


class ProteinDescriptorCalculator(DescriptorsCalculator):
    """Class for calculating protein descriptors.

    Arguments:
        desc_sets (list[ProteinDescriptorSet]): a list of protein descriptor sets to
            calculate protein descriptors.
        msa_provider(ClustalMSA): a provider of multiple sequence alignment (MSA)
            functionality. Defaults to ClustalMSA().
    """
    def __init__(
        self,
        desc_sets: list[ProteinDescriptorSet],
        msa_provider: MSAProvider = ClustalMSA()
    ) -> None:
        """Initialize the protein descriptor calculator.

        Args:
            desc_sets (list[ProteinDescriptorSet]): a list of protein descriptor sets to
                calculate protein descriptors.
            msa_provider (MSAProvider): a provide of multiple sequence alignment
                functionality. Defaults to `ClustalMSA`.
        """
        super().__init__(desc_sets)
        self.msaProvider = msa_provider

    def __call__(
        self,
        acc_keys: list[str],
        sequences: dict[str:str] | None = None,
        dtype: type = np.float32,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculates descriptors for the given protein accession keys.

        Args:
            acc_keys (list[str]):
                List of protein accession keys.
            sequences (dict[str:str] | None)
                Dictionary of protein sequences mapping accession keys to
                the protein sequence. This is only to be specified if
                one or more descriptor sets require a multiple sequence
                alignment or the sequences as they are.
            dtype (type):
                Data type of the returned dataframe.
            **kwargs (dict):
                Additional keyword arguments to be passed to the descriptor sets
                and the MSA provider.
        Returns:
            pd.DataFrame:
                Dataframe containing the calculated descriptors.
        """
        df = pd.DataFrame(index=acc_keys)
        for descset in self.descSets:
            # calculate the descriptor values
            if hasattr(descset, "setMSA"):
                msa = self.msaProvider(sequences, **kwargs)
                descset.setMSA(msa)
            values = descset(acc_keys, sequences, **kwargs)
            # compile the data into a dataframe
            if descset.isFP:
                values.add_prefix(f"{descset.fingerprint_type}_")
            values = values.astype(dtype)
            values = self.treatInfs(values)
            values = values.add_prefix(f"{self.getPrefix()}_{descset}_")
            df = df.merge(values, left_index=True, right_index=True)
        # return the dataframe
        return df

    def getPrefix(self) -> str:
        return "Descriptor_PCM"

    def toFile(self, fname: str):
        """Saves the descriptor calculator to file.

        The `msaProvider` is saved to a separate file with the extension
        `.msaprovider`. The calculated alignment is saved as `.msaprovider.msa`.

        Args:
            fname (str): File name to save to.
        """
        super().toFile(fname)
        # save msa if available
        self.msaProvider.toFile(f"{fname}.msaprovider")

    @classmethod
    def fromFile(cls, fname: str) -> "ProteinDescriptorCalculator":
        """Loads the descriptor calculator from file.

        Args:
            fname: Name of the file to load from.

        Returns:
            ProteinDescriptorCalculator: The loaded descriptor calculator.
        """

        ret = super().fromFile(fname)
        with open(f"{fname}.msaprovider", "r") as fh:  # file handle
            msa_provider_cls = json.load(fh)["class"]
        msa_provider_cls = import_class(msa_provider_cls)
        ret.msaProvider = msa_provider_cls.fromFile(f"{fname}.msaprovider")
        return ret

"""
descriptorcalculator

Created by: Martin Sicho
On: 12.05.23, 17:10
"""
import json
from typing import List

import numpy as np
import pandas as pd

from ....data.utils.descriptorcalculator import DescriptorsCalculator
from ....utils.inspect import import_class
from .descriptor_utils.msa_calculator import ClustalMSA
from .descriptorsets import ProteinDescriptorSet


class ProteinDescriptorCalculator(DescriptorsCalculator):
    def __init__(
        self, desc_sets: List[ProteinDescriptorSet], msa_provider=ClustalMSA()
    ) -> None:
        super().__init__(desc_sets)
        self.msaProvider = msa_provider

    def __call__(
        self,
        acc_keys,
        sequences: dict[str:str] = None,
        dtype=np.float32,
        **kwargs
    ) -> pd.DataFrame:
        df = pd.DataFrame(index=acc_keys)
        for descset in self.desc_sets:
            if hasattr(descset, "setMSA"):
                msa = self.msaProvider(sequences, **kwargs)
                descset.setMSA(msa)
            values = descset(acc_keys, sequences, **kwargs)

            if descset.is_fp:
                values.add_prefix(f"{descset.fingerprint_type}_")
            values = values.astype(dtype)
            values = self.treatInfs(values)
            values = values.add_prefix(f"{self.getPrefix()}_{descset}_")
            df = df.merge(values, left_index=True, right_index=True)

        return df

    def getPrefix(self) -> str:
        return "Descriptor_PCM"

    def toFile(self, fname: str) -> None:
        super().toFile(fname)

        # save msa if available
        self.msaProvider.toFile(f"{fname}.msaprovider")

    @classmethod
    def fromFile(cls, fname: str) -> None:
        ret = super().fromFile(fname)
        msa_provider_cls = json.load(open(f"{fname}.msaprovider", "r"))["class"]
        msa_provider_cls = import_class(msa_provider_cls)
        ret.msaProvider = msa_provider_cls.fromFile(f"{fname}.msaprovider")
        return ret

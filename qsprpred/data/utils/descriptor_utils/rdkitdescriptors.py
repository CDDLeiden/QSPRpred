import numpy as np
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from qsprpred.data.utils.descriptor_utils.interfaces import Scorer


class RdkitDescriptors(Scorer):
    def __init__(
        self, rdkit_descriptors: list[str] = None, compute_3Drdkit: bool = False
    ):
        """Initialize a RdkitDescriptors class to calculate rdkit descriptors.

        Args:
            rdkit_descriptors (optional): rdkit.Chem.Descriptors._descList.
                Defaults to None.
            compute_3Drdkit (bool, optional): _description_. Defaults to False.
        """
        self.available = dict(Descriptors._descList)
        if rdkit_descriptors is not None:
            self.descriptors = {
                k: v
                for k, v in self.available.items() if k in rdkit_descriptors
            }
        else:
            self.descriptors = (
                rdkit_descriptors if rdkit_descriptors is not None else
                [x[0] for x in Descriptors._descList]
            )
        if compute_3Drdkit:
            self.descriptors = [
                *self.descriptors, "Asphericity", "Eccentricity", "InertialShapeFactor",
                "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration",
                "SpherocityIndex"
            ]

    def getScores(self, mols):
        scores = np.zeros((len(mols), len(self.descriptors)))
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptors)
        for i, mol in enumerate(mols):
            try:
                scores[i] = calc.CalcDescriptors(mol)
            except AttributeError:
                continue
        return scores

    def getKey(self):
        # TODO: This should probably return a single str...
        return self.descriptors

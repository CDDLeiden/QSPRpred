import numpy as np
from qsprpred.data.utils.descriptor_utils.interfaces import Scorer
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


class RDKit_desc(Scorer):
    def __init__(self, rdkit_descriptors=None, compute_3Drdkit=False):
        self.descriptors = (
            rdkit_descriptors if rdkit_descriptors is not None else [x[0] for x in Descriptors._descList]
        )
        if compute_3Drdkit:
            self.descriptors = self.descriptors + [
                "Asphericity",
                "Eccentricity",
                "InertialShapeFactor",
                "NPR1",
                "NPR2",
                "PMI1",
                "PMI2",
                "PMI3",
                "RadiusOfGyration",
                "SpherocityIndex",
            ]

    def getScores(self, mols):
        scores = np.zeros((len(mols), len(self.descriptors)))
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptors)
        for i, mol in enumerate(mols):
            try:
                scores[i] = calc.CalcDescriptors(mol)
            except:
                continue
        return scores

    def getKey(self):
        return self.descriptors

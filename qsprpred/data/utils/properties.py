"""
properties

Created by: Martin Sicho
On: 06.06.22, 20:17
"""

import numpy as np

from rdkit.Chem.QED import qed
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem import Descriptors as desc, Crippen, AllChem, Lipinski

from qsprpred.data.interfaces import Scorer
from qsprpred.data.utils.sascorer import calculateScore

class Property(Scorer):

    def __init__(self, props=['MW'], modifier=None):
        super().__init__(modifier)
        self.props = props
        self.prop_dict = {'MW': desc.MolWt,
                          'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'Valence': desc.NumValenceElectrons,
                          'MR': Crippen.MolMR,
                          'QED': qed,
                        #   'SA': calculateScore,
                          'Bertz': BertzCT}

    def getScores(self, mols):
        scores = np.zeros((len(mols), len(self.props)))
        for i, mol in enumerate(mols):
            for prop in self.props:
                try:
                    scores[i] = self.prop_dict[prop](mol)
                except:
                    continue
        return scores

    def getKey(self):
        return self.props
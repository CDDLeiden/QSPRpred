"""
predictors

Created by: Martin Sicho
On: 06.06.22, 20:15
"""
import joblib
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem

from drugpk.training.interfaces import Scorer
from drugpk.training.scorers.properties import Property
from torch.utils.data import DataLoader, TensorDataset
import torch


class Predictor(Scorer):

    def __init__(self, model, type='CLS', name=None, modifier=None):
        super().__init__(modifier)
        self.type = type
        self.model = model
        self.key = f"{self.type}_{self.model.__class__.__name__}" if not name else name

    @staticmethod
    def fromFile(path, type='CLS', name="Predictor", modifier=None):
        return Predictor(joblib.load(path), type=type, name=name, modifier=modifier)

    def getScores(self, mols, frags=None):
        fps = self.calculateDescriptors(mols)
        if (self.model.__class__.__name__ == "STFullyConnected"):
            fps_loader = DataLoader(TensorDataset(torch.Tensor(fps)))
            scores = self.model.predict(fps_loader)
        elif (self.type == 'CLS'):
            scores = self.model.predict_proba(fps)[:, 1]
        else:
            scores = self.model.predict(fps)
        return scores

    @staticmethod
    def calculateDescriptors(mols, radius=3, bit_len=2048):
        ecfp = Predictor.calc_ecfp(mols, radius=radius, bit_len=bit_len)
        phch = Predictor.calc_physchem(mols)
        fps = np.concatenate([ecfp, phch], axis=1)
        return fps

    @staticmethod
    def calc_ecfp(mols, radius=3, bit_len=2048):
        fps = np.zeros((len(mols), bit_len))
        for i, mol in enumerate(mols):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, fps[i, :])
            except: pass
        return fps

    @staticmethod
    def calc_ecfp_rd(mols, radius=3):
        fps = []
        for i, mol in enumerate(mols):
            try:
                fp = AllChem.GetMorganFingerprint(mol, radius)
            except:
                fp = None
            fps.append(fp)
        return fps

    @staticmethod
    def calc_physchem(mols):
        prop_list = ['MW', 'logP', 'HBA', 'HBD', 'Rotable', 'Amide',
                     'Bridge', 'Hetero', 'Heavy', 'Spiro', 'FCSP3', 'Ring',
                     'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'Valence', 'MR']
        fps = np.zeros((len(mols), 19))
        props = Property()
        for i, prop in enumerate(prop_list):
            props.prop = prop
            fps[:, i] = props(mols)
        return fps

    def getKey(self):
        return self.key
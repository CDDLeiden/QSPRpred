"""
predictors

Created by: Martin Sicho
On: 06.06.22, 20:15
"""
import joblib
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem

from drugpk.training.interfaces import Scorer
from torch.utils.data import DataLoader, TensorDataset
import torch

class Predictor(Scorer):

    def __init__(self, model, features, type='CLS', name=None, modifier=None):
        super().__init__(modifier)
        self.type = type
        self.model = model
        self.features = features
        self.key = f"{self.type}_{self.model.__class__.__name__}" if not name else name

    @staticmethod
    def fromFile(path, type='CLS', name="Predictor", modifier=None):
        return Predictor(joblib.load(path), type=type, name=name, modifier=modifier)

    def getScores(self, mols):
        fps = np.array(self.features(mols))
        if (self.model.__class__.__name__ == "STFullyConnected"):
            fps_loader = DataLoader(TensorDataset(torch.Tensor(fps)))
            scores = self.model.predict(fps_loader)
        elif (self.type == 'CLS'):
            scores = self.model.predict_proba(fps)[:, 1]
        else:
            scores = self.model.predict(fps)
        return scores

    def getKey(self):
        return self.key

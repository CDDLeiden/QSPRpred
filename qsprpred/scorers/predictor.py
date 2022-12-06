"""
predictors

Created by: Martin Sicho
On: 06.06.22, 20:15
"""
from typing import List


import json
import sklearn_json as skljson
import numpy as np

from qsprpred.data.utils.descriptor_utils.interfaces import Scorer
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.models.neural_network import STFullyConnected

import torch

from qsprpred.data.utils.feature_standardization import SKLearnStandardizer


class Predictor(Scorer):
 
    def __init__(self, model, feature_calculators, scaler : SKLearnStandardizer, type='CLS', th=1, name=None, modifier=None):
        """Construct predictor model, feature calculator & scaler.

        Args:
            model: fitted sklearn or toch model
            feature_calculators: DescriptorsCalculator object, calculates features from smiles
            scaler: StandardStandardizer, scales features
            type: regression or classification
            th: if classification give activity threshold
            name: name for predictor
            modifier: score modifier
        """
        super().__init__(modifier)
        self.model = model
        self.feature_calculators = feature_calculators
        self.scaler = scaler 
        self.type = type
        self.th = th
        self.key = f"{self.model.__class__.__name__}" if not name else name

    @staticmethod
    def fromFile(base_dir, algorithm, target, type='CLS', th=1, scale = True, name="Predictor", modifier=None):
        """Construct predictor from files with serialized model, feature calculator & scaler.

        Args:
            base_dir: base directory with folder qsprmodels/ containing the serialized mode, feature_descriptor and optionally scaler
            algorithm: type of model
            target: name of property to predict
            type: regression or classification
            scale: bool if true, apply feature scaling
            th: if classification give activity threshold
            name: name for predictor
            modifier: score modifier

        Returns:
            predictor
            
        """
        path = base_dir + '/qsprmodels/' + '_'.join([algorithm, type, target]) + '.json'
        feature_calculators = DescriptorsCalculator.fromFile(base_dir + '/qsprmodels/' + '_'.join([type, target]) + '_DescCalc.json')
        #TODO do not hardcode when to use scaler
        scaler = None
        if scale:
            scaler = SKLearnStandardizer.fromFile(base_dir + '/qsprmodels/' + '_'.join([type, target]) + '_scaler.json')
              
        if "DNN" in path:
            with open(path) as f:
                model_params = json.load(f)
            model = STFullyConnected(**model_params)
            model.load_state_dict(torch.load(f"{path[:-5]}_weights.pkg"))
            return Predictor(model, feature_calculators=feature_calculators, scaler=scaler, type=type, th=th, name=name, modifier=modifier)
        return Predictor(skljson.from_json(path), feature_calculators=feature_calculators, scaler=scaler, type=type, th=th, name=name, modifier=modifier)

    def getScores(self, mols, frags=None):
        """
        Returns scores for the input molecules.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (numpy.ndarray): 'np.array' of scores for "mols"
        """

        features = self.feature_calculators(mols)
        if self.scaler:
            features = self.scaler(features)
        if (self.model.__class__.__name__ == "STFullyConnected"):
            fps_loader = self.model.get_dataloader(features)
            if len(self.th) > 1:
                scores = np.argmax(self.model.predict(fps_loader), axis=1).astype(float)
            elif len(self.th) == 1:
                scores = self.model.predict(fps_loader)[:,1].astype(float)
            else:
                scores = self.model.predict(fps_loader).flatten()
        elif (self.model.__class__.__name__ == 'PLSRegression'):
            scores = self.model.predict(features)[:, 0]
        elif (self.type == 'REG'):
            scores = self.model.predict(features)
        elif len(self.th) > 1:
            scores = np.argmax(self.model.predict_proba(features), axis=1).astype(float)
        elif (self.type == 'CLS'):
            scores = self.model.predict_proba(features)[:, 1]
        
        if len(scores.shape) > 1 and scores.shape[1] == 1:
            scores = scores[:,0]
        return scores

    def getKey(self):
        return self.key


"""
predictors

Created by: Martin Sicho
On: 06.06.22, 20:15
"""
import json
import os

import numpy as np
import sklearn_json as skljson
import torch
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.descriptor_utils.interfaces import Scorer
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.feature_standardization import (
    SKLearnStandardizer,
    apply_feature_standardizers,
)
from qsprpred.logs import logger
from qsprpred.models.neural_network import STFullyConnected


class Predictor(Scorer):

    def __init__(
            self, model, feature_calculators, standardizers: SKLearnStandardizer,
            type: str, th=(1,), name=None, modifier=None):
        """Construct predictor model, feature calculator & standardizer.

        Args:
            model: fitted sklearn or toch model
            feature_calculators: DescriptorsCalculator object, calculates features from smiles
            standardizers: StandardStandardizer(s), scales features
            type: regression or classification
            th: if classification give activity threshold
            name: name for predictor
            modifier: score modifier
        """
        super().__init__(modifier)
        self.model = model
        self.feature_calculators = feature_calculators
        self.standardizers = standardizers
        self.type = type
        self.th = th
        self.key = f"{self.model.__class__.__name__}" if not name else name

    @staticmethod
    def fromFile(model_path: str, metadata_path: str, scale: bool = True, modifier=None):
        """Construct predictor from files with serialized model, feature calculator & standardizer.

        Args:
            model_path: Path to saved json model file, if DNN assumes weight file also in the same folder
            metadata_path: Path to QSPRdata metadata file
            modifier: score modifier

        Returns:
            predictor

        """
        with open(metadata_path) as f:
            meta = json.load(f)

        feature_calculators = DescriptorsCalculator.fromFile(meta["descriptorcalculator_path"])
        standardizers = []
        if meta['standardizer_paths'] is not None and scale:
            for standardizer_path in meta['standardizer_paths']:
                standardizers.append(SKLearnStandardizer.fromFile(standardizer_path))

        th = meta['th'] if 'th' in meta.keys() else None
        type = 'REG' if meta['init']['task'] == "REGRESSION" else 'CLS'

        # load model
        name = os.path.basename(model_path)[:-5]

        if "DNN" in model_path:
            with open(model_path) as f:
                model_params = json.load(f)
            model = STFullyConnected(**model_params)
            model.load_state_dict(torch.load(f"{model_path[:-5]}_weights.pkg"))
            return Predictor(
                model, feature_calculators=feature_calculators, standardizers=standardizers,
                type=type, th=th, name=name, modifier=modifier)
        return Predictor(
            skljson.from_json(model_path),
            feature_calculators=feature_calculators, standardizers=standardizers, type=type,
            th=th, name=name, modifier=modifier)

    def getScores(self, mols, frags=None):
        """Return scores for the input molecules.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (numpy.ndarray): 'np.array' of scores for "mols"
        """
        # Calculate and scale the features
        features = self.feature_calculators(mols)
        if self.standardizers is not None:
            features, _ = apply_feature_standardizers(self.standardizers, features, fit=False)

        # Special case DNN
        if (self.model.__class__.__name__ == "STFullyConnected"):
            fps_loader = self.model.get_dataloader(features)
            if self.type == 'CLS':
                if len(self.th) > 1:
                    scores = np.argmax(
                        self.model.predict(fps_loader), axis=1).astype(float)
                elif len(self.th) == 1:
                    scores = self.model.predict(fps_loader)[:, 1].astype(float)
            else:
                scores = self.model.predict(fps_loader).flatten()
        # Special case PLS
        elif self.model.__class__.__name__ == 'PLSRegression':
            scores = self.model.predict(features)[:, 0]
        # Regression
        elif self.type == 'REG':
            scores = self.model.predict(features)
        # Multi-class classification
        elif len(self.th) > 1:
            scores = np.argmax(
                self.model.predict_proba(features),
                axis=1).astype(float)
        # Single-class classification
        elif (self.type == 'CLS'):
            scores = self.model.predict_proba(features)[:, 1]

        if len(scores.shape) > 1 and scores.shape[1] == 1:
            scores = scores[:, 0]
        return scores

    def getKey(self):
        """Return model identifier."""
        return self.key

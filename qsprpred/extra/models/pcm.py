"""
pcm

Created by: Martin Sicho
On: 12.05.23, 23:15
"""
from abc import ABC
from typing import List, Union

import numpy as np

from qsprpred.data.data import MoleculeTable
from qsprpred.extra.data.data import PCMDataset
from qsprpred.extra.data.utils.descriptorcalculator import ProteinDescriptorCalculator
from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.models import QSPRsklearn


class ModelPCM(QSPRModel, ABC):

    def createPredictionDatasetFromMols(self, mols: List[str], protein_id : str, smiles_standardizer: Union[str, callable] = 'chembl', n_jobs: int = 1, fill_value: float = np.nan):
        dataset = MoleculeTable.fromSMILES(f"{self.__class__.__name__}_{hash(self)}", mols, drop_invalids=False,
                                           n_jobs=n_jobs)
        for targetproperty in self.targetProperties:
            dataset.addProperty(targetproperty.name, np.nan)
        dataset.addProperty("protein_id", protein_id)

        dataset = PCMDataset.fromMolTable(
            dataset,
            "protein_id",
            target_props=self.targetProperties,
            drop_empty=False,
            drop_invalids=False,
            n_jobs=n_jobs
        )

        dataset.standardizeSmiles(smiles_standardizer, drop_invalid=False)
        failed_mask = dataset.dropInvalids().values

        dataset.prepareDataset(
            smiles_standardizer=smiles_standardizer,
            feature_calculators=self.featureCalculators,
            feature_standardizer=self.featureStandardizer,
            feature_fill_value=fill_value
        )
        return dataset, failed_mask

    def predictMols(self, mols: List[str], protein_id : str, use_probas: bool = False,
                    smiles_standardizer: Union[str, callable] = 'chembl',
                    n_jobs: int = 1, fill_value: float = np.nan):
        # check if the model contains a feature calculator
        if not self.featureCalculators:
            raise ValueError("No feature calculator set on this instance.")

        # run PCM checks to validate the protein ids and descriptors
        is_pcm = False
        protein_ids = set()
        for calc in self.featureCalculators:
            if isinstance(calc, ProteinDescriptorCalculator):
                is_pcm = True
                if not protein_ids:
                    protein_ids = set(calc.msaProvider.current.keys())
                else:
                    assert protein_ids == set(calc.msaProvider.current.keys()), "All protein descriptor calculators must have the same protein ids."
            if isinstance(calc, ProteinDescriptorCalculator) and calc.msaProvider and protein_id not in calc.msaProvider.current.keys():
                raise ValueError(f"Protein id {protein_id} not found in the available MSA, cannot calculate PCM descriptors. Options are: {protein_ids}.")
        if not is_pcm:
            raise ValueError("No protein descriptors found on this instance. Are you sure this is a PCM model?")

        # create data set from mols
        dataset, failed_mask = self.createPredictionDatasetFromMols(mols, protein_id, smiles_standardizer, n_jobs, fill_value)

        # make predictions for the dataset
        predictions = self.predictDataset(dataset, use_probas)

        # handle invalids
        predictions = self.handleInvalidsInPredictions(mols, predictions, failed_mask)

        return predictions


class QSPRsklearnPCM(QSPRsklearn, ModelPCM):
    pass

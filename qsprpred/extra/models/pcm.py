"""Specialized models for proteochemometric models (PCM).

"""

from abc import ABC
from typing import Callable

import numpy as np

from ...data.data import MoleculeTable
from ...models.interfaces import QSPRModel
from ...models.sklearn import SklearnModel
from ..data.data import PCMDataSet
from ..data.utils.descriptorcalculator import ProteinDescriptorCalculator


class PCMModel(QSPRModel, ABC):
    """Base class for PCM models.

    Extension of `QSPRModel` for proteochemometric models (PCM). It modifies
    the `predictMols` method to handle PCM descriptors and specification of protein ids.
    """
    def createPredictionDatasetFromMols(
        self,
        mols: list[str],
        protein_id: str,  # FIXME: this changes the signature from the base class
        smiles_standardizer: str | Callable = "chembl",
        n_jobs: int = 1,
        fill_value: float = np.nan,
    ) -> tuple[PCMDataSet, np.ndarray]:
        """
        Create a prediction data set of compounds using a PCM model
        given as a list of SMILES strings and a protein identifier.
        The protein identifier is used to calculate the protein descriptors.

        Args:
            mols (list[str]):
                List of SMILES strings.
            protein_id (str):
                Protein identifier.
            smiles_standardizer (str | Callable, optional):
                Smiles standardizer. Defaults to "chembl".
            n_jobs (int, optional):
                Number of parallel jobs. Defaults to 1.
            fill_value (float, optional):
                Value to fill missing features with. Defaults to np.nan.
        Returns:
            PCMDataSet:
                Dataset with the features calculated for the molecules.
        """
        dataset = MoleculeTable.fromSMILES(
            f"{self.__class__.__name__}_{hash(self)}",
            mols,
            drop_invalids=False,
            n_jobs=n_jobs,
        )
        for targetproperty in self.targetProperties:
            dataset.addProperty(targetproperty.name, np.nan)
        dataset.addProperty("protein_id", protein_id)
        # convert to PCMDataSet
        dataset = PCMDataSet.fromMolTable(
            dataset,
            "protein_id",
            target_props=self.targetProperties,
            drop_empty=False,
            drop_invalids=False,
            n_jobs=n_jobs,
        )
        # standardize smiles
        dataset.standardizeSmiles(smiles_standardizer, drop_invalid=False)
        failed_mask = dataset.dropInvalids().values
        # calculate features and prepare dataset
        dataset.prepareDataset(
            smiles_standardizer=smiles_standardizer,
            feature_calculators=self.featureCalculators,
            feature_standardizer=self.featureStandardizer,
            feature_fill_value=fill_value,
            shuffle=False,
        )
        return dataset, failed_mask

    def predictMols(
        self,
        mols: list[str],
        protein_id: str,  # FIXME: this changes the signature from the base class
        use_probas: bool = False,
        smiles_standardizer: str | Callable = "chembl",
        n_jobs: int = 1,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        Predict the target properties of a list of molecules using a PCM model.
        The protein identifier is used to calculate the protein descriptors for
        a target of interest.

        Args:
            mols (list[str]):
                List of SMILES strings.
            protein_id (str):
                Protein identifier.
            use_probas (bool, optional):
                Whether to return class probabilities. Defaults to False.
            smiles_standardizer (str | Callable, optional):
                Smiles standardizer. Defaults to "chembl".
            n_jobs (int, optional):
                Number of parallel jobs. Defaults to 1.
            fill_value (float, optional):
                Value to fill missing features with. Defaults to np.nan.

        Returns:
            np.ndarray:
                Array of predictions.

        """
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
                    assert protein_ids == set(calc.msaProvider.current.keys()), (
                        "All protein descriptor calculators "
                        "must have the same protein ids."
                    )
            if (
                isinstance(calc, ProteinDescriptorCalculator) and calc.msaProvider and
                protein_id not in calc.msaProvider.current.keys()
            ):
                raise ValueError(
                    f"Protein id {protein_id} not found in the available MSA, "
                    f"cannot calculate PCM descriptors. Options are: {protein_ids}."
                )
        if not is_pcm:
            raise ValueError(
                "No protein descriptors found on this instance. "
                "Are you sure this is a PCM model?"
            )
        # create data set from mols
        dataset, failed_mask = self.createPredictionDatasetFromMols(
            mols, protein_id, smiles_standardizer, n_jobs, fill_value
        )
        # make predictions for the dataset
        predictions = self.predictDataset(dataset, use_probas)
        # handle invalids
        predictions = self.handleInvalidsInPredictions(mols, predictions, failed_mask)
        return predictions


class SklearnPCMModel(SklearnModel, PCMModel):
    """Wrapper for sklearn models for PCM.

    Just replaces some methods in `SklearnModel` with those in `PCMModel`.
    """

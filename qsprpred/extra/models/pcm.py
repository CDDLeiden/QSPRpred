"""Specialized models for proteochemometric models (PCM).

"""

from abc import ABC

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data import MoleculeTable
from qsprpred.extra.data.tables.pcm import PCMDataSet
from ..data.descriptors.sets import ProteinDescriptorSet
from ...data.storage.tabular.basic_storage import PandasChemStore
from ...models.model import QSPRModel
from ...models.scikit_learn import SklearnModel


class PCMModel(QSPRModel, ABC):
    """Base class for PCM models.

    Extension of `QSPRModel` for proteochemometric models (PCM). It modifies
    the `predictMols` method to handle PCM descriptors and specification of protein ids.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "proteins"):
            self.proteins = None

    def initFromDataset(self, data: PCMDataSet | None):
        super().initFromDataset(data)
        if data:
            self.proteins = data.proteins

    def createPredictionDatasetFromMols(
            self,
            mols: list[str | Mol],
            protein_id: str,
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
        # make a molecule table first and add the target properties
        if isinstance(mols[0], Mol):
            mols = [Chem.MolToSmiles(mol) for mol in mols]
        storage = PandasChemStore(
            f"{self.__class__.__name__}_{hash(self)}_store",
            self.baseDir,
            pd.DataFrame({"SMILES": mols}),
            standardizer=self.chemStandardizer,
            n_jobs=n_jobs,
        )
        failed_mask = np.full(len(mols), False)
        if len(storage) != len(mols):
            original_smiles = storage.getProperty("SMILES_original")
            for i, mol in enumerate(mols):
                if mol not in original_smiles:
                    failed_mask[i] = True
        dataset = MoleculeTable(
            storage,
            f"{self.__class__.__name__}_{hash(self)}",
            path=self.baseDir,
        )
        dataset.addProperty(self.proteins.idProp, protein_id)
        for target_property in self.targetProperties:
            target_property.imputer = None
            dataset.addProperty(target_property.name, np.nan)
        # create the dataset and get failed molecules
        dataset = PCMDataSet.fromMolTable(
            dataset,
            self.targetProperties,
            drop_empty_target_props=False,
            proteins=self.proteins,
        )
        # prepare dataset and return it
        dataset.prepareDataset(
            feature_calculators=self.featureCalculators,
            feature_standardizer=self.featureStandardizer,
            feature_fill_value=fill_value,
            shuffle=False,
        )
        return dataset, failed_mask

    def predictMols(
            self,
            mols: list[str],
            protein_id: str,
            use_probas: bool = False,
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
            if isinstance(calc, ProteinDescriptorSet):
                is_pcm = True
                if not protein_ids and hasattr(calc, "msaProvider"):
                    protein_ids = set(calc.msaProvider.current.keys())
                if protein_ids and hasattr(calc, "msaProvider"):
                    assert protein_ids == set(calc.msaProvider.current.keys()), (
                        "All protein descriptor calculators "
                        "must have the same protein ids."
                    )
            if (
                    isinstance(calc, ProteinDescriptorSet)
                    and hasattr(calc, "msaProvider")
                    and calc.msaProvider
                    and protein_id not in calc.msaProvider.current.keys()
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
            mols, protein_id, n_jobs, fill_value
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

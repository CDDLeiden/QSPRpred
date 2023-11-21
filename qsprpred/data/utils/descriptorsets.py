"""Descriptorset: a collection of descriptors that can be calculated for a molecule.

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Mol, Descriptors
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors

from qsprpred.utils.serialization import JSONSerializable
from ...logs import logger

from ...models.models import QSPRModel
from .descriptor_utils.fingerprints import get_fingerprint


class DescriptorSet(JSONSerializable, ABC):
    __len__ = lambda self: self.getLen()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Calculate the descriptors for a given input.

        Args:
            *args: arguments to be passed to perform the calculation
            **kwargs: keyword arguments to be passed to perform the calculation

        Returns:
            DataFrame or np.array of descriptors with shape [n_inputs, n_descriptors]
        """

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["descriptors"] = self.descriptors
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.descriptors = state["descriptors"]

    @property
    @abstractmethod
    def descriptors(self):
        """Return a list of descriptor names."""

    @descriptors.setter
    @abstractmethod
    def descriptors(self, value):
        """Set the descriptor names."""

    def getLen(self):
        """Return the number of descriptors."""
        return len(self.descriptors)

    @property
    @abstractmethod
    def isFP(self):
        """Return True if descriptorset is fingerprint."""

    @property
    @abstractmethod
    def settings(self):
        """Return dictionary with arguments used to initialize the descriptorset."""

    @abstractmethod
    def __str__(self):
        """Return string representation of the descriptorset."""


class MoleculeDescriptorSet(DescriptorSet):
    """Abstract base class for descriptorsets.

    Descriptorset: a collection of descriptors that can be calculated for a molecule.
    """
    @abstractmethod
    def __call__(self, mols: list[str | Mol]) -> np.ndarray | pd.DataFrame:
        """
        Calculate the descriptor for a molecule.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """

    @staticmethod
    def iterMols(mols: list[str | Mol], to_list=False):
        """
        Create a molecule iterator or list from RDKit molecules or SMILES.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)
            to_list: if True, return a list instead of an iterator

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """
        ret = (Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols)
        if to_list:
            ret = list(ret)
        return ret


class DataFrameDescriptorSet(DescriptorSet):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._descriptors = df.columns.tolist() if df is not None else []

    def getDF(self):
        return self._df

    def getIndex(self):
        return self._df.index if self._df is not None else None

    def __call__(self, index, *args, **kwargs):
        if self._df is None:
            raise ValueError("No dataframe set.")
        ret = pd.DataFrame(index=index)
        ret = ret.merge(self._df, how="left", left_index=True, right_index=True)
        return ret[self.descriptors]

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    @property
    def isFP(self):
        return False

    @property
    def settings(self):
        return {"df": None}

    def __str__(self):
        return "DataFrame"


class FingerprintSet(MoleculeDescriptorSet):
    """Generic fingerprint descriptorset can be used to calculate any fingerprint type
    defined in descriptorutils.fingerprints.
    """

    _notJSON = [*MoleculeDescriptorSet._notJSON, "getFingerprint"]

    def __init__(self, fingerprint_type, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to your
        fingerprint type of choice.

        Args:
            fingerprint_type: fingerprint type
            *args: fingerprint specific arguments
            **kwargs: fingerprint specific arguments keyword arguments
        """
        self._isFP = True
        self.fingerprintType = fingerprint_type
        self._args = args
        self._kwargs = kwargs
        self.getFingerprint = get_fingerprint(
            self.fingerprintType, *self._args, **self._kwargs
        )
        self._keepindices = None

    def __setstate__(self, state):
        super().__setstate__(state)
        self.getFingerprint = get_fingerprint(
            self.fingerprintType, *self._args, **self._kwargs
        )

    def __call__(self, mols):
        """Calculate the fingerprint for a list of molecules."""
        mols = [Chem.AddHs(mol) for mol in self.iterMols(mols)]
        ret = self.getFingerprint(mols)

        if self.keepindices:
            ret = ret[:, self.keepindices]

        return ret

    @property
    def keepindices(self):
        """Return the indices of the fingerprint to keep."""
        return self._keepindices

    @keepindices.setter
    def keepindices(self, val):
        """Set the indices of the fingerprint to keep."""
        self._keepindices = [int(x) for x in val] if val else None

    @property
    def isFP(self):
        """Return True if descriptorset is fingerprint."""
        return self._isFP

    @property
    def settings(self):
        """Return dictionary with arguments used to initialize the descriptorset."""
        return {
            "fingerprint_type": self.fingerprintType,
            **self.getFingerprint.settings,
        }

    def getLen(self):
        """Return the length of the fingerprint."""
        return len(self.getFingerprint)

    def __str__(self):
        return f"FingerprintSet_{self.fingerprintType}"

    @property
    def descriptors(self):
        """Return the indices of the fingerprint that are kept."""
        indices = self.keepindices if self.keepindices else range(self.getLen())
        return [f"{idx}" for idx in indices]

    @descriptors.setter
    def descriptors(self, value):
        """Set the indices of the fingerprint to keep."""
        self.keepindices = value


class DrugExPhyschem(MoleculeDescriptorSet):
    """Various properties used for scoring in DrugEx.

    Args:
        props: list of properties to calculate
    """
    def __init__(self, physchem_props: list[str] | None = None):
        """Initialize the descriptorset with Property arguments (a list of properties to
        calculate) to select a subset.

        Args:
            physchem_props: list of properties to calculate
        """
        self._prop_dict = {
            "MW": desc.MolWt,
            "logP": Crippen.MolLogP,
            "HBA": AllChem.CalcNumLipinskiHBA,
            "HBD": AllChem.CalcNumLipinskiHBD,
            "Rotable": AllChem.CalcNumRotatableBonds,
            "Amide": AllChem.CalcNumAmideBonds,
            "Bridge": AllChem.CalcNumBridgeheadAtoms,
            "Hetero": AllChem.CalcNumHeteroatoms,
            "Heavy": Lipinski.HeavyAtomCount,
            "Spiro": AllChem.CalcNumSpiroAtoms,
            "FCSP3": AllChem.CalcFractionCSP3,
            "Ring": Lipinski.RingCount,
            "Aliphatic": AllChem.CalcNumAliphaticRings,
            "Aromatic": AllChem.CalcNumAromaticRings,
            "Saturated": AllChem.CalcNumSaturatedRings,
            "HeteroR": AllChem.CalcNumHeterocycles,
            "TPSA": AllChem.CalcTPSA,
            "Valence": desc.NumValenceElectrons,
            "MR": Crippen.MolMR,
        }
        if physchem_props:
            self.props = physchem_props
        else:
            self.props = self._prop_dict.keys()
        self._isFP = False

    def __call__(self, mols):
        """Calculate the DrugEx properties for a molecule."""
        mols = self.iterMols(mols, to_list=True)
        scores = np.zeros((len(mols), len(self.props)))
        for i, mol in enumerate(mols):
            for j, prop in enumerate(self.props):
                try:
                    scores[i, j] = self._prop_dict[prop](mol)
                except Exception as exp:  # noqa: E722
                    logger.warning(f"Could not calculate {prop} for {mol}.")
                    logger.exception(exp)
                    continue
        return scores

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {"physchem_props": self.props}

    @property
    def descriptors(self):
        return self.props

    @descriptors.setter
    def descriptors(self, props):
        """Set new props as a list of names."""
        self.props = props

    def __str__(self):
        return "DrugExPhyschem"


class RDKitDescs(MoleculeDescriptorSet):
    """
    Calculate RDkit descriptors.

    Args:
        rdkit_descriptors: list of descriptors to calculate, if none, all 2D rdkit
            descriptors will be calculated
        include_3d: if True, 3D descriptors will be calculated
    """
    def __init__(
            self,
            rdkit_descriptors: list[str] | None = None,
            include_3d: bool = False
    ):
        self._isFP = False
        self._descriptors = (
            rdkit_descriptors
            if rdkit_descriptors is not None
            else sorted(list(set([x[0] for x in Descriptors._descList])))
        )
        if include_3d:
            self.descriptors = [
                *self.descriptors,
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
        self.include3D = include_3d

    def __call__(self, mols):
        mols = self.iterMols(mols, to_list=True)
        scores = np.zeros((len(mols), len(self.descriptors)))
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptors)
        for i, mol in enumerate(mols):
            try:
                scores[i] = calc.CalcDescriptors(mol)
            except AttributeError:
                continue
        return scores

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {
            "rdkit_descriptors": self.descriptors,
            "include_3d": self.include3D,
        }

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        self._descriptors = descriptors

    def __str__(self):
        return "RDkit"


class TanimotoDistances(MoleculeDescriptorSet):
    """
    Calculate Tanimoto distances to a list of SMILES sequences.

    Args:
        list_of_smiles (list of strings): list of SMILES to calculate the distances.
        fingerprint_type (str): fingerprint type to use.
        *args: `fingerprint` arguments
        **kwargs: `fingerprint` keyword arguments, should contain fingerprint_type
    """
    def __init__(self, list_of_smiles, fingerprint_type, *args, **kwargs):
        """Initialize the descriptorset with a list of SMILES sequences and a
        fingerprint type.

        Args:
            list_of_smiles (list of strings): list of SMILES sequences to calculate
                distance to
            fingerprint_type (str): fingerprint type to use
        """
        self._descriptors = list_of_smiles
        self.fingerprintType = fingerprint_type
        self._args = args
        self._kwargs = kwargs
        self._isFP = False

        # intialize fingerprint calculator
        self.getFingerprint = get_fingerprint(
            self.fingerprintType, *self._args, **self._kwargs
        )
        self.fps = self.calculate_fingerprints(list_of_smiles)

    def __call__(self, mols):
        """Calculate the Tanimoto distances to the list of SMILES sequences.

        Args:
            mols (List[str] or List[rdkit.Chem.rdchem.Mol]): SMILES sequences or RDKit
                molecules to calculate the distances.
        """
        mols = [
            Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols
        ]
        # Convert np.arrays to BitVects
        fps = [
            DataStructs.CreateFromBitString("".join(map(str, x)))
            for x in self.getFingerprint(mols)
        ]
        return [
            list(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, self.fps)))
            for fp in fps
        ]

    def calculate_fingerprints(self, list_of_smiles):
        """Calculate the fingerprints for the list of SMILES sequences."""
        # Convert np.arrays to BitVects
        return [
            DataStructs.CreateFromBitString("".join(map(str, x))) for x in self.
            getFingerprint([Chem.MolFromSmiles(smiles) for smiles in list_of_smiles])
        ]

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {
            "fingerprint_type": self.fingerprintType,
            "list_of_smiles": self._descriptors,
            "args": self._args,
            "kwargs": self._kwargs,
        }

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, list_of_smiles):
        """Set new list of SMILES sequences to calculate distance to."""
        self._descriptors = list_of_smiles
        self.list_of_smiles = list_of_smiles
        self.fps = self.calculate_fingerprints(self.list_of_smiles)

    def __str__(self):
        return "TanimotoDistances"


class PredictorDesc(MoleculeDescriptorSet):
    """MoleculeDescriptorSet that uses a Predictor object to calculate descriptors from
    a molecule."""

    _notJSON = [*MoleculeDescriptorSet._notJSON, "model"]

    def __init__(self, model: QSPRModel | str):
        """
        Initialize the descriptorset with a `QSPRModel` object.

        Args:
            model: a fitted model instance or a path to the model's meta file
        """

        if isinstance(model, str):
            from ...models.models import QSPRModel

            self.model = QSPRModel.fromFile(model)
        else:
            self.model = model

        self._descriptors = [self.model.name]

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["model"] = self.model.metaFile
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.model = QSPRModel.fromFile(self.model)

    def __call__(self, mols):
        """
        Calculate the descriptor for a list of molecules.

        Args:
            mols (list): list of smiles or rdkit molecules

        Returns:
            an array of descriptor values
        """
        mols = list(mols)
        if type(mols[0]) != str:
            mols = [Chem.MolToSmiles(mol) for mol in mols]
        return self.model.predictMols(mols, use_probas=False)

    @property
    def isFP(self):
        return False

    @property
    def settings(self):
        """Return args and kwargs used to initialize the descriptorset."""
        return {
            "model":
                self.model.
                metaFile  # FIXME: we save absolute path to meta file so this descriptor
            # set is not really portable
        }

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        self._descriptors = descriptors

    def getLen(self):
        return 1

    def __str__(self):
        return "PredictorDesc"


class SmilesDesc(MoleculeDescriptorSet):
    """Descriptorset that calculates descriptors from a SMILES sequence."""
    def __call__(self, mols: list[str | Mol]):
        """Return smiles as descriptors.

        Args:
            mols (list): list of smiles or rdkit molecules

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """
        if all(isinstance(mol, str) for mol in mols):
            return np.array(mols)
        elif all(isinstance(mol, Mol) for mol in mols):
            return np.array([Chem.MolToSmiles(mol) for mol in mols])
        else:
            raise ValueError("Molecules should be either SMILES or RDKit Mol objects.")

    @property
    def isFP(self):
        return False

    @property
    def settings(self):
        return {}

    @property
    def descriptors(self):
        return ["SMILES"]

    @descriptors.setter
    def descriptors(self, descriptors):
        pass

    def __str__(self):
        return "SmilesDesc"


class _DescriptorSetRetriever:
    """Based on recipe 8.21 of the book "Python Cookbook".

    To support a new type of descriptor, just add a function "get_descname(self, *args,
    **kwargs)".
    """
    def getDescriptor(self, desc_type, *args, **kwargs):
        if desc_type.lower() == "descriptor":
            raise Exception("Please specify the type of fingerprint you want to use.")
        method_name = "get" + desc_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{desc_type} is not a supported descriptor set type.")
        return method(*args, **kwargs)

    def getFingerprintSet(self, *args, **kwargs):
        """Wrapper to get a fingerprint set."""
        return FingerprintSet(*args, **kwargs)

    def getDrugExPhyschem(self, *args, **kwargs):
        """Wrapper to get DrugEx physichochemical properties."""
        return DrugExPhyschem(*args, **kwargs)

    def getMordred(self, *args, **kwargs):
        """Wrapper to get Mordred descriptors - depends on optional dependency."""
        from ...extra.data.utils.descriptorsets import Mordred

        return Mordred(*args, **kwargs)

    def getMold2(self, *args, **kwargs):
        """Wrapper to get Mold2 descriptors - depends on optional dependency."""
        from ...extra.data.utils.descriptorsets import Mold2

        return Mold2(*args, **kwargs)

    def getPaDEL(self, *args, **kwargs):
        """Wrapper to get PaDEL descriptors - depends on optional dependency."""
        from ...extra.data.utils.descriptorsets import PaDEL

        return PaDEL(*args, **kwargs)

    def getRDkit(self, *args, **kwargs):
        """Wrapper to get rdkit descriptors."""
        return RDKitDescs(*args, **kwargs)

    def getExtendedValenceSignature(self, *args, **kwargs):
        from ...extra.data.utils.descriptorsets import ExtendedValenceSignature

        return ExtendedValenceSignature(*args, **kwargs)

    def getPredictorDesc(self, *args, **kwargs):
        """Wrapper to get predictors as descriptors."""
        return PredictorDesc(*args, **kwargs)

    def getProDec(self, *args, **kwargs):
        """Wrapper to get protein descriptorsfrom prodec - depends on optional
        dependency."""
        from ...extra.data.utils.descriptorsets import ProDec

        return ProDec(*args, **kwargs)

    def getTanimotoDistances(self, *args, **kwargs):
        """Wrapper to get bulk tanimoto distances."""
        return TanimotoDistances(*args, **kwargs)

    def getSmilesDesc(self, *args, **kwargs):
        """Wrapper to get SMILES as descriptors."""
        return SmilesDesc(*args, **kwargs)

    def getDataFrame(self, *args, **kwargs):
        return DataFrameDescriptorSet(*args, **kwargs)


def get_descriptor(desc_type: str, *args, **kwargs):
    """Wrapper to get a descriptorfunction from _DescriptorSetRetriever."""
    return _DescriptorSetRetriever().getDescriptor(desc_type, *args, **kwargs)

"""Descriptorset: a collection of descriptors that can be calculated for a molecule.
To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""
from abc import ABC, abstractmethod
from typing import Type, Any, Generator

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import Lipinski
from rdkit.Chem import Mol, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from qsprpred.data.storage.interfaces.stored_mol import StoredMol
from ..processing.mol_processor import MolProcessorWithID
from ...logs import logger
from ...utils.serialization import JSONSerializable


class DescriptorSet(JSONSerializable, MolProcessorWithID, ABC):
    """`MolProcessorWithID` that calculates descriptors for a molecule.
    
    Attributes:
        idProp (str): Name of the property to use as the index.
        _notJSON (list[str]): List of attributes that should not be serialized.
    """

    @staticmethod
    def treatInfs(df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values by NaNs.

        Args:
            df: dataframe to treat

        Returns:
            dataframe with infinite values replaced by NaNs
        """
        if np.isinf(df).any().any():
            col_names = df.columns
            x_loc, y_loc = np.where(np.isinf(df.values))
            inf_cols = np.take(col_names, np.unique(y_loc))
            logger.debug(
                "Infinite values in dataframe at columns:"
                f"\n{inf_cols}"
                "And rows:"
                f"\n{np.unique(x_loc)}"
            )
            # Convert absurdly high values to NaNs
            df = df.replace([np.inf, -np.inf], np.NAN)
        return df

    @staticmethod
    def iterMols(
            mols: list[str | Mol], to_list=False
    ) -> list[Mol] | Generator[Mol, None, None]:
        """Create a molecule generator or list from RDKit molecules or SMILES.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)
            to_list: if True, return a list instead of a generator

        Returns:
            generator or list of RDKit molecules
        """
        ret = (Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols)
        if to_list:
            ret = list(ret)
        return ret

    def prepMols(self, mols: list[str | Mol]) -> list[Mol]:
        """Prepare the molecules for descriptor calculation.
        
        Args:
            mols: list of SMILES or RDKit molecules
            
        Returns:
            list of RDKit molecules
        """
        return self.iterMols(mols, to_list=True)

    def __len__(self) -> int:
        """Return the number of descriptors currently calculated by this instance."""
        return len(self.descriptors)

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["descriptors"] = self.descriptors
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.descriptors = state["descriptors"]

    @property
    @abstractmethod
    def descriptors(self) -> list[str]:
        """Return a list of current descriptor names."""

    @descriptors.setter
    @abstractmethod
    def descriptors(self, names: list[str]):
        """Set calculated descriptors for this instance.
        
        Args:
            names: list of descriptor names
        """

    @property
    def isFP(self):
        """Return True if descriptor set is a binary fingerprint."""
        return False

    @abstractmethod
    def __str__(self):
        """Return string representation of the descriptor set.
        This is used to uniquely identify the descriptor set.
        It is used to name the created `DescriptorTable` instances.
        """

    @property
    def supportsParallel(self) -> bool:
        """Return `True` if the descriptor set supports parallel calculation."""
        return True

    @property
    def dtype(self):
        """Return the data type of the descriptor values."""
        return np.float32

    def parsePropsAndMols(self, mols: list[str | Mol], props: dict[str, list[Any]] | None):
        """Parse the properties and molecules passed to the descriptor set.
        
        Args:
            mols: list of SMILES or RDKit molecules
            props: dictionary of properties for the passed molecules
            
        Returns:
            tuple of molecules and properties 
            
        Raises:
            AssertionError: if the properties are not provided for a `StoredMol` instance
        """
        if isinstance(mols[0], StoredMol):
            rd_mols = []
            props = props or dict()
            for mol in mols:
                rd_mols.append(mol.as_rd_mol())
                for mol_prop in mol.props:
                    if mol_prop not in props:
                        props[mol_prop] = []
                    props[mol_prop].append(mol.props[mol_prop])
        else:
            rd_mols = self.iterMols(mols, to_list=True)
            assert props is not None, "Required properties must be provided if not using a StoredMol instance. See the `requiredProps` property."
        return rd_mols, props

    def __call__(
            self, mols: list[str | Mol | StoredMol],
            props: dict[str, list[Any]] | None = None,
            *args,
            **kwargs
    ) -> pd.DataFrame:
        """Calculate the descriptors for a list of molecules and convert them
        to a data frame with the molecule IDs as index. The values are converted
        to the dtype specified by `self.dtype`. Infinite values are replaced by NaNs
        using the `treatInfs` method.

        The molecules are prepared first by calling the `DescriptorSet.prepMols` method.
        If you call `DescriptorSet.getDescriptors` directly, you can skip this step.

        Args:
            mols(list): list of SMILES or RDKit molecules
            props(dict): dictionary of properties for the passed molecules
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            pd.Dataframe: descriptor values of shape (n_mols, n_descriptors)
            
        Raises:
            AssertionError: if the descriptor names are not unique
        """
        if not mols:
            return pd.DataFrame(
                index=pd.Index(name=self.idProp),
                columns=self.descriptors
            ).astype(self.dtype)
        mols, props = self.parsePropsAndMols(mols, props)
        values = self.getDescriptors(self.prepMols(mols), props, *args, **kwargs)
        # check if descriptors have unique names
        assert len(set(self.descriptors)) == len(
            self.descriptors
        ), f"Descriptor names are not unique for set '{self}': {self.descriptors}"
        df = pd.DataFrame(
            values,
            index=pd.Index(props[self.idProp], name=self.idProp),
            columns=self.transformToFeatureNames(),
        )
        try:
            df = df.astype(self.dtype)
        except ValueError as exp:
            logger.warning(
                f"Could not convert descriptor values to '{self.dtype}': {exp}\n"
                f"Keeping original types: {df.dtypes}"
            )
        try:
            df = self.treatInfs(df)
        except TypeError as exp:
            logger.warning(
                f"Could not treat infinite values because of type mismatch: {exp}"
            )
        return df

    def transformToFeatureNames(self):
        return [f"{self}_{x}" for x in self.descriptors]

    @abstractmethod
    def getDescriptors(
            self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Method to calculate descriptors for a list of molecules.

        This method should use molecules as they are without any preparation.
        Any preparation steps should be defined in the `DescriptorSet.prepMols` method.,
        which is picked up by the main `DescriptorSet.__call__`.

        Args:
            mols(list): list of SMILES or RDKit molecules
            props(dict): dictionary of properties
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
           (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
        """


class DataFrameDescriptorSet(DescriptorSet):
    """`DescriptorSet` that uses a `pandas.DataFrame` of precalculated descriptors.
    
    Attributes:
        _df (pd.DataFrame): DataFrame of descriptors
        _cols (list): list of columns to use as the index
        suffix (str): suffix to add to the descriptor name
        idProp (str): Name of the property to use as the index.
        _notJSON (list[str]): List of attributes that should not be serialized.
    """

    @staticmethod
    def setIndex(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Create a multi-index from several columns of the data set.

        Args:
            df (pd.DataFrame): DataFrame to set index for.
            cols (list[str]): List of columns to use as the new multi-index.
            
        Returns:
            pd.DataFrame: DataFrame with the new multi-index set.
        """
        df_index_tuples = df[cols].values
        df_index_tuples = tuple(map(tuple, df_index_tuples))
        df_index = pd.MultiIndex.from_tuples(df_index_tuples, names=cols)
        df.index = df_index
        return df

    def __init__(
            self,
            df: pd.DataFrame,
            joining_cols: list[str] | None = None,
            suffix: str = "",
            source_is_multi_index: bool = False,
    ):
        """Initialize the descriptor set with a dataframe of descriptors.

        Args:
            df:
                dataframe of descriptors
            joining_cols:
                list of columns to use as joining index,
                properties of the same name must exist in the data set
                this descriptor is added to
            suffix:
                suffix to add to the descriptor name
            source_is_multi_index:
                assume that a multi-index is already present in the supplied dataframe.
                If `True`, the `joining_cols` argument must
                also be specified to indicate which properties should
                be used to create the multi-index in the destination.
        """
        super().__init__()
        if source_is_multi_index and not joining_cols:
            raise ValueError(
                "When 'source_is_multi_index=True', 'joining_cols' must be specified."
            )
        self._df = df
        if joining_cols and not source_is_multi_index:
            self._df = self.setIndex(self._df, joining_cols)
        self._cols = joining_cols
        self._descriptors = df.columns.tolist() if df is not None else []
        if joining_cols:
            self._descriptors = [
                col for col in self._descriptors if col not in joining_cols
            ]
        self.suffix = suffix

    @property
    def requiredProps(self) -> list[str]:
        """Return the required properties for the dataframe."""
        prior = super().requiredProps
        new = prior + self._cols if self._cols is not None else prior
        return list(set(new))  # remove duplicates

    def getDF(self):
        """Return the dataframe of descriptors."""
        return self._df

    def getIndex(self):
        """Return the index of the dataframe."""
        return self._df.index if self._df is not None else None

    def getIndexCols(self):
        """Return the index columns of the dataframe."""
        return self._cols if self._df is not None else None

    def getDescriptors(
            self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the descriptors for the input molecules. It simply searches
        for descriptor values in the data frame using the `idProp` as index.

        Args:
            mols: list of SMILES or RDKit molecules
            props: dictionary of properties
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            np.ndarray: descriptor values of shape (n_mols, n_descriptors)
        """
        # create a return data frame with the desired columns as index
        index_cols = self.getIndexCols()
        if index_cols:
            ret = pd.DataFrame(
                # fetch the join columns from our required props
                {col: props[col] for col in index_cols}
            )
            ret = self.setIndex(ret, index_cols)  # set our multi-index
            ret.drop(columns=index_cols, inplace=True)  # only keep the index
        else:
            ret = pd.DataFrame(index=pd.Index(props[self.idProp], name=self.idProp))
        ret = ret.join(
            # join in our descriptors
            # each molecule gets the correct descriptors from the data frame
            self._df,
            how="left",
            on=index_cols,
            validate="one_to_one",
        )
        # ret is in the same order as the input mols, so we can just return the values
        return ret[self.descriptors].values

    @property
    def descriptors(self) -> list[str]:
        """Return the descriptor names."""
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value: list[str]):
        """Set the descriptor names.
        
        Args:
            value (list[str]): list of descriptor names
        """
        self._descriptors = value

    def __str__(self):
        return "DataFrame" if not self.suffix else f"{self.suffix}_DataFrame"


class DrugExPhyschem(DescriptorSet):
    """Various properties used for scoring in DrugEx.
    
        idProp (str): Name of the property to use as the index.
        _notJSON (list[str]): List of attributes that should not be serialized.
    """

    _notJSON = [*DescriptorSet._notJSON, "_prop_dict"]

    def __init__(self, physchem_props: list[str] | None = None):
        """Initialize the descriptorset with Property arguments (a list of properties to
        calculate) to select a subset.

        Args:
            physchem_props: list of properties to calculate
        """
        super().__init__()
        self._prop_dict = self.getPropDict()
        if physchem_props:
            self.props = physchem_props
        else:
            self.props = sorted(self._prop_dict.keys())

    def getDescriptors(self, mols, props, *args, **kwargs):
        """Calculate the DrugEx properties for a molecule.
        
        Args:
            mols: list of SMILES or RDKit molecules
            props: dictionary of properties for the passed molecules
            args: positional arguments
            kwargs: keyword arguments
        
        Returns:
            np.ndarray: array of descriptor values of shape (n_mols, n_descriptors)
        """
        mols = self.iterMols(mols, to_list=True)
        scores = np.zeros((len(mols), len(self.props)))
        for i, mol in enumerate(mols):
            for j, prop in enumerate(self.props):
                try:
                    scores[i, j] = self._prop_dict[prop](mol)
                except Exception as exp:
                    logger.warning(f"Could not calculate {prop} for {mol}.")
                    logger.exception(exp)
                    continue
        return scores

    def __setstate__(self, state):
        super().__setstate__(state)
        self._prop_dict = self.getPropDict()

    @property
    def descriptors(self) -> list[str]:
        """Return the list of properties to calculate.
        
        Returns:
            list[str]: list of property names
        """
        return self.props

    @descriptors.setter
    def descriptors(self, props: list[str]):
        """Set new props as a list of names."""
        self.props = props

    def __str__(self):
        return "DrugExPhyschem"

    @staticmethod
    def getPropDict() -> dict[str, callable]:
        """Return a dictionary of DrugEx properties and their corresponding functions."""
        return {
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


class RDKitDescs(DescriptorSet):
    """Calculate RDkit descriptors.

    Attributes:
        descriptors (list): list of RDKit descriptors to calculate
        include3D (bool): include 3D descriptors
        idProp (str): Name of the property to use as the index.
        _notJSON (list[str]): List of attributes that should not be serialized.
    """

    def __init__(
            self, rdkit_descriptors: list[str] | None = None, include_3d: bool = False
    ):
        """Initialize the descriptorset with a list of RDKit descriptors to calculate.
        
        Args:
            rdkit_descriptors (list[str]): 
                list of descriptors to calculate, if none, all 2D rdkit descriptors
                will be calculated
            include_3d: if True, 3D descriptors will be calculated
        """
        super().__init__()
        self.descriptors = (
            rdkit_descriptors
            if rdkit_descriptors is not None
            else sorted({x[0] for x in Descriptors._descList})
        )
        if include_3d:
            self.descriptors = sorted(
                [
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
            )
        self.include3D = include_3d

    def getDescriptors(
            self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the RDKit descriptors for a molecule.
        
        Args:
            mols (list[Mol]): list of RDKit molecules
            props (dict[str, list[Any]]): 
                dictionary of properties for the passed molecules
            args: positional arguments
            kwargs: keyword arguments
            
        Returns:
            np.ndarray: array of descriptor values of shape (n_mols, n_descriptors)
        """
        mols = self.iterMols(mols, to_list=True)
        scores = np.zeros((len(mols), len(self.descriptors)))
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptors)
        for i, mol in enumerate(mols):
            try:
                scores[i] = calc.CalcDescriptors(mol)
            except AttributeError:
                scores[i] = [np.nan] * len(self.descriptors)
        return scores

    @property
    def descriptors(self) -> list[str]:
        """Return the list of RDKit descriptors to calculate."""
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        """Set the list of RDKit descriptors to calculate.
        
        Args:
            descriptors (list[str]): list of descriptor names
        """
        self._descriptors = descriptors

    def __str__(self):
        return "RDkit"


class TanimotoDistances(DescriptorSet):
    """Calculate Tanimoto distances to a list of SMILES sequences.

    Attributes:
        fingerprintType (Fingerprint): fingerprint type to use.
        _descriptors (list): list of SMILES sequences to calculate the distances.
        _args: `fingerprint` arguments
        _kwargs: `fingerprint` keyword arguments, should contain fingerprint_type
        idProp (str): Name of the property to use as the index.
        _notJSON (list[str]): List of attributes that should not be serialized.
    """

    def __init__(self, list_of_smiles: list[str], fingerprint_type: Type["Fingerprint"], *args, **kwargs):
        """Initialize the descriptorset with a list of SMILES sequences and a
        fingerprint type.

        Args:
            list_of_smiles (list of strings): list of SMILES to calculate the distances.
            fingerprint_type (Fingerprint): fingerprint type to use.
            *args: `fingerprint` arguments
            **kwargs: `fingerprint` keyword arguments, should contain fingerprint_type
        """
        super().__init__()
        self._descriptors = list_of_smiles
        self.fingerprintType = fingerprint_type
        self._args = args
        self._kwargs = kwargs

        # intialize fingerprint calculator
        self.fp = fingerprint_type
        self.fps = self.calculate_fingerprints(list_of_smiles)

    def getDescriptors(
            self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the Tanimoto distances to the list of SMILES sequences.

        Args:
            mols (List[str] or List[rdkit.Chem.rdchem.Mol]): 
                SMILES sequences or RDKit molecules to calculate the distances.
            props (dict): dictionary of properties for the passed molecules
            args: positional arguments for the fingerprint calculator
            kwargs: keyword arguments for the fingerprint calculator
        """
        mols = [
            Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols
        ]
        # Convert np.arrays to BitVects
        fps = [
            DataStructs.CreateFromBitString("".join(map(str, x)))
            for x in self.fp.getDescriptors(mols, props, *args, **kwargs)
        ]
        return np.array(
            [
                list(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, self.fps)))
                for fp in fps
            ]
        )

    def calculate_fingerprints(self, list_of_smiles: list[str]) -> list[DataStructs.BitVect]:
        """Calculate the fingerprints for the list of SMILES sequences
        
        Args:
            list_of_smiles (list[str]): list of SMILES sequences
            
        Returns:
            list[DataStructs.BitVect]: list of fingerprints
        """
        # Convert np.arrays to BitVects
        return [
            DataStructs.CreateFromBitString("".join(map(str, x)))
            for x in self.fp.getDescriptors(
                [Chem.MolFromSmiles(smiles) for smiles in list_of_smiles],
                props={"ID": list_of_smiles},
            )
        ]

    @property
    def descriptors(self) -> list[str]:
        """Return the list of SMILES sequences to calculate the distances."""	
        return self._descriptors

    @descriptors.setter
    def descriptors(self, list_of_smiles: list[str]):
        """Set new list of SMILES sequences to calculate distance to.
        
        Args:
            list_of_smiles (list[str]): list of SMILES sequences
        """
        self._descriptors = list_of_smiles
        self.list_of_smiles = list_of_smiles
        self.fps = self.calculate_fingerprints(self.list_of_smiles)

    def __str__(self):
        return "TanimotoDistances"


class PredictorDesc(DescriptorSet):
    """MoleculeDescriptorSet that uses a Predictor object to calculate descriptors from
    a molecule.
    
    Attributes:
        model (QSPRModel): a fitted model instance
        _descriptors (list): list of descriptors
        idProp (str): Name of the property to use as the index.
        _notJSON (list): List of attributes that should not be serialized.
    """

    _notJSON = [*DescriptorSet._notJSON, "model"]

    def __init__(self, model: Type["QSPRModel"] | str):
        """Initialize the descriptorset with a `QSPRModel` object.

        Args:
            model (QSPRModel): a fitted model instance or a path to the model's meta file
        """
        super().__init__()
        if isinstance(model, str):
            from ...models.model import QSPRModel

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
        from ...models.model import QSPRModel

        self.model = QSPRModel.fromFile(self.model)

    def getDescriptors(self, mols: list[str | Mol], props: dict[str, list[Any]], *args, **kwargs) -> np.ndarray:
        """Calculate the descriptor for a list of molecules.

        Args:
            mols (list): list of smiles or rdkit molecules
            props (dict): dictionary of properties for the passed molecules
            args: positional arguments
            kwargs: keyword arguments

        Returns:
            np.ndarray: array of descriptor values of shape (n_mols, n_descriptors)
        """
        mols = self.iterMols(mols, to_list=True)
        return self.model.predictMols(mols, use_probas=False)

    @property
    def descriptors(self):
        """Return the descriptors names."""
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors: list[str]):
        """Set descriptors to calculate."""
        self._descriptors = descriptors

    def __str__(self):
        return "PredictorDesc"


class SmilesDesc(DescriptorSet):
    """Descriptorset that calculates descriptors from a SMILES sequence.
    
    Attributes:
        idProp (str): Name of the property to use as the index.
        _notJSON (list): List of attributes that should not be serialized
    """

    @staticmethod
    def treatInfs(df: pd.DataFrame) -> pd.DataFrame:
        """handle infinite values in the dataframe
        
        Just return the dataframe as is since SMILES are strings.
        
        Args:
            df (pd.DataFrame): dataframe to treat
        """
        return df

    def getDescriptors(
            self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return smiles as descriptors.

        Args:
            mols (list): list of smiles or rdkit molecules
            props (dict): dictionary of properties for the passed molecules
            args: positional arguments
            kwargs: keyword arguments

        Returns:
            np.ndarray: array of descriptor values of shape (n_mols, n_descriptors)
        """
        if all(isinstance(mol, str) for mol in mols):
            return np.array(mols)
        elif all(isinstance(mol, Mol) for mol in mols):
            return np.array([Chem.MolToSmiles(mol) for mol in mols])
        else:
            raise ValueError("Molecules should be either SMILES or RDKit Mol objects.")

    @property
    def dtype(self):
        """Return the data type of the descriptor values."""
        return str

    @property
    def descriptors(self):
        """Return the descriptor names."""
        return ["SMILES"]

    @descriptors.setter
    def descriptors(self, descriptors):
        """Set the descriptor names.
        
        Ignore the input since SMILES are the only descriptor.
        
        Args:
            descriptors (list): list of descriptor names
        """
        pass

    def __str__(self):
        return "SmilesDesc"

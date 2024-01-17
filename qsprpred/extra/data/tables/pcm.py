from typing import Callable

import pandas as pd

from qsprpred.data.descriptors.calculators import (
    DescriptorsCalculator,
    MoleculeDescriptorsCalculator,
)
from qsprpred.data.tables.mol import MoleculeTable
from qsprpred.data.tables.qspr import QSPRDataset
from qsprpred.extra.data.descriptors.calculators import ProteinDescriptorCalculator
from qsprpred.logs import logger
from qsprpred.tasks import TargetProperty
from qsprpred.utils.serialization import function_as_string, function_from_string


class PCMDataSet(QSPRDataset):
    """Extension of `QSARDataset` for PCM modelling.

    It allows specification of a column with protein identifiers
    and the calculation of protein descriptors.

    Attributes:
        proteinCol (str):
            name of column in df containing the protein target identifier (usually a
            UniProt ID) to use for protein descriptors for PCM modelling
            and other protein related tasks.
        proteinSeqProvider (Callable):
            function that takes a list of protein identifiers and returns a `dict`
            mapping those identifiers to their sequences. Defaults to `None`.
    """

    def __init__(
        self,
        name: str,
        protein_col: str,
        target_props: list[TargetProperty | dict],
        df: pd.DataFrame | None = None,
        smiles_col: str = "SMILES",
        protein_seq_provider: Callable | None = None,
        add_rdkit: bool = False,
        store_dir: str = ".",
        overwrite: bool = False,
        n_jobs: int = 1,
        chunk_size: int = 50,
        drop_invalids: bool = True,
        drop_empty: bool = True,
        index_cols: list[str] | None = None,
        autoindex_name: str = "QSPRID",
        random_state: int | None = None,
        store_format: str = "pkl",
    ):
        """Construct a data set to handle PCM data.

        Args:
            name (str): data name, used in saving the data
            protein_col (str): name of column in df containing the protein target
                identifier (usually a UniProt ID) to use for protein descriptors for PCM
                modelling and other protein related tasks.
            protein_seq_provider: Callable = None, optional):
                function that takes a list of protein identifiers and returns a `dict`
                mapping those identifiers to their sequences. Defaults to `None`.
            target_props (list[TargetProperty | dict]):
                target properties,
                names should correspond with target column name in `df`
            df (pd.DataFrame, optional):
                input dataframe containing smiles and target property.
                Defaults to `None`.
            smiles_col (str, optional):
                name of column in `df` containing SMILES. Defaults to "SMILES".
            add_rdkit (bool, optional):
                if `True`, column with rdkit molecules will be added to `df`.
                Defaults to `False`.
            store_dir (str, optional):
                directory for saving the output data. Defaults to '.'.
            overwrite (bool, optional):
                if `True`, existing data will be overwritten. Defaults to `False`.
            n_jobs (int, optional):
                number of parallel jobs. If <= 0, all available cores will be used.
                Defaults to 1.
            chunk_size (int, optional):
                chunk size for parallel processing. Defaults to 50.
            drop_invalids (bool, optional):
                If `True`, invalid SMILES will be dropped. Defaults to `True`.
            drop_empty (bool, optional):
                If `True`, rows with empty SMILES will be dropped. Defaults to `True`.
            index_cols (List[str], optional):
                columns to be used as index in the dataframe.
                Defaults to `None` in which case a custom ID will be generated.
            autoindex_name (str, optional):
                Column name to use for automatically generated IDs.
            random_state (int, optional):
                random state for reproducibility. Defaults to `None`.
            store_format
                format to use for storing the data ('pkl' or 'csv').

        Raises:
            `ValueError`:
                Raised if threshold given with non-classification task.
        """
        super().__init__(
            name,
            df=df,
            smiles_col=smiles_col,
            add_rdkit=add_rdkit,
            store_dir=store_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            drop_invalids=drop_invalids,
            index_cols=index_cols,
            target_props=target_props,
            drop_empty=drop_empty,
            autoindex_name=autoindex_name,
            random_state=random_state,
            store_format=store_format,
        )
        self.proteinCol = protein_col
        self.proteinSeqProvider = protein_seq_provider

    def getProteinKeys(self) -> list[str]:
        """Return a list of keys identifying the proteins in the data frame.

        Returns:
            keys (list): List of protein keys.
        """
        return self.df[self.proteinCol].unique().tolist()

    def getProteinSequences(self) -> dict[str, str]:
        """Return a dictionary of protein sequences for the proteins in the data frame.

        Returns:
            sequences (dict): Dictionary of protein sequences.
        """
        if not self.proteinSeqProvider:
            raise ValueError(
                "Protein sequence provider not set. Cannot get protein sequences."
            )
        return self.proteinSeqProvider(self.getProteinKeys())

    def addProteinDescriptors(
        self, calculator: ProteinDescriptorCalculator, recalculate=False, featurize=True
    ):
        """
        Add protein descriptors to the data frame.

        Args:
            calculator (ProteinDescriptorCalculator):
                `ProteinDescriptorCalculator` instance to use.
            recalculate (bool):
                Whether to recalculate descriptors even if they are already present
                in the data frame.
            featurize (bool):
                Whether to featurize the descriptors after adding them
                to the data frame.
        """
        if recalculate:
            self.dropDescriptors(calculator)
        elif self.getDescriptorNames(prefix=calculator.getPrefix()):
            logger.warning(
                f"Protein descriptors already exist in {self.name}. "
                f"Use `recalculate=True` to overwrite them."
            )
            return

        if not self.proteinCol:
            raise ValueError(
                "Protein column not set. Cannot calculate protein descriptors."
            )
        # calculate the descriptors
        sequences, info = (
            self.proteinSeqProvider(self.df[self.proteinCol].unique().tolist())
            if self.proteinSeqProvider
            else (None, {})
        )
        descriptors = calculator(self.df[self.proteinCol].unique(), sequences, **info)
        descriptors[self.proteinCol] = descriptors.index.values
        # add the descriptors to the descriptor list
        self.attachDescriptors(calculator, descriptors, [self.proteinCol])
        self.featurize(update_splits=featurize)

    def __getstate__(self):
        o_dict = super().__getstate__()
        if self.proteinSeqProvider:
            o_dict["proteinSeqProvider"] = function_as_string(self.proteinSeqProvider)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.proteinSeqProvider:
            try:
                self.proteinSeqProvider = function_from_string(self.proteinSeqProvider)
            except Exception as e:
                logger.warning(
                    "Failed to load protein sequence provider from metadata. "
                    f"The function object could not be recreated from the code. "
                    f"\nError: {e}"
                    f"\nDeserialized Code: {self.proteinSeqProvider}"
                    f"\nSetting protein sequence provider to `None` for now."
                )
                self.proteinSeqProvider = None

    @staticmethod
    def fromSDF(name, filename, smiles_prop, *args, **kwargs):
        raise NotImplementedError(
            f"SDF loading not implemented for {PCMDataSet.__name__}, yet. "
            f"Use `PCMDataSet.fromMolTable` to convert a `MoleculeTable`"
            f"read from an SDF instead."
        )

    @staticmethod
    def fromMolTable(
        mol_table: MoleculeTable,
        protein_col: str,
        target_props: list[TargetProperty | dict] | None = None,
        name: str | None = None,
        **kwargs,
    ) -> "PCMDataSet":
        """Construct a data set to handle PCM data from a `MoleculeTable`.

        Args:
            mol_table (MoleculeTable):
                `MoleculeTable` instance containing the PCM data.
            protein_col (str):
                name of column in df containing the protein target identifier (usually a
                UniProt ID) to use for protein descriptors for PCM modelling
                and other protein related tasks.
            target_props (list[TargetProperty | dict], optional):
                target properties,
                names should correspond with target column name in `df`
            name (str, optional):
                data name, used in saving the data. Defaults to `None`.
            **kwargs:
                keyword arguments to be passed to the `PCMDataset` constructor.

        Returns:
            PCMDataSet:
                `PCMDataset` instance containing the PCM data.
        """
        kwargs["store_dir"] = (
            mol_table.storeDir if "store_dir" not in kwargs else kwargs["store_dir"]
        )
        name = mol_table.name if name is None else name
        ds = PCMDataSet(
            name, protein_col, target_props=target_props, df=mol_table.getDF(), **kwargs
        )
        if not ds.descriptors and mol_table.descriptors:
            ds.descriptors = mol_table.descriptors
        return ds

    def addFeatures(
        self,
        feature_calculators: list[DescriptorsCalculator] | None = None,
        recalculate: bool = False,
    ):
        """Add features to the feature matrix.

        Args:
            feature_calculators (list[DescriptorsCalculator], optional):
                list of feature calculators to use. Defaults to `None` in which case
            recalculate:
                whether to recalculate features even if they are already present
        """
        for calc in feature_calculators:
            if isinstance(calc, MoleculeDescriptorsCalculator):
                self.addDescriptors(calc, recalculate=recalculate, featurize=False)
            elif isinstance(calc, ProteinDescriptorCalculator):
                self.addProteinDescriptors(
                    calc, recalculate=recalculate, featurize=False
                )
            else:
                raise ValueError("Unknown feature calculator type: %s" % type(calc))
        self.featurize(update_splits=True)
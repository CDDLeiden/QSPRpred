import multiprocessing
import os
import pickle
from multiprocessing import Pool
from typing import Optional, ClassVar, Generator, Literal

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools

from qsprpred.data.tables.searchable import SearchableMolTable
from .pandas import PandasDataTable
from ..chem.matching import match_mol_to_smarts
from ...data.chem.scaffolds import Scaffold
from ...data.chem.standardization import (
    check_smiles_valid,
    chembl_smi_standardizer,
    old_standardize_sanitize,
)
from ...logs import logger
from ...utils.interfaces.summarizable import Summarizable


class DescriptorTable(PandasDataTable):
    """Pandas table that holds descriptor data
    for modelling and other analyses.
    """

    def __init__(
        self,
        calculator,
        name_prefix: str,
        df: Optional[pd.DataFrame] = None,
        store_dir: str = ".",
        overwrite: bool = False,
        key_cols: list | None = None,
        n_jobs: int = 1,
        chunk_size: int = 1000,
    ):
        """Initialize a `DescriptorTable` object.

        Args:
            calculator (DescriptorsCalculator):
                `DescriptorsCalculator` used for descriptor calculation.
            name_prefix (str):
                Prefix to use for the name of the descriptor table.
            df (pd.DataFrame):
                data frame containing the descriptors. If you provide a
                dataframe for a dataset that already exists on disk,
                the dataframe from disk will override the supplied data
                frame. Set 'overwrite' to `True` to override
                the data frame on disk.
            store_dir (str):
                Directory to store the dataset files. Defaults to the
                current directory. If it already contains files with the same name,
                the existing data will be loaded.
            overwrite (bool):
                Overwrite existing dataset.
            key_cols (list):
                list of columns to use as index. If None, the index
                will be a custom generated ID.
            n_jobs (int):
                Number of jobs to use for parallel processing. If <= 0,
                all available cores will be used.
            chunk_size (int):
                Size of chunks to use per job in parallel processing.
        """
        super().__init__(
            f"{name_prefix}_{calculator.getPrefix()}",
            df,
            store_dir,
            overwrite,
            key_cols,
            n_jobs,
            chunk_size,
        )
        self.calculator = calculator

    @property
    def prefix(self):
        """Get the prefix of the descriptors in this table."""
        return self.calculator.getPrefix()

    @property
    def keyCols(self):
        """Get the key columns of this table."""
        return self.indexCols

    def getDescriptors(self):
        """Get the descriptors in this table."""
        return self.df[self.getDescriptorNames()]

    def getDescriptorNames(self):
        """Get the names of the descriptors in this table."""
        return [x for x in self.df.columns if x.startswith(f"{self.prefix}_")]

    def fillMissing(self, fill_value, names):
        """Fill missing values in the descriptor table.

        Args:
            fill_value (float): Value to fill missing values with.
            names (list): List of descriptor names to fill. If `None`, all descriptors
                are filled.
        """
        columns = names if names else self.getDescriptorNames()
        self.df[columns] = self.df[columns].fillna(fill_value)


class MoleculeTable(PandasDataTable, SearchableMolTable, Summarizable):
    """Class that holds and prepares molecule data for modelling and other analyses."""

    _notJSON: ClassVar = PandasDataTable._notJSON + ["descriptors"]

    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        smiles_col: str = "SMILES",
        add_rdkit: bool = False,
        store_dir: str = ".",
        overwrite: bool = False,
        n_jobs: int = 1,
        chunk_size: int = 50,
        drop_invalids: bool = True,
        index_cols: Optional[list[str]] = None,
        autoindex_name: str = "QSPRID",
        random_state: int | None = None,
    ):
        """Initialize a `MoleculeTable` object.

        This object wraps a pandas dataframe and provides short-hand methods to prepare
        molecule data for modelling and analysis.

        Args:
            name (str): Name of the dataset. You can use this name to load the dataset
                from disk anytime and create a new instance.
            df (pd.DataFrame): Pandas dataframe containing the data. If you provide a
                dataframe for a dataset that already exists on disk,
            the dataframe from disk will override the supplied data frame. Set
                'overwrite' to `True` to override the data frame on disk.
            smiles_col (str): Name of the column containing the SMILES sequences
                of molecules.
            add_rdkit (bool): Add RDKit molecule instances to the dataframe.
                WARNING: This can take a lot of memory.
            store_dir (str): Directory to store the dataset files. Defaults to the
                current directory. If it already contains files with the same name,
                the existing data will be loaded.
            overwrite (bool): Overwrite existing dataset.
            n_jobs (int): Number of jobs to use for parallel processing. If <= 0, all
                available cores will be used.
            chunk_size (int): Size of chunks to use per job in parallel processing.
            drop_invalids (bool): Drop invalid molecules from the data frame.
            index_cols (list[str]): list of columns to use as index. If None, the index
                will be a custom generated ID.
            autoindex_name (str): Column name to use for automatically generated IDs.
            random_state (int): Random state to use for shuffling and other random ops.
        """
        super().__init__(
            name,
            df,
            store_dir,
            overwrite,
            index_cols,
            n_jobs,
            chunk_size,
            autoindex_name,
            random_state,
        )
        # the descriptors
        self.descriptors = []
        # settings
        self.smilesCol = smiles_col
        self.includesRdkit = add_rdkit
        # add rdkit molecules if requested
        if self.includesRdkit and "RDMol" not in self.df.columns:
            PandasTools.AddMoleculeColumnToFrame(
                self.df,
                smilesCol=self.smilesCol,
                molCol="RDMol",
                includeFingerprints=False,
            )
            self.includesRdkit = True
        # drop invalid columns
        if drop_invalids:
            self.dropInvalids()

    def searchWithIndex(
        self, index: pd.Index, name: str | None = None
    ) -> "MoleculeTable":
        """
        Create a new table from a list of indices.

        Args:
            index(pd.Index):
                Indices in this table to create the new table from.
            name(str):
                Name of the new table. Defaults to the name of the old table,
                plus the `_searched` suffix.

        Returns:
            MoleculeTable:
                A new table with the molecules from the
                old table with the given indices.
        """
        name = f"{self.name}_searched" if name is None else name
        ret = MoleculeTable(
            name=name,
            df=self.df.loc[index, :],
            smiles_col=self.smilesCol,
            add_rdkit=False,
            store_dir=self.storeDir,
            overwrite=True,
            n_jobs=self.nJobs,
            chunk_size=self.chunkSize,
            drop_invalids=False,
            index_cols=self.indexCols,
        )
        for table, calc in zip(self.descriptors, self.descriptorCalculators):
            ret.descriptors.append(
                DescriptorTable(
                    calc,
                    name_prefix=name,
                    df=table.getDF().loc[index, :],
                    store_dir=table.storeDir,
                    overwrite=True,
                    key_cols=table.indexCols,
                    n_jobs=table.nJobs,
                    chunk_size=table.chunkSize,
                )
            )
        return ret

    def searchOnProperty(
        self, prop_name: str, values: list[str], name: str | None = None, exact=False
    ) -> "MoleculeTable":
        mask = [False] * len(self.df)
        for value in values:
            mask = (
                mask | (self.df[prop_name].str.contains(value))
                if not exact
                else mask | (self.df[prop_name] == value)
            )
        matches = self.df.index[mask]
        return self.searchWithIndex(matches, name)

    def searchWithSMARTS(
        self,
        patterns: list[str],
        operator: Literal["or", "and"] = "or",
        use_chirality: bool = False,
        name: str | None = None,
    ) -> "MoleculeTable":
        """
        Search the molecules in the table with a SMARTS pattern.

        Args:
            patterns:
                List of SMARTS patterns to search with.
            operator (object):
                Whether to use an "or" or "and" operator on patterns. Defaults to "or".
            use_chirality:
                Whether to use chirality in the search.
            name:
                Name of the new table. Defaults to the name of the old table,
                plus the `smarts_searched` suffix.

        Returns:
            (MolTable): A dataframe with the molecules that match the pattern.
        """
        matches = self.df.index[
            self.df[self.smilesCol].apply(
                lambda x: match_mol_to_smarts(
                    x, patterns, operator=operator, use_chirality=use_chirality
                )
            )
        ]
        return self.searchWithIndex(
            matches, name=f"{self.name}_smarts_searched" if name is None else name
        )

    def getSummary(self):
        """
        Make a summary with some statistics about the molecules in this table.
        The summary contains the number of molecules per target and the number of
        unique molecules per target.

        Returns:
            (pd.DataFrame): A dataframe with the summary statistics.

        """
        summary = {
            "mols_per_target": self.df.groupby("accession")
            .count()["InChIKey"]
            .to_dict(),
            "mols_per_target_unique": self.df.groupby("accession")
            .aggregate(lambda x: len(set(x)))["InChIKey"]
            .to_dict(),
        }
        return pd.DataFrame(summary)

    def sample(
        self, n: int, name: str | None = None, random_state: int = None
    ) -> "MoleculeTable":
        """
        Sample n molecules from the table.

        Args:
            n (int):
                Number of molecules to sample.
            name (str):
                Name of the new table. Defaults to the name of the old
                table, plus the `_sampled` suffix.
            random_state (int):
                Random state to use for shuffling and other random ops.

        Returns:
            (MoleculeTable): A dataframe with the sampled molecules.
        """
        random_state = random_state or self.randomState
        name = f"{self.name}_sampled" if name is None else name
        index = self.df.sample(n=n, random_state=random_state).index
        return self.searchWithIndex(index, name=name)

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["descriptors"] = []
        for desc in self.descriptors:
            o_dict["descriptors"].append(os.path.basename(desc.storeDir))
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.descriptors = []
        for desc in state["descriptors"]:
            desc = os.path.join(self.storeDir, desc, f"{desc}_meta.json")
            self.descriptors.append(DescriptorTable.fromFile(desc))

    def toFile(self, filename: str):
        ret = super().toFile(filename)
        for desc in self.descriptors:
            desc.save()
        return ret

    @property
    def descriptorCalculators(self):
        """Get the descriptor calculators for this table."""
        return [x.calculator for x in self.descriptors]

    @staticmethod
    def fromSMILES(name: str, smiles: list, *args, **kwargs):
        """Create a `MoleculeTable` instance from a list of SMILES sequences.

        Args:
            name (str): Name of the data set.
            smiles (list): list of SMILES sequences.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable`
                constructor.
        """
        smilescol = "SMILES"
        df = pd.DataFrame({smilescol: smiles})
        return MoleculeTable(name, df, *args, smiles_col=smilescol, **kwargs)

    @staticmethod
    def fromTableFile(name: str, filename: str, sep="\t", *args, **kwargs):
        """Create a `MoleculeTable` instance from a file containing a table of molecules
        (i.e. a CSV file).

        Args:
            name (str): Name of the data set.
            filename (str): Path to the file containing the table.
            sep (str): Separator used in the file for different columns.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable`
                constructor.
        """
        return MoleculeTable(name, pd.read_table(filename, sep=sep), *args, **kwargs)

    @staticmethod
    def fromSDF(name, filename, smiles_prop, *args, **kwargs):
        """Create a `MoleculeTable` instance from an SDF file.

        Args:
            name (str): Name of the data set.
            filename (str): Path to the SDF file.
            smiles_prop (str): Name of the property in the SDF file containing the
                SMILES sequence.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable`
                constructor.
        """
        # FIXME: the RDKit mols are always added here, which might be unnecessary
        return MoleculeTable(
            name,
            PandasTools.LoadSDF(filename, molColName="RDMol"),
            smiles_col=smiles_prop,
            *args,  # noqa: B026 # FIXME: this is a bug in flake8...
            **kwargs,
        )

    def checkMols(self, throw: bool = True):
        """
        Returns a boolean array indicating whether each molecule is valid or not.
        If `throw` is `True`, an exception is thrown if any molecule is invalid.

        Args:
            throw (bool): Whether to throw an exception if any molecule is invalid.

        Returns:
            mask (pd.Series): Boolean series indicating whether each molecule is valid.
        """
        if self.nJobs > 1:
            with multiprocessing.Pool(self.nJobs) as pool:
                mask = pool.starmap(
                    check_smiles_valid,
                    zip(self.df[self.smilesCol], [throw] * len(self.df)),
                )
            return pd.Series(mask, index=self.df.index)
        else:
            return self.df[self.smilesCol].apply(check_smiles_valid, throw=throw)

    def dropDescriptors(self, calculator: "DescriptorsCalculator"):  # noqa: F821
        """
        Drop descriptors from the data frame
        that were calculated with a specific calculator.

        Args:
            calculator (DescriptorsCalculator): DescriptorsCalculator object to
                use for descriptor calculation.
        """
        to_remove = []
        for idx, calc in enumerate(self.descriptorCalculators):
            if calc.getPrefix() == calculator.getPrefix():
                logger.debug(
                    "Removing existing descriptors with prefix "
                    f"{self.name}_{calculator.getPrefix()}"
                )
                to_remove.append(idx)
        for idx in reversed(to_remove):
            self.descriptors[idx].clearFiles()
            self.descriptors.pop(idx)

    def addCustomDescriptors(
        self,
        calculator: "CustomDescriptorsCalculator",  # noqa: F821
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Add custom descriptors to the data frame using a `CustomDescriptorsCalculator`
        object.

        Args:
            calculator (CustomDescriptorsCalculator): CustomDescriptorsCalculator object
                to use for descriptor calculation.
            recalculate (bool): Whether to recalculate descriptors even if they are
                already present in the data frame. If `False`, existing descriptors
                are kept and no calculation takes place.
            **kwargs: Additional keyword arguments to pass to the calculator.
        """
        if recalculate:
            self.dropDescriptors(calculator)
        elif self.getDescriptorNames(prefix=calculator.getPrefix()):
            logger.warning(
                f"Custom molecular descriptors already exist in {self.name}. "
                "Use `recalculate=True` to overwrite them."
            )
            return
        descriptors = calculator(self.df.index, **kwargs)
        descriptors[self.indexCols] = self.df[self.indexCols]
        self.attachDescriptors(calculator, descriptors, self.indexCols)

    def attachDescriptors(
        self,
        calculator: "DescriptorsCalculator",  # noqa: F821
        descriptors: pd.DataFrame,
        index_cols: list,
    ):
        """Attach descriptors to the data frame.

        Args:
            calculator (DescriptorsCalculator): DescriptorsCalculator object to use for
                descriptor calculation.
            descriptors (pd.DataFrame): DataFrame containing the descriptors to attach.
            index_cols (list): List of column names to use as index.
        """
        self.descriptors.append(
            DescriptorTable(
                calculator,
                self.name,
                descriptors,
                store_dir=self.storeDir,
                n_jobs=self.nJobs,
                overwrite=True,
                key_cols=index_cols,
                chunk_size=self.chunkSize,
            )
        )

    def addDescriptors(
        self,
        calculator: "MoleculeDescriptorsCalculator",  # noqa: F821
        recalculate: bool = False,
        fail_on_invalid: bool = True,
    ):
        """
        Add descriptors to the data frame using a `DescriptorsCalculator` object.

        Args:
            calculator (MoleculeDescriptorsCalculator): DescriptorsCalculator object to
                use for descriptor calculation.
            recalculate (bool): Whether to recalculate descriptors even if they are
                already present in the data frame. If `False`, existing descriptors are
                kept and no calculation takes place.
            fail_on_invalid (bool): Whether to throw an exception if any molecule
                is invalid.
        """
        if recalculate:
            self.dropDescriptors(calculator)
        elif self.getDescriptorNames(prefix=calculator.getPrefix()):
            logger.warning(
                f"Molecular descriptors already exist in {self.name}. "
                "Use `recalculate=True` to overwrite them."
            )
            return

        if fail_on_invalid:
            try:
                self.checkMols(throw=True)
            except Exception as exp:
                logger.error(
                    f"Cannot add descriptors to {self.name} because it contains one or "
                    "more invalid molecules. Remove the invalid molecules from your "
                    "data or try to standardize the data set first with "
                    "'standardizeSmiles()'. You can also pass 'fail_on_invalid=False' "
                    "to remove this exception, but the calculation might not be "
                    "successful or correct. See the following list of invalid molecule "
                    "SMILES for more information:"
                )
                logger.error(
                    self.df[~self.checkMols(throw=False)][self.smilesCol].to_numpy()
                )
                raise exp
        # get the data frame with the descriptors
        descriptors = self.apply(
            calculator, axis=0, subset=[self.smilesCol], result_type="reduce"
        )
        descriptors = descriptors.to_list()
        descriptors = pd.concat(descriptors, axis=0)
        descriptors.index = self.df.index
        descriptors[self.indexCols] = self.df[self.indexCols]
        # add the descriptors to the descriptor list
        self.attachDescriptors(calculator, descriptors, self.indexCols)

    def getDescriptors(self):
        """Get the calculated descriptors as a pandas data frame.

        Returns:
            pd.DataFrame: Data frame containing only descriptors.
        """
        join_cols = set()
        for descriptors in self.descriptors:
            join_cols.update(set(descriptors.keyCols))
        join_cols = list(join_cols)
        ret = self.df[join_cols].copy()
        ret.reset_index(drop=True, inplace=True)
        for descriptors in self.descriptors:
            df_descriptors = descriptors.getDF()
            ret = ret.merge(
                df_descriptors,
                left_on=descriptors.keyCols,
                how="left",
                right_index=True,
                suffixes=("_left", "_right"),
            )
            for x in descriptors.keyCols:
                ret.drop(columns=[f"{x}_right"], inplace=True)
                ret.rename(columns={f"{x}_left": x}, inplace=True)
        ret.set_index(self.df.index, inplace=True)
        ret.drop(columns=join_cols, inplace=True)
        return ret

    def getDescriptorNames(self, prefix: Optional[str] = None):
        """Get the names of the descriptors in the data frame.

        Args:
            prefix (str): Prefix of the descriptors to get. If `None`, all descriptors
                are returned.

        Returns:
            list: list of descriptor names.
        """
        if not prefix:
            prefixes = (
                [f"{self.name}_{x.getPrefix()}" for x in self.descriptorCalculators]
                if self.descriptorCalculators
                else []
            )
        else:
            prefixes = [f"{self.name}_{prefix}"]
        names = []
        for x in self.descriptors:
            if f"{self.name}_{x.prefix}" in prefixes:
                names.extend(x.getDescriptorNames())
        return names

    @property
    def hasDescriptors(self):
        """Check whether the data frame contains descriptors."""
        return len(self.getDescriptorNames()) > 0

    @property
    def smiles(self) -> Generator[str, None, None]:
        """Get the SMILES strings of the molecules in the data frame.

        Returns:
            Generator[str, None, None]: Generator of SMILES strings.
        """
        return iter(self.df[self.smilesCol].values)

    def getProperties(self):
        """Get names of all properties/variables saved in the data frame (all columns).

        Returns:
            list: list of property names.
        """
        return self.df.columns.tolist()

    def hasProperty(self, name):
        """Check whether a property is present in the data frame.

        Args:
            name (str): Name of the property.

        Returns:
            bool: Whether the property is present.
        """
        return name in self.df.columns

    def addProperty(self, name: str, data: list):
        """Add a property to the data frame.

        Args:
            name (str): Name of the property.
            data (list): list of property values.
        """
        if isinstance(data, pd.Series):
            if not np.array_equal(data.index.txt, self.df.index.txt):
                logger.info(
                    f"Adding property '{name}' to data set might be introducing 'nan' "
                    "values due to index with pandas series. Make sure the index of "
                    "the data frame and the series match or convert series to list."
                )
        self.df[name] = data

    def removeProperty(self, name):
        """Remove a property from the data frame.

        Args:
            name (str): Name of the property to delete.
        """
        del self.df[name]

    @staticmethod
    def _scaffold_calculator(mol, scaffold: Scaffold):
        """Just a helper method to calculate the scaffold of a molecule more easily."""
        return scaffold(mol[0])

    def addScaffolds(
        self,
        scaffolds: list[Scaffold],
        add_rdkit_scaffold: bool = False,
        recalculate: bool = False,
    ):
        """Add scaffolds to the data frame.

        A new column is created that contains the SMILES of the corresponding scaffold.
        If `add_rdkit_scaffold` is set to `True`, a new column is created that contains
        the RDKit scaffold of the corresponding molecule.

        Args:
            scaffolds (list): list of `Scaffold` calculators.
            add_rdkit_scaffold (bool): Whether to add the RDKit scaffold of the molecule
                as a new column.
            recalculate (bool): Whether to recalculate scaffolds even if they are
                already present in the data frame.
        """
        for scaffold in scaffolds:
            if not recalculate and f"Scaffold_{scaffold}" in self.df.columns:
                continue
            self.df[f"Scaffold_{scaffold}"] = self.apply(
                self._scaffold_calculator,
                func_args=(scaffold,),
                subset=[self.smilesCol],
                axis=1,
                raw=False,
            )
            if add_rdkit_scaffold:
                PandasTools.AddMoleculeColumnToFrame(
                    self.df,
                    smilesCol=f"Scaffold_{scaffold}",
                    molCol=f"Scaffold_{scaffold}_RDMol",
                )

    def getScaffoldNames(self, include_mols: bool = False):
        """Get the names of the scaffolds in the data frame.

        Args:
            include_mols (bool): Whether to include the RDKit scaffold columns as well.

        Returns:
            list: List of scaffold names.
        """
        return [
            col
            for col in self.df.columns
            if col.startswith("Scaffold_")
            and (include_mols or not col.endswith("_RDMol"))
        ]

    def getScaffolds(self, includeMols: bool = False):
        """Get the subset of the data frame that contains only scaffolds.

        Args:
            includeMols (bool): Whether to include the RDKit scaffold columns as well.

        Returns:
            pd.DataFrame: Data frame containing only scaffolds.
        """
        if includeMols:
            return self.df[
                [col for col in self.df.columns if col.startswith("Scaffold_")]
            ]
        else:
            return self.df[self.getScaffoldNames()]

    @property
    def hasScaffolds(self):
        """Check whether the data frame contains scaffolds.

        Returns:
            bool: Whether the data frame contains scaffolds.
        """
        return len(self.getScaffoldNames()) > 0

    def createScaffoldGroups(self, mols_per_group: int = 10):
        """Create scaffold groups.

        A scaffold group is a list of molecules that share the same scaffold. New
        columns are created that contain the scaffold group ID and the scaffold group
        size.

        Args:
            mols_per_group (int): Number of molecules per scaffold group.
        """
        scaffolds = self.getScaffolds(includeMols=False)
        for scaffold in scaffolds.columns:
            counts = pd.value_counts(self.df[scaffold])
            mask = counts.lt(mols_per_group)
            name = f"ScaffoldGroup_{scaffold}_{mols_per_group}"
            if name not in self.df.columns:
                self.df[name] = np.where(
                    self.df[scaffold].isin(counts[mask].index),
                    "Other",
                    self.df[scaffold],
                )

    def getScaffoldGroups(self, scaffold_name: str, mol_per_group: int = 10):
        """Get the scaffold groups for a given combination of scaffold and number of
        molecules per scaffold group.

        Args:
            scaffold_name (str): Name of the scaffold.
            mol_per_group (int): Number of molecules per scaffold group.

        Returns:
            list: list of scaffold groups.
        """
        return self.df[
            self.df.columns[
                self.df.columns.str.startswith(
                    f"ScaffoldGroup_{scaffold_name}_{mol_per_group}"
                )
            ][0]
        ]

    @property
    def hasScaffoldGroups(self):
        """Check whether the data frame contains scaffold groups.

        Returns:
            bool: Whether the data frame contains scaffold groups.
        """
        return (
            len([col for col in self.df.columns if col.startswith("ScaffoldGroup_")])
            > 0
        )

    def standardizeSmiles(self, smiles_standardizer, drop_invalid=True):
        """Apply smiles_standardizer to the compounds in parallel

        Args:
            smiles_standardizer (): either `None` to skip the
                standardization, `chembl`, `old`, or a partial function that reads
                and standardizes smiles.
            drop_invalid (bool): whether to drop invalid SMILES from the data set.
                Defaults to `True`. If `False`, invalid SMILES will be retained in
                their original form.

        Raises:
            ValueError: when smiles_standardizer is not a callable or one of the
                predefined strings.
        """
        std_jobs = self.nJobs
        if smiles_standardizer is None:
            return
        if callable(smiles_standardizer):
            try:  # Prevents weird error if the user inputs a lambda function
                pickle.dumps(smiles_standardizer)
            except pickle.PicklingError:
                logger.warning("Standardizer is not pickleable. Will set n_jobs to 1")
                std_jobs = 1
            std_func = smiles_standardizer
        elif smiles_standardizer.lower() == "chembl":
            std_func = chembl_smi_standardizer
        elif smiles_standardizer.lower() == "old":
            std_func = old_standardize_sanitize
        else:
            raise ValueError("Standardizer must be either 'chembl', or a callable")
        if std_jobs == 1:
            std_smi = [std_func(smi) for smi in self.df[self.smilesCol].values]
        else:
            with Pool(std_jobs) as pool:
                std_smi = pool.map(std_func, self.df[self.smilesCol].values)
        self.df[self.smilesCol] = std_smi
        if drop_invalid:
            self.dropInvalids()

    def dropInvalids(self):
        """
        Drops invalid molecules from the data set.

        Returns:
            mask (pd.Series): Boolean mask of invalid molecules in the original
                data set.
        """
        invalid_mask = self.checkMols(throw=False)
        self.df.drop(self.df.index[~invalid_mask], inplace=True)
        invalids = (~invalid_mask).sum()
        if invalids > 0:
            logger.warning(f"Dropped {invalids} invalid molecules from the data set.")
        return ~invalid_mask

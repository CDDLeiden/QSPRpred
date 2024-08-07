import os
import shutil
from typing import Generator, Any, Iterable, Sized, ClassVar

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools

from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.interfaces.descriptor_provider import DescriptorProvider
from qsprpred.data.storage.interfaces.mol_processable import MolProcessable
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.stored_mol import StoredMol
from qsprpred.data.storage.tabular.basic_storage import TabularStorageBasic
from .descriptor import DescriptorTable
from ...data.chem.scaffolds import Scaffold
from ...logs import logger


class MoleculeTable(PropertyStorage, MolProcessable, DescriptorProvider):
    """Class that holds and prepares molecule data for modelling and other analyses.

    Attributes:
        smilesCol (str):
            Name of the column containing the SMILES sequences
            of molecules.
        includesRdkit (bool):
            Whether the data frame contains RDKit molecules as one of
            the properties.
        descriptors (list[DescriptorTable]):
            List of `DescriptorTable` objects containing the descriptors
            calculated for this table.
    """

    _notJSON: ClassVar = PropertyStorage._notJSON + ["descriptors", "storage"]

    def __init__(
            self,
            storage: ChemStore,
            path: str = ".",
            random_state: int | None = None,
            store_format: str = "pkl",
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
            store_format (str): Format to use for storing the data ('pkl' or 'csv').
        """
        self.storage = storage
        self.name = f"{self.storage}_mol_table"
        self.randomState = random_state
        self.descriptors = []
        self.path = os.path.abspath(os.path.join(path, self.name))
        self.storeFormat = store_format

    def sample(
            self, n: int, name: str | None = None, random_state: int | None = None
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
        name = f"{self.storage}_sampled" if name is None else name
        df_sample = self.storage.getDF().sample(n=n, random_state=random_state)
        storage = self.storage.fromDF(df_sample, name=name)
        mt = MoleculeTable(storage, random_state=random_state)
        for descs in self.descriptors:
            mt.attachDescriptors(
                descs.calculator,
                descs.getDescriptors(),
                [self.idProp]
            )
        return mt

    @staticmethod
    def fromSMILES(name: str, smiles: list, path: str, *args, **kwargs):
        """Create a `MoleculeTable` instance from a list of SMILES sequences.

        Args:
            name (str): Name of the data set.
            smiles (list): list of SMILES sequences.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable`
                constructor.
        """
        smiles_col = "SMILES"
        df = pd.DataFrame({smiles_col: smiles})
        storage = TabularStorageBasic.fromDF(df, name=name, path=path, *args, **kwargs)
        return MoleculeTable(storage, path=os.path.dirname(storage.path))

    @staticmethod
    def fromTableFile(name: str, filename: str, path: str, sep="\t", *args, **kwargs):
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
        df = pd.read_table(filename, sep=sep)
        storage = TabularStorageBasic.fromDF(df, name=name, path=path, *args, **kwargs)
        return MoleculeTable(storage, path=os.path.dirname(storage.path))

    @staticmethod
    def fromSDF(name, filename, smiles_prop, path: str, *args, **kwargs):
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
        df = PandasTools.LoadSDF(filename, molColName="RDMol")
        storage = TabularStorageBasic.fromDF(
            df,
            name=name,
            smiles_prop=smiles_prop,
            path=path,
            *args,
            **kwargs
        )
        return MoleculeTable(storage, path=os.path.dirname(storage.path))

    @property
    def smiles(self) -> Generator[str, None, None]:
        """Get the SMILES strings of the molecules in the data frame.

        Returns:
            Generator[str, None, None]: Generator of SMILES strings.
        """
        return self.storage.smiles

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
            scaffolds = pd.Series(
                [None] * len(self.storage),
                index=self.storage.getProperty(self.storage.idProp)
            )
            if not recalculate and self.storage.hasProperty(f"Scaffold_{scaffold}"):
                continue
            for scaffolds in self.storage.processMols(scaffold):
                scaffolds.loc[scaffolds.index] = scaffolds.values
            self.storage.addProperty(f"Scaffold_{scaffold}", scaffolds)
            if add_rdkit_scaffold:
                raise NotImplementedError(
                    "Adding RDKit molecules of scaffolds is not yet supported."
                )

    def getScaffoldNames(
            self,
            scaffolds: list[Scaffold] | None = None,
            include_mols: bool = False
    ):
        """Get the names of the scaffolds in the data frame.

        Args:
            scaffolds (list): List of scaffold calculators of scaffolds to include.
            include_mols (bool): Whether to include the RDKit scaffold columns as well.


        Returns:
            list: List of scaffold names.
        """
        all_names = [
            col
            for col in self.getProperties()
            if col.startswith("Scaffold_")
               and (include_mols or not col.endswith("_RDMol"))
        ]
        if scaffolds:
            wanted = [str(x) for x in scaffolds]
            return [x for x in all_names if x.split("_", 1)[1] in wanted]
        return all_names

    def getScaffolds(
            self,
            scaffolds: list[Scaffold] | None = None,
            include_mols: bool = False
    ):
        """Get the subset of the data frame that contains only scaffolds.

        Args:
            include_mols (bool): Whether to include the RDKit scaffold columns as well.

        Returns:
            pd.DataFrame: Data frame containing only scaffolds.
        """
        names = self.getScaffoldNames(scaffolds, include_mols=include_mols)
        return self.getDF()[names]

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
        scaffolds = self.getScaffolds(include_mols=False)
        for scaffold in scaffolds.columns:
            scaffolds = self.getDF()[scaffold]
            counts = pd.value_counts(scaffolds)
            mask = counts.lt(mols_per_group)
            name = f"ScaffoldGroup_{scaffold}_{mols_per_group}"
            groups = np.where(
                scaffolds.isin(counts[mask].index),
                "Other",
                scaffolds,
            )
            self.storage.addProperty(name, groups)

    def getScaffoldGroups(self, scaffold_name: str, mol_per_group: int = 10):
        """Get the scaffold groups for a given combination of scaffold and number of
        molecules per scaffold group.

        Args:
            scaffold_name (str): Name of the scaffold.
            mol_per_group (int): Number of molecules per scaffold group.

        Returns:
            list: list of scaffold groups.
        """
        df = self.getDF()
        return df[
            df.columns[
                df.columns.str.startswith(
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
                len([col for col in self.getProperties() if
                     col.startswith("ScaffoldGroup_")])
                > 0
        )

    @property
    def descsPath(self):
        return os.path.join(self.path, "descriptors")

    def __getstate__(self):
        o_dict = super().__getstate__()
        os.makedirs(self.descsPath, exist_ok=True)
        o_dict["descriptors"] = []
        for desc in self.descriptors:
            o_dict["descriptors"].append(os.path.basename(desc.storeDir))
        o_dict["storage"] = os.path.relpath(self.storage.save(), self.path)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        if hasattr(self, "_json_main"):
            self.path = os.path.abspath(os.path.dirname(self._json_main))
        self.descriptors = []
        for desc in state["descriptors"]:
            desc = os.path.join(self.descsPath, desc, f"{desc}_meta.json")
            self.descriptors.append(DescriptorTable.fromFile(desc))
        self.storage = ChemStore.fromFile(
            os.path.join(self.path, state["storage"])
        )

    @property
    def descriptorSets(self):
        """Get the descriptor calculators for this table."""
        return [x.calculator for x in self.descriptors]

    def generateDescriptorDataSetName(self, ds_set: str | DescriptorSet):
        """Generate a descriptor set name from a descriptor set."""
        return f"Descriptors_{self.name}_{ds_set}"

    def dropDescriptors(self, descriptors: list[str]):
        """Drop descriptors by name. Performs a simple feature selection by removing
        the given descriptor names from the data set.

        Args:
            descriptors (list[str]): List of descriptor names to drop.
        """
        for ds in self.descriptors:
            calc = ds.calculator
            ds_names = calc.transformToFeatureNames()
            to_keep = [x for x in ds_names if x not in descriptors]
            ds.keepDescriptors(to_keep)

    def dropDescriptorSets(
            self,
            descriptors: list[DescriptorSet | str],
            full_removal: bool = False,
    ):
        """
        Drop descriptors from the given sets from the data frame.

        Args:
            descriptors (list[DescriptorSet | str]):
                List of `DescriptorSet` objects or their names. Name of a descriptor
                set corresponds to the result returned by its `__str__` method.
            full_removal (bool):
                Whether to remove the descriptor data (will perform full removal).
                By default, a soft removal is performed by just rendering the
                descriptors inactive. A full removal will remove the descriptorSet from the
                dataset, including the saved files. It is not possible to restore a
                descriptorSet after a full removal.
        """
        # sanity check
        assert (
                len(self.descriptors) != 0
        ), "Cannot drop descriptors because the data set does not contain any."
        if len(descriptors) == 0:
            logger.warning(
                "No descriptors specified to drop. All descriptors will be retained."
            )
            return
        if not isinstance(descriptors[0], str):
            descriptors = [str(x) for x in descriptors]
        # remove the descriptors
        to_remove = []
        to_drop = []
        for name in descriptors:
            for idx, ds in enumerate(self.descriptors):
                calc = ds.calculator
                if name == str(calc):
                    to_drop.extend(ds.getDescriptorNames())
                    if full_removal:
                        to_remove.append(idx)
        self.dropDescriptors(to_drop)
        for idx in reversed(to_remove):
            self.descriptors[idx].clear()
            self.descriptors.pop(idx)

    def restoreDescriptorSets(self, descriptors: list[DescriptorSet | str]):
        """Restore descriptors that were previously removed.

        Args:
            descriptors (list[DescriptorSet | str]):
                List of `DescriptorSet` objects or their names. Name of a descriptor
                set corresponds to the result returned by its `__str__` method.
        """
        if not isinstance(descriptors[0], str):
            descriptors = [str(x) for x in descriptors]
        for name in descriptors:
            for ds in self.descriptors:
                calc = ds.calculator
                if name == str(calc):
                    ds.restoreDescriptors()

    def attachDescriptors(
            self,
            calculator: DescriptorSet,
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
                self.generateDescriptorDataSetName(calculator),
                descriptors,
                store_dir=self.descsPath,
                overwrite=True,
                key_cols=index_cols,
                store_format=self.storeFormat,
            )
        )

    def addDescriptors(
            self,
            descriptors: list[DescriptorSet],
            recalculate: bool = False,
            *args,
            **kwargs,
    ):
        """Add descriptors to the data frame with the given descriptor calculators.

        Args:
            descriptors (list[DescriptorSet]):
                List of `DescriptorSet` objects to use for descriptor
                calculation.
            recalculate (bool):
                Whether to recalculate descriptors even if they are
                already present in the data frame. If `False`, existing descriptors are
                kept and no calculation takes place.
            *args:
                Additional positional arguments to pass to each descriptor set.
            **kwargs:
                Additional keyword arguments to pass to each descriptor set.
        """
        if recalculate and self.hasDescriptors():
            self.dropDescriptorSets(descriptors, full_removal=True)
        to_calculate = []
        for desc_set, exists in zip(descriptors, self.hasDescriptors(descriptors)):
            if exists:
                logger.warning(
                    f"Molecular descriptors already exist in {self.name}. "
                    "Calculation will be skipped. "
                    "Use `recalculate=True` to overwrite them."
                )
            else:
                to_calculate.append(desc_set)
        # get the data frame with the descriptors
        # and attach it to this table as descriptors
        for calculator in to_calculate:
            df_descriptors = []
            for result in self.storage.processMols(
                    calculator, proc_args=args, proc_kwargs=kwargs
            ):
                if not isinstance(result, pd.DataFrame):
                    raise ValueError(
                        f"Expected a pandas DataFrame from the descriptor calculator. "
                        f"Got {type(result)} instead: {result}"
                    )
                result[self.idProp] = result.index.values
                df_descriptors.append(result)
            df_descriptors = pd.concat(df_descriptors, axis=0)
            self.attachDescriptors(calculator, df_descriptors, [self.idProp])

    def updateDescriptors(self):
        for desc_table in self.descriptors:
            self.addDescriptors([desc_table.calculator])

    def getDescriptors(self, active_only=True):
        """Get the calculated descriptors as a pandas data frame.

        Returns:
            pd.DataFrame: Data frame containing only descriptors.
        """
        ret = pd.DataFrame(
            index=pd.Index(
                self.getProperty(self.idProp),
                name=self.idProp
            ))
        for descriptors in self.descriptors:
            df_descriptors = descriptors.getDescriptors(active_only=active_only)
            ret = ret.join(df_descriptors, how="left")
        return ret

    def getDescriptorNames(self):
        """Get the names of the descriptors present for  molecules  in  this data  set.

        Returns:
            list: list of descriptor names.
        """
        names = []
        for ds in self.descriptors:
            names.extend(ds.getDescriptorNames())
        return names

    def hasDescriptors(
            self, descriptors: list[DescriptorSet | str] | None = None
    ) -> bool | list[bool]:
        """Check whether the data frame contains given descriptors.

        Args:
            descriptors (list): list of `DescriptorSet` objects or prefixes of
                descriptors to check for. If `None`,
                all descriptors are checked for and
                a single boolean is returned if any descriptors are found.

        Returns:
            list: list of booleans indicating whether each descriptor is present or not.
        """
        if not descriptors:
            return len(self.getDescriptorNames()) > 0
        else:
            descriptors = [self.generateDescriptorDataSetName(x) for x in descriptors]
            descriptors_in = [x.name for x in self.descriptors]
            ret = []
            for name in descriptors:
                if name in descriptors_in:
                    ret.append(True)
                else:
                    ret.append(False)
            return ret

    @property
    def idProp(self) -> str:
        return self.storage.idProp

    def getProperty(self, name: str, ids: tuple[str] | None = None) -> Iterable[Any]:
        return self.storage.getProperty(name, ids)

    def getProperties(self) -> list[str]:
        return self.storage.getProperties() + self.getDescriptorNames()

    def addProperty(self, name: str, data: Sized, ids: list[str] | None = None):
        self.storage.addProperty(name, data, ids)

    def removeProperty(self, name: str):
        self.storage.removeProperty(name)

    def getSubset(self, subset: list[str],
                  ids: list[str] | None = None) -> "MoleculeTable":
        # FIXME:  this needs to return a new table with properly subsetted descriptors
        raise NotImplementedError()

    # def transformProperties(self, names: list[str],
    #                         transformer: Callable[[Iterable[Any]], Iterable[Any]]):
    #     subset = self.getSubset(names)
    #     ret = pd.concat(list(subset.apply(transformer, on_props=names)))
    #     return ret

    def getDF(self) -> pd.DataFrame:
        return self.getDescriptors().join(self.storage.getDF())

    @classmethod
    def fromDF(cls, df: pd.DataFrame, *args, **kwargs) -> "MoleculeTable":
        return cls(TabularStorageBasic.fromDF(df), *args, **kwargs)

    def apply(self, func: callable, func_args: list | None = None,
              func_kwargs: dict | None = None, on_props: tuple[str, ...] | None = None,
              as_df: bool = True) -> Generator[Iterable[Any], None, None]:
        # TODO: extend this to descriptors as well
        return self.storage.apply(func, func_args, func_kwargs, on_props, as_df)

    def dropEntries(self, ids: tuple[str, ...]):
        self.storage.dropEntries(ids)
        for dset in self.descriptors:
            dset.dropEntries(ids)

    def addEntries(self, ids: list[str], props: dict[str, list],
                   raise_on_existing: bool = True):
        self.storage.addEntries(ids, props, raise_on_existing)

    def __len__(self):
        return len(self.storage)

    def __contains__(self, item):
        return item in self.storage

    def __getitem__(self, item):
        return self.storage[item]

    def save(self):
        """Save the whole storage to disk."""
        self.toFile(self.metaFile)

    def reload(self):
        self.__dict__.update(self.fromFile(self.metaFile).__dict__)

    def clear(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @property
    def metaFile(self) -> str:
        return os.path.join(self.path, "meta.json")

    def toFile(self, filename: str):
        ret = super().toFile(filename)
        for desc in self.descriptors:
            desc.save()
        return ret

    def iterChunks(
            self,
            size: int | None = None,
            on_props: list | None = None
    ) -> Generator[list[StoredMol], None, None]:
        # TODO: extend this to descriptors as well
        return self.storage.iterChunks(size, on_props)

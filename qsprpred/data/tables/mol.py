import os
import shutil
from typing import Generator, Any, Iterable, Sized, ClassVar, Literal, Callable

import numpy as np
import pandas as pd

from qsprpred.data.chem.clustering import MoleculeClusters
from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.processing.mol_processor import MolProcessor
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.stored_mol import StoredMol
from qsprpred.data.storage.tabular.basic_storage import TabularStorageBasic
from .descriptor import DescriptorTable
from .interfaces.molecule_data_set import MoleculeDataSet
from ..chem.identifiers import ChemIdentifier
from ..chem.standardizers import ChemStandardizer
from ...data.chem.scaffolds import Scaffold
from ...logs import logger
from ...utils.parallel import Parallelizable


class MoleculeTable(MoleculeDataSet, Parallelizable):
    """Class that holds and prepares molecule data for modelling and other analyses
    organized as a collection of `PandasDataTable` objects.

    Attributes:
        descriptors (list[DescriptorTable]): List of descriptor tables attached to this
            data set.
        randomState (int): Random state to use for shuffling and other random ops.
        storeFormat (str): Format to use for storing the data set.
        rootDir (str): Path to the directory where the data set is stored.
        storage (ChemStore): The storage object that holds the molecule data.
        path (str): Path to the directory where the data set will be stored.
        name (str): Name of the data set.
    """

    _notJSON: ClassVar = [*PropertyStorage._notJSON, "descriptors", "storage"]

    def __init__(
            self,
            storage: ChemStore | None = None,
            name: str | None = None,
            path: str = ".",
            random_state: int | None = None,
            store_format: str = "pkl",
    ):
        """Initialize a `MoleculeTable` object.

        This object wraps a pandas dataframe and provides short-hand methods to prepare
        molecule data for modelling and analysis.

        Args:
            storage (ChemStore): The storage object that holds the molecule data.
            name (str): Name of the data set.
            path (str): Path to the directory where the data set will be stored.
            random_state (int): Random state to use for shuffling and other random ops.
            store_format (str): Format to use for storing the data set.
        """
        assert storage is not None or name is not None, "Either storage or name must be provided."
        self.descriptors = []
        self.randomState = random_state
        self.storeFormat = store_format
        self.rootDir = path
        name = name or f"{storage}_mol_table"
        if storage is not None:
            self.storage = storage
            self.path = os.path.abspath(os.path.join(self.rootDir, name))
            self.name = name
            if os.path.exists(self.metaFile):
                self.reload()
                if random_state is not None and self.randomState != random_state:
                    logger.warning(
                        "Random state in the data set "
                        "does not match the given random state. Setting to given value:"
                        f" {random_state}."
                    )
                    self.randomState = random_state
        else:
            self.path = os.path.abspath(os.path.join(self.rootDir, name))
            self.name = name
            if os.path.exists(self.metaFile):
                self.reload()
            else:
                raise ValueError(f"Could not initialize from meta file: {self.metaFile}"
                                 f"Are you sure the path parameter is correct? "
                                 f"Path supplied: {self.path}")

    @property
    def randomState(self) -> int:
        """Get the random state to use for shuffling and other random ops."""
        return self._randomState

    @randomState.setter
    def randomState(self, seed: int | None):
        """Set the random state to use for shuffling and other random ops."""
        self._randomState = seed or np.random.randint(0, 2 ** 32 - 1)

    @property
    def name(self) -> str:
        """Get the name of the data set."""
        return self._name

    @name.setter
    def name(self, name: str):
        """Set the name of the data set."""
        self._name = name
        self.path = os.path.abspath(os.path.join(self.rootDir, self.name))

    def sample(
            self, n: int, name: str | None = None, random_state: int | None = None
    ) -> "MoleculeTable":
        """Sample n molecules from the table.

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
        df_sample = self.storage.getDF().sample(n=n, random_state=random_state)
        return self.getSubset(
            self.getProperties(),
            df_sample[self.idProp].values,
            name=name
        )

    @property
    def identifier(self) -> ChemIdentifier:
        """Get the identifier to use for the data set."""
        return self.storage.identifier

    def applyIdentifier(self, identifier: ChemIdentifier):
        """Apply an identifier to the data set.
        
        Args:
            identifier (ChemIdentifier): Identifier to apply.
        """
        self.storage.applyIdentifier(identifier)
        if self.descriptorSets:
            # FIXME: this should not drop the descriptors, but just reindex the data
            self.dropDescriptorSets([str(x) for x in self.descriptorSets],
                                    full_removal=True)
            logger.warning(f"Applied identifier {identifier} to the data set.")
            logger.warning("The data set has been reindexed and the old index is lost.")
            logger.warning(
                "This means that the descriptor data is no longer valid "
                "and has been removed. "
                "You can reload this data set if this is not what you want."
            )

    @property
    def standardizer(self) -> ChemStandardizer:
        """Get the standardizer to use for the data set."""
        return self.storage.standardizer

    def applyStandardizer(self, standardizer: ChemStandardizer):
        """Apply a standardizer to the data set.
        
        Args:
            standardizer (ChemStandardizer): Standardizer to apply.
        """
        self.storage.applyStandardizer(standardizer)
        if self.descriptorSets:
            # FIXME: this should not drop the descriptors, but just reindex the data
            self.dropDescriptorSets([str(x) for x in self.descriptorSets],
                                    full_removal=True)
            logger.warning(f"Applied standardizer {standardizer} to the data set.")
            logger.warning("The data set has been reindexed and the old index is lost.")
            logger.warning(
                "This means that the descriptor data is no longer valid "
                "and has been removed. "
                "You can reload this data set if this is not what you want."
            )

    @property
    def chunkSize(self) -> int:
        """Get the size of chunks to use per job in parallel processing."""
        return self.storage.chunkSize

    @chunkSize.setter
    def chunkSize(self, size: int):
        """Set the size of chunks to use per job in parallel processing.
        
        Args:
            size (int): Size of the chunks.
        """
        self.storage.chunkSize = size

    @property
    def nJobs(self) -> int:
        """Get the number of jobs to use for parallel processing."""
        if hasattr(self.storage, "nJobs"):
            return self.storage.nJobs
        else:
            raise NotImplementedError(
                "The used storage does not seem to support parallelization."
            )

    @nJobs.setter
    def nJobs(self, n_jobs: int):
        """Set the number of jobs to use for parallel processing.
        
        Args:
            n_jobs (int): Number of jobs to use.
            
        Raises:
            NotImplementedError: If the storage does not support parallelization.
        """
        if hasattr(self.storage, "nJobs"):
            self.storage.nJobs = n_jobs
        else:
            raise NotImplementedError(
                "The used storage does not seem to support parallelization."
            )

    @classmethod
    def fromDF(
            cls,
            name: str,
            df: pd.DataFrame,
            path: str = ".",
            smiles_col: str = "SMILES",
            **kwargs,
    ) -> "MoleculeTable":
        """Create a `MoleculeTable` instance from a pandas DataFrame.

        Args:
            name (str): Name of the data set.
            df (pd.DataFrame): DataFrame containing the molecule data.
            path (str): Path to the directory where the data set will be stored.
            smiles_col (str): Name of the column in the data frame containing the SMILES
                sequences.
            **kwargs: 
                Additional keyword arguments to pass to the `MoleculeTable` constructor.
        
        Returns:
            (MoleculeTable): The created data set.
        """
        storage = TabularStorageBasic(f"{name}_storage", path, df,
                                      smiles_col=smiles_col, **kwargs)
        return MoleculeTable(storage, name=name, path=path)

    @classmethod
    def fromSMILES(cls, name: str, smiles: list, path: str, *args, **kwargs):
        """Create a `MoleculeTable` instance from a list of SMILES sequences.

        Args:
            name (str): Name of the data set.
            smiles (list): list of SMILES sequences.
            path (str): Path to the directory where the data set will be stored.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: 
                Additional keyword arguments to pass to the `MoleculeTable` constructor.
                
        Returns:
            (MoleculeTable): The created data set.
        """
        smiles_col = "SMILES"
        df = pd.DataFrame({smiles_col: smiles})
        storage = TabularStorageBasic(name, path, df, *args, **kwargs)
        return cls(storage, path=os.path.dirname(storage.path))

    @classmethod
    def fromTableFile(cls, name: str, filename: str, path: str, *args, sep="\t",
                      **kwargs):
        """Create a `MoleculeTable` instance from a file containing a table of molecules
        (i.e. a CSV file).

        Args:
            name (str): Name of the data set.
            filename (str): Path to the file containing the table.
            path (str): Path to the directory where the data set will be stored.
            sep (str): Separator used in the file for different columns.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: 
                Additional keyword arguments to pass to the `MoleculeTable` constructor.
                
        Returns:
            (MoleculeTable): The created data set.
        """
        df = pd.read_table(filename, sep=sep)
        storage = TabularStorageBasic(f"{name}_storage", path, df)
        return MoleculeTable(storage, name, path=path, *args, **kwargs)

    @classmethod
    def fromSDF(cls, name: str, filename: str, path: str, smiles_prop: str, *args,
                **kwargs):
        """Create a `MoleculeTable` instance from an SDF file.

        Args:
            name (str): Name of the data set.
            filename (str): Path to the SDF file.
            path (str): Path to the directory where the data set will be stored.
            smiles_prop (str): 
                Name of the property in the SDF file containing the SMILES sequence.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: 
                Additional keyword arguments to pass to the `MoleculeTable` constructor.
        """
        # FIXME: the RDKit mols are always added here, which might be unnecessary
        from rdkit.Chem import PandasTools
        df = PandasTools.LoadSDF(filename, molColName="RDMol")
        storage = TabularStorageBasic(
            name,
            path,
            df,
            *args,
            smiles_col=smiles_prop,
            **kwargs
        )
        return cls(storage, path=os.path.dirname(storage.path))

    @property
    def smilesProp(self) -> str:
        """Get the name of the property that contains the SMILES strings."""
        return self.storage.smilesProp

    @property
    def smiles(self) -> Generator[str, None, None]:
        """Generator of SMILES strings of all molecules in the data set."""
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
    ) -> list[str]:
        """Get the names of the scaffolds in the data frame.

        Args:
            scaffolds (list): List of scaffold calculators of scaffolds to include.
            include_mols (bool): Whether to include the RDKit scaffold columns as well.

        Returns:
            (list[str]): List of scaffold names.
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
    ) -> pd.DataFrame:
        """Get the subset of the data frame that contains only scaffolds.

        Args:
            scaffolds (list): List of scaffold calculators of scaffolds to include.
            include_mols (bool): Whether to include the RDKit scaffold columns as well.

        Returns:
            pd.DataFrame: Data frame containing only scaffolds.
        """
        names = self.getScaffoldNames(scaffolds, include_mols=include_mols)
        return self.getDF()[names]

    @property
    def hasScaffolds(self) -> bool:
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

    def getScaffoldGroups(self, scaffold_name: str,
                          mol_per_group: int = 10) -> pd.Series:
        """Get the scaffold groups for a given combination of scaffold and number of
        molecules per scaffold group.

        Args:
            scaffold_name (str): Name of the scaffold.
            mol_per_group (int): Number of molecules per scaffold group.

        Returns:
            (pd.Series): Series containing the scaffold groups.
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
    def hasScaffoldGroups(self) -> bool:
        """Check whether the data frame contains scaffold groups.

        Returns:
            (bool): Whether the data frame contains scaffold groups.
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

    def hasProperty(self, name: str) -> bool:
        """Check whether a property is present in the data frame.
        
        Args:
            name (str): Name of the property.
        """
        return self.storage.hasProperty(name)

    @property
    def descriptorSets(self) -> list[DescriptorSet]:
        """Get the descriptor calculators for this table."""
        return [x.calculator for x in self.descriptors]

    def generateDescriptorDataSetName(self, ds_set: str | DescriptorSet) -> str:
        """Generate a descriptor set name from a descriptor set.
        
        Args:
            ds_set (str): Name of the descriptor set.
        
        Returns:
            (str): Name of the descriptor set.
        """
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
        """Drop descriptors from the given sets from the data frame.

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
        
        Raises:
            AssertionError: If the data set does not contain any descriptors.
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
        Raises:
            ValueError: If any of the descriptors are not present in the data set.
        """
        if not isinstance(descriptors[0], str):
            descriptors = [str(x) for x in descriptors]
        for name in descriptors:
            restored = False
            for ds in self.descriptors:
                calc = ds.calculator
                if name == str(calc):
                    ds.restoreDescriptors()
                    restored = True
            if not restored:
                raise ValueError(
                    f"Could not restore descriptors for '{name}'. "
                    "The descriptor set was not found in the data set."
                )

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
                index_cols=index_cols,
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
            before = list(df_descriptors.columns)
            df_descriptors = df_descriptors[
                calculator.transformToFeatureNames() + [self.idProp]]
            after = list(df_descriptors.columns)
            if len(before) != len(after):
                logger.warning(
                    f"Descriptor set {calculator} has been reduced from "
                    f"{len(before)} to {len(after)} descriptors."
                    "Returned data frame contained more columns than expected."
                    f"Extra columns: {set(before) - set(after)}"
                )
            self.attachDescriptors(calculator, df_descriptors, [self.idProp])

    def getDescriptors(self, active_only: bool = True) -> pd.DataFrame:
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

    def getDescriptorNames(self) -> list[str]:
        """Get the names of the descriptors present for  molecules  in  this data  set.

        Returns:
            (list[str]): list of descriptor names.
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
            (list[DescriptorSet | str] | None):
                List of descriptor objects or prefixes of descriptors to check for. 
                If `None`, all descriptors are checked for and a single boolean is 
                returned if any descriptors are found.

        Returns:
            (bool | list[bool]): Whether the data frame contains the given descriptors.
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
        """Get the name of the property that contains the molecule IDs."""
        return self.storage.idProp

    def getProperty(self, name: str, ids: tuple[str] | None = None) -> Iterable[Any]:
        """Get the property with the given name.
        
        Args:
            name (str): Name of the property.
            ids (tuple[str], optional): IDs of the molecules to get the property for.
        
        Returns:
            (Iterable[Any]): Property values.
        """
        return self.storage.getProperty(name, ids)

    def getProperties(self) -> list[str]:
        """Get the names of the properties in the data frame."""
        return self.storage.getProperties()

    def addProperty(self, name: str, data: Sized, ids: list[str] | None = None):
        """Add a property to the data frame.

        Args:
            name (str): Name of the property.
            data (Sized): Property values.
            ids (list[str], optional): IDs of the molecules to add the property for.
        
        Returns:
            (bool): Whether the property was added successfully.
        """
        return self.storage.addProperty(name, data, ids)

    def removeProperty(self, name: str) -> bool:
        """Remove a property from the data frame.
        
        Args:
            name (str): Name of the property.
        
        Returns:
            (bool): Whether the property was removed successfully.
        """
        return self.storage.removeProperty(name)

    def getSubset(
            self,
            subset: Iterable[str],
            ids: Iterable[str] | None = None,
            name: str | None = None,
            path: str = ".",
            **kwargs,
    ) -> "MoleculeTable":
        """Get a subset of the data frame.
        
        Args:
            subset (Iterable[str]): List of properties to include in the subset.
            ids (Iterable[str], optional): IDs of the molecules to include in the subset.
            name (str, optional): Name of the new data set.
            path (str): Path to the directory where the data set will be stored.
            **kwargs: 
                Additional keyword arguments to pass to the `MoleculeTable` constructor.
        
        Returns:
            (MoleculeTable): The created data set.
        """
        name = name or f"{self.name}_subset"
        path = path or self.path
        store_subset = self.storage.getSubset(subset, ids)
        ret = MoleculeTable(store_subset, name, path, **kwargs)
        descriptors = []
        for desc in self.descriptors:
            descriptors.append(desc.getSubset(
                self.getDescriptorNames(),
                ids,
                name=name,
                path=ret.descsPath,
            ))
        ret.descriptors = descriptors
        return ret

    def transformProperties(
            self,
            names: list[str],
            transformer: Callable[[Iterable[Any]], Iterable[Any]]
    ):
        """Transform the properties of the data frame.
        
        Args:
            names (list[str]): List of property names to transform.
            transformer (Callable): Function to use for transformation.
        """
        subset = self.getDF()[names]
        ret = subset.apply(transformer, axis=1)
        for col in ret.columns:
            self.addProperty(f"{col}_before_transform", subset[col])
            self.addProperty(col, ret[col])

    def getDF(self) -> pd.DataFrame:
        """Get the data frame of the data set."""
        return self.storage.getDF()

    def apply(
            self,
            func: callable,
            func_args: list | None = None,
            func_kwargs: dict | None = None,
            on_props: tuple[str, ...] | None = None,
            chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
    ) -> Generator[Iterable[Any], None, None]:
        """Apply a function to the data set.
        
        Args:
            func (callable): Function to apply.
            func_args (list, optional): Positional arguments to pass to the function.
            func_kwargs (dict, optional): Keyword arguments to pass to the function.
            on_props (tuple[str, ...], optional): Properties to apply the function on.
            chunk_type (Literal["mol", "smiles", "rdkit", "df"], optional): 
                Type of chunks to use for processing.
        
        Returns:
            (Generator[Iterable[Any], None, None]): Generator of the results.
        """
        return self.storage.apply(func, func_args, func_kwargs, on_props, chunk_type)

    def dropEntries(self, ids: Iterable[str]):
        """Drop entries from the data set.
        
        Args:
            ids (Iterable[str]): IDs of the entries to drop.
        """
        # FIXME: do not drop from storage here, but just mask the removed entries
        self.storage.dropEntries(ids)
        for dset in self.descriptors:
            dset.dropEntries(ids)

    def addEntries(
            self,
            ids: list[str],
            props: dict[str, list],
            raise_on_existing: bool = True
    ):
        """Add entries to the data set.
        
        Args:
            ids (list[str]): IDs of the entries to add.
            props (dict[str, list]): Properties to add.
            raise_on_existing (bool): 
            Whether to raise an error if the entries already exist.
        
        Raises:
            NotImplementedError: Adding entries is not yet available for the data set.
        """
        # FIXME: make sure descriptors are calculated for new entries as well
        raise NotImplementedError("Adding entries not yet available for MoleculeTable.")

    def __len__(self):
        """Get the number of molecules in the data set."""
        return len(self.storage)

    def __contains__(self, item):
        return item in self.storage

    def __getitem__(self, item):
        return self.storage[item]

    def save(self):
        """Save the whole storage to disk."""
        self.toFile(self.metaFile)

    def reload(self):
        """Reload the data set from disk."""
        self.__dict__.update(self.fromFile(self.metaFile).__dict__)

    def clear(self):
        """Clear the data set from memory and disk."""
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @property
    def metaFile(self) -> str:
        """Get the path to the meta file of the data set."""
        return os.path.join(self.path, "meta.json")

    def toFile(self, filename: str):
        """Save the data set to a file.
        
        Args:
            filename (str): Path to the file to save the data set to.
        """
        ret = super().toFile(filename)
        for desc in self.descriptors:
            desc.save()
        return ret

    def iterChunks(
            self,
            size: int | None = None,
            on_props: list | None = None,
            chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
    ) -> Generator[list[StoredMol], None, None]:
        """Iterate over chunks of the data set.
        
        Args:
            size (int, optional): Size of the chunks.
            on_props (list, optional): Properties to iterate over.
            chunk_type (Literal["mol", "smiles", "rdkit", "df"], optional): 
                Type of chunks to use for processing.
        
        Returns:
            (Generator[list[StoredMol], None, None]): Generator of the chunks.
        """
        # TODO: extend this to descriptors as well
        return self.storage.iterChunks(size, on_props, chunk_type)

    def getSummary(self) -> pd.DataFrame:
        """Get a summary of the data set.	
        
        Returns:	
            (pd.DataFrame): Summary of the data set.
            
        Raises:
            NotImplementedError: Summary not yet available for MoleculeTable.	
        """
        raise NotImplementedError("Summary not yet available for MoleculeTable.")

    def searchWithSMARTS(
            self,
            patterns: list[str],
            operator: Literal["or", "and"] = "or",
            use_chirality: bool = False,
            name: str | None = None,
            path: str = "."
    ) -> "MoleculeTable":
        """Search the data set with SMARTS patterns.
        
        Args:
            patterns (list[str]): List of SMARTS patterns to search for.
            operator (Literal["or", "and"]): Operator to use for combining the patterns.
            use_chirality (bool): Whether to use chirality in the search.
            name (str): Name of the new table.
            path (str): Path to the directory where the new table will be stored.
            
        Returns:
            (MoleculeTable): Data set containing the search results.
        """
        if hasattr(self.storage, "searchWithSMARTS"):
            result = self.storage.searchWithSMARTS(
                patterns,
                operator,
                use_chirality,
                name,
            )
            mol_ids = result.getProperty(result.idProp)
            return self.getSubset(self.getProperties(), mol_ids, name=name, path=path)
        raise NotImplementedError(
            "The underlying storage does not support SMARTS search."
        )

    def searchOnProperty(
            self,
            prop_name: str,
            values: list[float | int | str],
            exact=False, name: str | None = None,
            path: str = "."
    ) -> "MoleculeTable":
        """Search the data set based on a property.	
        
        Args:
            prop_name (str): Name of the property to search on.
            values (list[float | int | str]): Values to search for.
            exact (bool): Whether to perform an exact search.
            name (str): Name of the new table.
            path (str): Path to the directory where the new table will be stored.
        
        Returns:
            (MoleculeTable): Data set containing the search results.
        """
        result = self.storage.searchOnProperty(prop_name, values, exact)
        mol_ids = result.getProperty(result.idProp)
        return self.getSubset(self.getProperties(), mol_ids, name=name, path=path)

    def addClusters(
            self,
            clusters: list[MoleculeClusters],
            recalculate: bool = False,
    ):
        """Add clusters to the data frame.

        A new column is created that contains the identifier of the corresponding
        cluster calculator.

        Args:
            clusters (list): list of `MoleculeClusters` calculators.
            recalculate (bool): Whether to recalculate clusters even if they are
                already present in the data frame.
        """
        for cluster in clusters:
            if not recalculate and f"Cluster_{cluster}" in self.getProperties():
                continue
            for clusters in self.storage.processMols(cluster):
                self.addProperty(f"Cluster_{cluster}", clusters.values,
                                 clusters.index.values)

    def getClusterNames(
            self, clusters: list[MoleculeClusters] | None = None
    ) -> list[str]:
        """Get the names of the clusters in the data frame.
        
        Args:
            clusters (list): List of cluster calculators of clusters to include

        Returns:
            (list[str]): List of cluster names.
        """
        all_names = [
            col
            for col in self.getProperties()
            if col.startswith("Cluster_")
        ]
        if clusters:
            wanted = [str(x) for x in clusters]
            return [x for x in all_names if x.split("_", 1)[1] in wanted]
        return all_names

    def getClusters(
            self, clusters: list[MoleculeClusters] | None = None
    ):
        """Get the subset of the data frame that contains only clusters.
        
        Args:
            clusters (list): List of cluster calculators of clusters to include.

        Returns:
            pd.DataFrame: Data frame containing only clusters.
        """
        names = self.getClusterNames(clusters)
        return self.getDF()[names]

    @property
    def hasClusters(self) -> bool:
        """Check whether the data frame contains clusters.

        Returns:
            bool: Whether the data frame contains clusters.
        """
        return len(self.getClusterNames()) > 0

    def imputeProperties(self, names: list[str], imputer: Callable):
        """Impute missing property values.

        Args:
            names (list):
                List of property names to impute.
            imputer (Callable):
                imputer object implementing the `fit_transform`
                 method from scikit-learn API.
        """
        df_subset = self.getDF()[names].copy()
        assert hasattr(imputer, "fit_transform"), (
            "Imputer object must implement the `fit_transform` "
            "method from scikit-learn API."
        )
        assert all(
            name in df_subset.columns for name in names
        ), "Not all properties in dataframe columns for imputation."
        names_old = [f"{name}_before_impute" for name in names]
        df_subset[names_old] = df_subset[names]
        df_subset[names] = imputer.fit_transform(df_subset[names])
        for name in df_subset.columns:
            self.addProperty(name, df_subset[name])
        logger.debug(f"Imputed missing values for properties: {names}")
        logger.debug(f"Old values saved in: {names_old}")

    def processMols(
            self,
            processor: MolProcessor,
            proc_args: tuple[Any, ...] | None = None,
            proc_kwargs: dict[str, Any] | None = None,
            mol_type: Literal["smiles", "mol", "rdkit"] = "mol",
            add_props: Iterable[str] | None = None,
    ) -> Generator[Any, None, None]:
        """Process molecules in the data set.	
        
        Args:
            processor (MolProcessor): Processor to use for molecule processing.
            proc_args (tuple, optional): Positional arguments to pass to the processor.
            proc_kwargs (dict, optional): Keyword arguments to pass to the processor.
            mol_type (Literal["smiles", "mol", "rdkit"], optional): 
                Type of molecules to process.
            add_props (Iterable[str], optional): 
                Additional properties to add to the data frame.
        
        Returns:
            (Generator[Any, None, None]): Generator of the results.
        """
        return self.storage.processMols(processor, proc_args, proc_kwargs,
                                        mol_type=mol_type, add_props=add_props)

import json
import os
from typing import Callable, Generator

import pandas as pd


from .interfaces.qspr_data_set import QSPRDataSet
from ...logs import logger
from ...tasks import TargetSpec, TargetTasks
from ..storage.interfaces.chem_store import ChemStore
from .mol import MoleculeTable
from qsprpred.data.sampling.splits import DataSplit
from qsprpred.data.processing.data_filters import DataFilter
from qsprpred.data.processing.target_transformers import Discretizer
import numpy as np


class QSPRTable(QSPRDataSet, MoleculeTable):
    """Implementation of `QSPRDataSet` using a collection of `PandasDataTable` objects.

    Attributes:
        targetProperties (str): property to be predicted with QSPRmodel
    """

    def __init__(
        self,
        storage: ChemStore | None = None,
        name: str | None = None,
        target_props: list[TargetSpec | dict] | None = None,
        path: str = ".",
        random_state: int | None = None,
        store_format: str = "pkl",
        drop_empty_target_props: bool = True,
    ):
        """Construct QSPRdata, also apply transformations of output property if
        specified.

        Args:
            storage (ChemStore | None):
                storage object to use for saving the data. Defaults to `None`.
            name (str):
                data name, used in saving the data
            target_props (list[TargetSpec | dict] | None):
                target properties, names should correspond with target column names
                in df. If `None`, target specifications will be inferred if this data 
                set has been saved previously. Defaults to `None`.
            path (str, optional): 
                path to the directory where the data set will be saved. Defaults to ".".
            random_state (int, optional): random state for splitting the data.
            store_format (str, optional):
                format to use for storing the data ('pkl' or 'csv').
            drop_empty_target_props (bool, optional):
                whether to ignore entries with empty target properties. Defaults to
                `True`.

        Raises:
            `ValueError`: Raised if threshold given with non-classification task.
        """
        super().__init__(
            storage=storage,
            name=name or f"{storage}_qspr_data",
            path=path,
            random_state=random_state,
            store_format=store_format,
        )
        # load target specifications if not specified and file exists
        if target_props is None and os.path.exists(self.metaFile):
            meta = json.load(open(self.metaFile, "r"))
            target_props = meta["py/state"]["_targetProperties"]
            target_props = [
                TargetSpec.fromJSON(json.dumps(x)) for x in target_props
            ]
        elif target_props is None:
            raise ValueError("Target specifications must be specified for a new QSPRTable.")
        # populate feature matrix and target specifications
        self._targetProperties = []
        self.setTargetProperties(target_props, drop_empty_target_props)
        logger.info(
            f"Dataset '{self.name}' created for "
            f"Targets: '{self.targetProperties}'. "
            f"Number of samples: {len(self.storage)}. "
        )
        self.splits = {}

    @property
    def targetProperties(self) -> list[TargetSpec]:
        """Returns the specifications of target properties of the dataset."""
        return self._targetProperties

    @targetProperties.setter
    def targetProperties(self, target_properties: list[TargetSpec]):
        """Set the target properties of the dataset."""
        raise NotImplementedError(
            "targetProperties is a read-only property. Use `setTargetProperties` to set "
            "the target properties."
        )

    @classmethod
    def fromDF(
        cls,
        name: str,
        df: pd.DataFrame,
        target_props: list[TargetSpec | dict],
        path: str = ".",
        smiles_col: str = "SMILES",
        drop_empty_target_props: bool = True,
        **kwargs,
    ) -> "QSPRTable":
        """Create `QSPRTable` from a pandas DataFrame.

        Args:
            name (str): name of the data set
            df (pd.DataFrame): data frame containing the data
            target_props (list[TargetProperty | dict]): target properties to use
            path (str): path to the directory where the data set will be saved
            smiles_col (str): name of the column containing SMILES
            drop_empty_target_props (bool, optional): whether to drop rows with empty
                target property values. Defaults to `True`.
            **kwargs: additional keyword arguments for `MoleculeTable` constructor

        Returns:
            QSPRTable: created data set
        """
        mt = super().fromDF(name, df, path, smiles_col, **kwargs)
        return QSPRTable.fromMolTable(mt, target_props, name=name, path=path, drop_empty_target_props=drop_empty_target_props)

    @classmethod
    def fromTableFile(
        cls,
        name: str,
        filename: str,
        path: str,
        *args,
        sep: str = "\t",
        target_props: list[TargetSpec | dict] | None = None,
        **kwargs,
    ):
        r"""Create `QSPRTable` from table file (i.e. CSV or TSV).

        Args:
            name (str): name of the data set
            filename (str): path to the table file
            path (str): path to the directory where the data set will be saved
            *args: additional arguments for `MolTable` constructor
            sep (str, optional): separator in the table file. Defaults to "\t".
            target_props (list[TargetProperty | dict], optional): target properties to
                use. Defaults to `None`.
            **kwargs: additional keyword arguments for `MolTable` constructor

        Returns:
            QSPRTable: `QSPRTable` object
        """
        mt = super().fromTableFile(name, filename, path, *args, sep=sep, **kwargs)
        return QSPRTable.fromMolTable(mt, target_props, name=mt.name, path=path)

    @classmethod
    def fromSDF(cls, name: str, filename: str, smiles_prop: str, *args, **kwargs):
        """Create `QSPRTable` from SDF file.

        It is currently not implemented for `QSPRTable`, but you can convert from
        'MoleculeTable' with the 'fromMolTable' method.

        Args:
            name (str): name of the data set
            filename (str): path to the SDF file
            smiles_prop (str): name of the property in the SDF file containing SMILES
            *args: additional arguments for `QSPRTable` constructor
            **kwargs: additional keyword arguments for `QSPRTable` constructor
        """
        raise NotImplementedError(
            f"SDF loading not implemented for {QSPRTable.__name__}, yet. You can "
            "convert from 'MoleculeTable' with 'fromMolTable'."
        )

    @classmethod
    def fromMolTable(
        cls,
        mol_table: MoleculeTable,
        target_props: list[TargetSpec | dict],
        *args,
        path: str = ".",
        name: str | None = None,
        **kwargs,
    ) -> "QSPRTable":
        """Create QSPRTable from a MoleculeTable.

        Args:
            mol_table (MoleculeTable): `MoleculeTable` to use as the data source
            target_props (list): list of target properties to use
            *args:
                additional positional arguments to pass to the constructor of
                `QSPRTable`
            path (str): path to the directory where the data set will be saved
            name (str): name of the data set
            **kwargs:
                additional keyword arguments to pass to the constructor of `QSPRTable`

        Returns:
            QSPRTable: created data set
        """
        name = mol_table.name if name is None else name
        kwargs["random_state"] = (
            mol_table.randomState
            if "random_state" not in kwargs else kwargs["random_state"]
        )
        kwargs["store_format"] = (
            mol_table.storeFormat
            if "store_format" not in kwargs else kwargs["store_format"]
        )
        ds = QSPRTable(
            mol_table.storage,
            name,
            target_props,
            path,
            *args,
            **kwargs,
        )
        ds.descriptors = mol_table.descriptors
        return ds

    def addTargetProperty(self, target_spec: TargetSpec | dict, drop_empty: bool = True):
        """Add a target property to the dataset.

        Args:
            target_spec (TargetSpec | dict):
                target property specification to add or dictionary to initialize a
                TargetSpec
            drop_empty (bool):
                whether to drop rows with empty target property values. Defaults to
                `True`.
        """
        logger.debug(f"Adding target property '{target_spec}' to dataset.")
        if isinstance(target_spec, dict):
            target_spec = TargetSpec.fromDict(target_spec)
        assert (
            target_spec.name in self.getProperties()
        ), f"Property {target_spec.name} not found in data set."
        self.restoreTargetProperty(target_spec)
        if target_spec.name in self.targetPropertiesNames:
            logger.warning(
                f"Target property '{target_spec}' already exists in dataset. It will be overwritten."
            )
            self._targetProperties = [
                tp for tp in self.targetProperties if tp.name != target_spec.name
            ]
        self._targetProperties.append(target_spec)
        if target_spec.task.isClassification():
            self.makeClassification(target_spec.name, target_spec.th)
            self.checkClassification(target_spec.name)
        if drop_empty:
            self.dropEmptyEntries([target_spec.name])

    def getTargetSpecs(self, names: list | None) -> list[TargetSpec]:
        """Get the target specifications with the given names.

        Args:
            names (list[str]): name of the target properties

        Returns:
            (list[TargetSpec]): list of target specifications
        """
        if names is None:
            return self.targetProperties
        if not all(name in self.targetPropertiesNames for name in names):
            logger.warning(
                f"Some target properties {names} not found in dataset. "
                f"Available target properties: {self.targetPropertiesNames}"
            )
        return [tp for tp in self.targetProperties if tp.name in names]

    def getTargetSpec(self, name: str) -> TargetSpec:
        """Get the target specification of a single target property by its name.

        Args:
            name (str): name of the target property

        Returns:
            TargetSpec: target specification with the given name

        Raises:
            ValueError: if the target property with the given name is not found
        """
        for tp in self.targetProperties:
            if tp.name == name:
                return tp
        raise ValueError(f"Target property '{name}' not found in dataset.")

    def setTargetProperties(
        self,
        target_props: list[TargetSpec | dict],
        drop_empty: bool = True,
    ):
        """Set list of target properties for the dataset.

        Args:
            target_props (list[TargetSpec | dict]):
                list of target properties specifications or dictionaries to initialize
                the TargetSpec objects from.
            drop_empty (bool, optional):
                whether to drop rows with empty target property values. Defaults to
                `True`.
        """
        assert isinstance(target_props, list), (
            "target_props should be a list of TargetSpec objects or dictionaries to "
            "initialize TargetSpec objects from. Not a %s." % type(target_props)
        )
        if isinstance(target_props[0], dict):
            assert all(isinstance(d, dict) for d in target_props), (
                "target_props should be a list of TargetSpec objects or "
                "dictionaries to initialize TargetSpec objects from, not a mix."
            )
            target_props = TargetSpec.fromList(target_props)
        else:
            assert all(isinstance(d, TargetSpec) for d in target_props), (
                "target_props should be a list of TargetSpec objects or "
                "dictionaries to initialize TargetSpec objects from, not a mix."
            )
        self._targetProperties = []
        for prop in target_props:
            self.addTargetProperty(prop, drop_empty)

    def unsetTargetProperty(self, name: str | TargetSpec):
        """Unset a target property. It will not remove it from the data set, but
        will make it unavailable for training.

        Args:
            name (str | TargetSpec):
                name or specification of the target property to drop
        """
        name = name.name if isinstance(name, TargetSpec) else name
        assert (
            name in self.targetPropertiesNames
        ), f"Target property '{name}' not found in dataset."
        assert (
            len(self.targetProperties) > 1
        ), "Cannot drop task from single-task dataset."
        self._targetProperties = [tp for tp in self.targetProperties if tp.name != name]

    def restoreTargetProperty(self, prop: TargetSpec | str):
        """Reset target property to its original value.

        Args:
            prop (TargetProperty | str): target property to reset
        """
        if isinstance(prop, str):
            prop = self.getTargetSpec(prop)
        if f"{prop.name}_original" in self.getProperties():
            # restore original values
            self.addProperty(prop.name, self.getProperty(f"{prop.name}_original"))
        else:
            # save original values for next reset
            self.addProperty(f"{prop.name}_original", self.getProperty(prop.name))
    
    def makeClassification(
        self,
        target_property: str,
        th: list[float] | None = None,
    ):
        """Switch to classification task using the given threshold values.

        Args:
            target_property (str):
                Name of target property to use for classification
            th (list[float], optional):
                list of threshold values. If not provided, it is assumed that
                the target property is already discretized and can be used for
                classification.
        """
        assert isinstance(th, (list, type(None))), (
            "Thresholds must be a list of floats or None. "
            f"Got {type(th)} instead."
        )
        if isinstance(th, list):
            assert len(th) > 0, (
                "Thresholds must be a non-empty list of floats. "
            )
            assert len(th) == 1 or len(th) > 3, (
                "Thresholds must be a single float for binary classification or "
                "a list of at least 3 floats for multi-class classification."
            )

        assert target_property in self.targetPropertiesNames, (
            f"Target property '{target_property}' not found in dataset. "
            f"Available target properties: {self.targetPropertiesNames} "
            f"To convert a regression task to classification, first add the "
            f"property as a target property with the "
            f"`addTargetProperty` method."
        )
        self.restoreTargetProperty(target_property)
        target_values = self.getTarget(target_property).copy()
        target_spec = self.getTargetSpec(target_property)
        if target_values.isna().all():
            logger.debug(
                f"Target property '{target_property}' has all NaNs. This happens "
                "on the initialization of a PredictionDataSet, but should not happen "
                "otherwise."
            )
            assert target_spec.task.isClassification(), (
                f"Target property '{target_property}' is not a classification task. "
                " and it has no values."
            )
        else:
            # convert target values to discrete classes if needed
            if th is None:
                assert all(
                    value is None or (type(value) in (int, bool)) or
                    (isinstance(value, float) and value.is_integer())
                    for value in target_values
                ), (
                    "Precomputed classification target must be integers or booleans."
                    "Set the `th` argument to a list of threshold values to convert "
                    "float values to discrete classes for classification."
                )
            else:
                discretizer = Discretizer(target=target_property, th=th)
                target_values = discretizer.fitTransform(None, target_values)[1][target_property]
                self.addProperty(target_property, target_values)

            # update target specification
            n_classes = len(target_values.dropna().unique())
            task = TargetTasks.MULTICLASS if n_classes > 2 else TargetTasks.SINGLECLASS
            target_spec.task = task
            if th is None:
                target_spec.setTh(th, n_classes=n_classes)
            else:
                target_spec.setTh(th)
            logger.info(f"Target property '{target_property}' converted to classification.")
        
        
    def makeRegression(self, target_property: str):
        """Switch to regression task using the given target property.

        Args:
            target_property (str): name of the target property to use for regression
        """
        target_spec = self.getTargetSpec(target_property)
        self.restoreTargetProperty(target_spec)
        target_spec.task = TargetTasks.REGRESSION
        if hasattr(target_spec, "th"):
            del target_spec.th
        logger.info(f"Target property '{target_property}' converted to regression.")


    def checkClassification(
        self,
        target_property: str,
    ) -> bool:
        """Checks the validity of the target property for classification tasks.

        Args:
            target_property (str):
                Name of the target property to use for classification

        Returns:
            bool: `True` if the target property is correctly set up for classification,
            `False` otherwise.
        """
        target_values = self.getTarget(target_property)
        target_spec = self.getTargetSpec(target_property)

        if not all(
            value is None or np.isnan(value) or (type(value) in (int, bool)) or
            (isinstance(value, float) and value.is_integer())
            for value in target_values
        ):  
            logger.warning(
                f"Classification target property '{target_property}' "
                "should only contain integers or booleans. "
                "Either convert it to discrete values using "
                "`makeClassification` method with a threshold, "
                "change the property values using `addProperty`, "
                "or set the task to REGRESSION."
            )
            return False
        n_classes = len(target_values.dropna().unique())
        if n_classes == 1:
            logger.warning(
                f"Classification target property '{target_property}' task "
                f"is set to {target_spec.task}, but it contains only "
                "1 class. Perhaps you meant to set the task to REGRESSION? "
                "Training a classification model with only one class "
                "is not meaningful."
            )
            return False
        elif n_classes == 2 and target_spec.task == TargetTasks.MULTICLASS:
            logger.warning(
                f"Classification target property '{target_property}' task "
                "is set to MULTICLASS, but it contains only "
                f"2 classes. Perhaps you meant to set the task to "
                "SINGLECLASS?"
            )
            return False
        elif n_classes > 2 and target_spec.task == TargetTasks.SINGLECLASS:
            logger.warning(
                f"Classification target property '{target_property}' task "
                "is set to SINGLECLASS, but it contains more than "
                f"2 classes ({n_classes}). Perhaps you meant to set the task to "
                "MULTICLASS?"
            )
            return False
        return True

    @property
    def targetPropertiesNames(self) -> list[str]:
        """Get the names of the target properties."""
        return TargetSpec.getNames(self.targetProperties)

    @property
    def isMultiTask(self) -> bool:
        """Check if the dataset contains multiple target properties.

        Returns:
            (bool): `True` if the dataset contains multiple target properties
        """
        return len(self.targetProperties) > 1

    @property
    def nTargetProperties(self) -> int:
        """Get the number of target properties in the dataset."""
        return len(self.targetProperties)

    def getTargets(self) -> pd.DataFrame:
        """Get the target property values

        Returns:
            (pd.DataFrame): target property values
        """
        return self.getDF()[self.targetPropertiesNames]
    
    def getTarget(self, name: str | TargetSpec) -> pd.Series:
        """Get the target property values for the given target property.

        Args:
            name (str | TargetSpec): name or specification of the target property

        Returns:
            (pd.Series): target property values
        """
        if isinstance(name, TargetSpec):
            name = name.name
        assert name in self.targetPropertiesNames, f"Target property '{name}' not found in dataset."
        return self.getDF()[name]

    def getSubset(
        self,
        subset: list[str],
        ids: list[str] | None = None,
        name: str | None = None,
        path: str = ".",
        **kwargs,
    ) -> "QSPRTable":
        """Get a subset of the data set.

        Args:
            subset (list[str]): list of columns to include in the subset
            ids (list[str], optional): list of IDs to include in the subset. Defaults to
                `None`.
            name (str, optional): name of the subset. Defaults to `None`.
            path (str, optional): path to the directory where the subset will be saved.
                Defaults to ".".
            **kwargs: additional keyword arguments for the constructor of `QSPRTable`.

        Returns:
            QSPRTable: subset of the data set
        """
        # add target properties if not already in the subset
        # as the QSPRTable requires them
        subset = list(set(subset + self.targetPropertiesNames))
        mt = super().getSubset(subset, ids, name, path, **kwargs)
        ds = self.fromMolTable(
            mt, self.targetProperties, name=mt.name, path=path, drop_empty_target_props=False, **kwargs
        )
        return ds

    def addSplit(self, split: DataSplit, name: str):
        """Add a split to the dataset.

        Performs the split and stores the split object and the indices of the split.
        If the split has a random state, it will be set to the random state of the
        dataset if it is not set.

        Args:
            split (DataSplit): split to add
            name (str): name of the split
        """
        self.splits[name] = {
            "split": split,
            "ids": [(train_idx, test_idx) for train_idx, test_idx in self.split(split)],
        }

    def getSplit(self, name: str, as_type: str = "split"
        ) -> (DataSplit |list[tuple[pd.Index, pd.Index]]):
        """Get the split with the given name.

        Args:
            name (str): name of the split
        as_type (str): Determines the type of output. Can be one of:
            - "split": Returns a DataSplit object.
            - "ids": Returns train and test indices.

        Returns:
            DataSplit: split if `as_type` is "split"
            list[tuple[pd.Index, pd.Index]]:
                train and test indices if `as_type` is "ids"
        """
        split = self.splits[name]
        if as_type == "split":
            return split["split"]
        if as_type == "ids":
            return split["ids"]
        else:
            raise ValueError(
                f"Unknown as_type: {as_type}, "
                "should be 'split' or 'ids'."
            )

    def iterSplit(self, name: str, as_type: str = "ids"
        ) -> (
            Generator[tuple[pd.Index, pd.Index], None, None] |
            Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None] |
            Generator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None] |
            Generator[tuple["QSPRTable", "QSPRTable"], None, None]
        ):
        """Get the split with the given name.

        Args:
            name (str): name of the split
        as_type (str): Determines the type of output. Can be one of:
            - "ids": yields train and test indices.
            - "numpy": Yields train and test numpy arrays.
            - "pandas": Yields train and test pandas DataFrames.
            - "QSPRTable": Yields train and test QSPRTable objects.

        Yields:
            tuple[pd.Index, pd.Index]: train and test indices if `as_type` is "ids"
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                train descriptors, train targets, test descriptors, test targets
                `as_type` is "numpy"
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                train descriptors, train targets, test descriptors, test targets
                `as_type` is "pandas"
            tuple[QSPRTable, QSPRTable]:
                train and test QSPRTable objects if `as_type` is "QSPRTable"
        """
        split = self.splits[name]
        if as_type == "ids":
            for ids in split["ids"]:
                yield ids
        elif as_type == "numpy":
            X = self.getDescriptors()
            y = self.getTargets()
            for ids in split["ids"]:
                train_idx, test_idx = ids
                yield (
                    X.loc[train_idx].values,
                    y.loc[train_idx].values,
                    X.loc[test_idx].values,
                    y.loc[test_idx].values
                )
        elif as_type == "pandas":
            X = self.getDescriptors()
            y = self.getTargets()
            for ids in split["ids"]:
                train_idx, test_idx = ids
                yield (
                    X.loc[train_idx],
                    y.loc[train_idx],
                    X.loc[test_idx],
                    y.loc[test_idx]
                )
        elif as_type == "QSPRTable":
            for ids in split["ids"]:
                train = self.getSubset(self.getProperties(), ids[0])
                test = self.getSubset(self.getProperties(), ids[1])
                yield train, test
        else:
            raise ValueError(
                f"Unknown as_type: {as_type}, "
                "should be 'ids', 'numpy', 'pandas' or 'QSPRTable'."
            )

    def split(
        self,
        split: DataSplit,
    ) -> Generator[
        tuple[
            pd.Index,
            pd.Index
        ],
        None,
        None,
    ]:
        """Create folds from Descriptors and Targets. Can be used either for 
        cross-validation, bootstrapping or train-test split.

        Args:
            split (DataSplit): Split to apply to the data

        Yields:
            pd.Index, pd.Index: indices of the train and test set
        """
        if hasattr(split, "dataSet"):
            split.setDataSet(self)
        if hasattr(split, "randomState"):
            if split.randomState is None:
                split.randomState = self.randomState

        X = self.getDescriptors()
        y = self.getTargets()
        folds = split.split(X, y)

        for train_idx, test_idx in folds:
            # get QSPRTable indices from numerical index
            train_idx = X.index[train_idx]
            test_idx = X.index[test_idx]
            yield train_idx, test_idx

    def __getitem__(self, ids: list[str]) -> "QSPRTable":
        """Get a subset of the data set.

        This method is used to get a subset of the data set by providing a list of IDs.
        It is the same as calling `getSubset` method for all properties.
        It uses the same random state as the original data set.

        Args:
            ids (list[str]): list of IDs to include in the subset

        Returns:
            QSPRTable: subset of the data set
        """
        #FIXME: setting the random state here is not ideal, this should be done in the
        # getSubset method
        return self.getSubset(self.getProperties(), ids, random_state=self.randomState)

    def filter(self, table_filters: list[DataFilter]):
        """Filter the data set using the given filters.

        Args:
            table_filters (list[DataFilter]): list of filters to apply
        """
        for filter in table_filters:
            ret, _ = filter.transform(self.getDescriptors(), self.getTargets())
            ids = pd.Series(
                self.getProperty(self.idProp), index=self.getProperty(self.idProp)
            )
            ids_to_drop = ids[~ids.isin(ret.index)].values
            self.dropEntries(ids_to_drop)

    def __setstate__(self, state):
        super().__setstate__(state)
        for split in self.splits.values():
            if hasattr(split["split"], "setdataSet"):
                split["split"].setDataSet(self)

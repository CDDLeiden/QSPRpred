import json
import os
from copy import deepcopy
from typing import Callable, Generator

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .interfaces.qspr_data_set import QSPRDataSet
from ...logs import logger
from ...tasks import TargetProperty, TargetTasks
from ..storage.interfaces.chem_store import ChemStore
from .mol import MoleculeTable
from qsprpred.data.sampling.splits import DataSplit
import numpy as np


class QSPRTable(QSPRDataSet, MoleculeTable):
    """Implementation of `QSPRDataSet` using a collection of `PandasDataTable` objects.

    Attributes:
        targetProperties (str): property to be predicted with QSPRmodel
    """

    # _notJSON: ClassVar = [*MoleculeDataSet._notJSON]

    def __init__(
        self,
        storage: ChemStore | None = None,
        name: str | None = None,
        target_props: list[TargetProperty | dict] | None = None,
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
            target_props (list[TargetProperty | dict] | None):
                target properties, names should correspond with target columnname in df.
                If `None`, target properties will be inferred if this data set has been
                saved previously. Defaults to `None`.
            path (str, optional): path to the directory where the data set will be saved.
                Defaults to ".".
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
        # load target properties if not specified and file exists
        if target_props is None and os.path.exists(self.metaFile):
            meta = json.load(open(self.metaFile, "r"))
            target_props = meta["py/state"]["targetProperties"]
            target_props = [
                TargetProperty.fromJSON(json.dumps(x)) for x in target_props
            ]
        elif target_props is None:
            raise ValueError("Target properties must be specified for a new QSPRTable.")
        # populate feature matrix and target properties
        self._targetProperties = []
        self.setTargetProperties(target_props, drop_empty_target_props)
        logger.info(
            f"Dataset '{self.name}' created for "
            f"target Properties: '{self.targetProperties}'. "
            f"Number of samples: {len(self.storage)}. "
        )
        self.splits = {}

    @property
    def targetProperties(self) -> list[TargetProperty]:
        """Get the target properties of the dataset."""
        return self._targetProperties

    @targetProperties.setter
    def targetProperties(self, target_properties: list[TargetProperty]):
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
        target_props: list[TargetProperty | dict],
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
        target_props: list[TargetProperty | dict] | None = None,
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
        target_props: list[TargetProperty | dict],
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

    def addTargetProperty(self, prop: TargetProperty | dict, drop_empty: bool = True):
        """Add a target property to the dataset.

        Args:
            prop (TargetProperty | dict):
                target property to add or dictionary to initialize a TargetProperty
            drop_empty (bool):
                whether to drop rows with empty target property values. Defaults to
                `True`.
        """
        logger.debug(f"Adding target property '{prop}' to dataset.")
        prop = deepcopy(prop)
        if isinstance(prop, dict):
            prop = TargetProperty.fromDict(prop)
        if prop.name in self.targetPropertyNames:
            logger.warning(
                f"Property '{prop}' already exists in dataset. It will be reset."
            )
        assert (
            prop.name in self.getProperties()
        ), f"Property {prop} not found in data set."
        self._targetProperties.append(prop)
        self.restoreTargetProperty(prop)
        if prop.task.isClassification():
            self.makeClassification(prop.name, prop.th)
        if prop.imputer is not None:
            self.imputeProperties([prop.name], prop.imputer)
        if prop.transformer is not None:
            self.transformProperties([prop.name], prop.transformer)
        if drop_empty:
            self.dropEmptyEntries([prop.name])

    def getTargetProperties(self, names: list) -> list[TargetProperty]:
        """Get the target properties with the given names.

        Args:
            names (list[str]): name of the target properties

        Returns:
            (list[TargetProperty]): list of target properties
        """
        return [tp for tp in self.targetProperties if tp.name in names]

    def getTargetPropertiesNames(self) -> list[str]:
        """Get the names of the target properties.

        Returns:
            (list[str]): list of target property names
        """
        return [tp.name for tp in self.targetProperties]

    def setTargetProperties(
        self,
        target_props: list[TargetProperty | dict],
        drop_empty: bool = True,
    ):
        """Set list of target properties and apply transformations if specified.

        Args:
            target_props (list[TargetProperty]):
                list of target properties
            drop_empty (bool, optional):
                whether to drop rows with empty target property values. Defaults to
                `True`.
        """
        assert isinstance(target_props, list), (
            "target_props should be a list of TargetProperty objects or dictionaries "
            "initialize TargetProperties from. Not a %s." % type(target_props)
        )
        if isinstance(target_props[0], dict):
            assert all(isinstance(d, dict) for d in target_props), (
                "target_props should be a list of TargetProperty objects or "
                "dictionaries to initialize TargetProperties from, not a mix."
            )
            target_props = TargetProperty.fromList(target_props)
        else:
            assert all(isinstance(d, TargetProperty) for d in target_props), (
                "target_props should be a list of TargetProperty objects or "
                "dictionaries to initialize TargetProperties from, not a mix."
            )
        self._targetProperties = []
        for prop in target_props:
            self.addTargetProperty(prop, drop_empty)

    def unsetTargetProperty(self, name: str | TargetProperty):
        """Unset the target property. It will not remove it from the data set, but
        will make it unavailable for training.

        Args:
            name (str | TargetProperty):
                name of the target property to drop or the property itself
        """
        name = name.name if isinstance(name, TargetProperty) else name
        assert (
            name in self.targetPropertyNames
        ), f"Target property '{name}' not found in dataset."
        assert (
            len(self.targetProperties) > 1
        ), "Cannot drop task from single-task dataset."
        self._targetProperties = [tp for tp in self.targetProperties if tp.name != name]

    def restoreTargetProperty(self, prop: TargetProperty | str):
        """Reset target property to its original value.

        Args:
            prop (TargetProperty | str): target property to reset
        """
        if isinstance(prop, str):
            prop = self.getTargetProperties([prop])[0]
        if f"{prop.name}_original" in self.getProperties():
            self.addProperty(prop.name, self.getProperty(f"{prop.name}_original"))
        # save original values for next reset
        self.addProperty(f"{prop.name}_original", self.getProperty(prop.name))

    def makeRegression(self, target_property: str):
        """Switch to regression task using the given target property.

        Args:
            target_property (str): name of the target property to use for regression
        """
        target_property = self.getTargetProperties([target_property])[0]
        self.restoreTargetProperty(target_property)
        target_property.task = TargetTasks.REGRESSION
        if hasattr(target_property, "th"):
            del target_property.th
        logger.info("Target property converted to regression.")

    def makeClassification(
        self,
        target_property: str,
        th: list[float] | None = None,
    ):
        """Switch to classification task using the given threshold values.

        Args:
            target_property (str):
                Target property to use for classification
                or name of the target property.
            th (list[float], optional):
                list of threshold values. If not provided, the
                values will be inferred from th specified in TargetProperty.
                Defaults to None.
        """
        prop_name = target_property
        target_property = self.getTargetProperties([target_property])[0]
        self.restoreTargetProperty(target_property)
        # perform some checks
        if th is not None:
            assert (
                isinstance(th, list) or th == "precomputed"
            ), "Threshold values should be provided as a list of floats."
            if isinstance(th, list):
                assert (
                    len(th) > 0
                ), "Threshold values should be provided as a list of floats."
        if isinstance(target_property, str):
            target_property = self.getTargetProperties([target_property])[0]
        # check if the column only has nan values
        df = self.getDF()
        if df[target_property.name].isna().all():
            logger.warning(
                f"Target property {target_property.name}"
                " is all nan, cannot convert to classification."
            )
            return target_property
        # if no threshold values provided, use the ones specified in the TargetProperty
        if th is None:
            assert hasattr(target_property, "th"), (
                "Target property does not have a threshold attribute and "
                "no threshold specified in function args."
            )
            th = target_property.th
        if th == "precomputed":
            assert all(
                value is None or (type(value) in (int, bool)) or
                (isinstance(value, float) and value.is_integer())
                for value in df[prop_name]
            ), "Precomputed classification target must be integers or booleans."
            n_classes = len(df[prop_name].dropna().unique())
            target_property.task = (
                TargetTasks.MULTICLASS if n_classes > 2  # noqa: PLR2004
                else TargetTasks.SINGLECLASS
            )
            target_property.th = th
            target_property.nClasses = n_classes
        else:
            assert len(th) > 0, "Threshold list must contain at least one value."
            if len(th) > 1:
                assert len(th) > 3, (  # noqa: PLR2004
                    "For multi-class classification, "
                    "set more than 3 values as threshold."
                )
                # get max value, ignore nan
                assert max(df[prop_name].dropna()) <= max(th), (
                    "Make sure final threshold value is not smaller "
                    "than largest value of property"
                )
                assert min(df[prop_name].dropna()) >= min(th), (
                    "Make sure first threshold value is not larger "
                    "than smallest value of property"
                )
                self.addProperty(
                    f"{prop_name}_intervals",
                    pd.cut(df[prop_name], bins=th, include_lowest=True).astype(str),
                )
                encoded_intervals = LabelEncoder().fit_transform(
                    self.getProperty(f"{prop_name}_intervals")
                )
                self.addProperty(
                    prop_name,
                    np.where(df[prop_name].notna(), encoded_intervals, np.nan).astype(float),
                )
            else:
                binary_target = df[prop_name] > th[0]
                self.addProperty(prop_name, binary_target.where(df[prop_name].notna(), np.nan).astype(float))
            target_property.task = (
                TargetTasks.SINGLECLASS if len(th) == 1 else TargetTasks.MULTICLASS
            )
            target_property.th = th
        logger.info(f"Target property '{prop_name}' converted to classification.")

    @property
    def targetPropertyNames(self) -> list[str]:
        """Get the names of the target properties."""
        return TargetProperty.getNames(self.targetProperties)

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
        return self.getDF()[self.targetPropertyNames]

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
        subset = list(set(subset + self.targetPropertyNames))
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
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
    ) -> Generator[
        tuple[
            pd.Index,
            pd.Index
        ],
        None,
        None,
    ]:
        """Create folds from X and y. Can be used either for cross-validation,
        bootstrapping or train-test split.

        Args:
            split (DataSplit): Split to apply to the data
            X (pd.DataFrame): data to apply the split to
            y (pd.DataFrame | None): target data to apply the split to

        Yields:
            tuple[pd.Index, pd.Index]: indices of the train and test set
        """
        if hasattr(split, "dataSet"):
            split.setDataSet(self)
        if hasattr(split, "randomState"):
            if split.randomState is None:
                split.randomState = self.randomState

        X = self.getDescriptors() if X is None else X
        y = self.getTargets() if y is None else y
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

    def filter(self, table_filters: list[Callable]):
        """Filter the data set using the given filters.

        Args:
            table_filters (list[Callable]): list of filters to apply
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
        for name, split in self.splits.items():
            if hasattr(split["split"], "setdataSet"):
                split["split"].setDataSet(self)

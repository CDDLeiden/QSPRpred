from typing import ClassVar, Optional, Callable, Generator

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .mol import MoleculeTable
from ...data.processing.data_filters import RepeatsFilter
from ...data.processing.feature_standardizers import (
    SKLearnStandardizer,
    apply_feature_standardizer,
)
from ...data.sampling.folds import FoldsFromDataSplit
from ...logs import logger
from ...tasks import TargetProperty
from ...tasks import TargetTasks


class QSPRDataset(MoleculeTable):
    """Prepare dataset for QSPR model training.

    It splits the data in train and test set, as well as creating cross-validation
    folds. Optionally low quality data is filtered out. For classification the dataset
    samples are labelled as active/inactive.

    Attributes:
        targetProperties (str) : property to be predicted with QSPRmodel
        df (pd.dataframe) : dataset
        X (np.ndarray/pd.DataFrame) : m x n feature matrix for cross validation, where m
            is the number of samplesand n is the number of features.
        y (np.ndarray/pd.DataFrame) : m-d label array for cross validation, where m is
            the number of samples and equals to row of X.
        X_ind (np.ndarray/pd.DataFrame) : m x n Feature matrix for independent set,
            where m is the number of samples and n is the number of features.
        y_ind (np.ndarray/pd.DataFrame) : m-l label array for independent set, where m
            is the number of samples and equals to row of X_ind, and l is the number of
            types.
        featureNames (list of str) : feature names
    """

    _notJSON: ClassVar = [*MoleculeTable._notJSON, "X", "X_ind", "y", "y_ind"]

    def __init__(
        self,
        name: str,
        target_props: list[TargetProperty | dict],
        df: Optional[pd.DataFrame] = None,
        smiles_col: str = "SMILES",
        add_rdkit: bool = False,
        store_dir: str = ".",
        overwrite: bool = False,
        n_jobs: int = 1,
        chunk_size: int = 50,
        drop_invalids: bool = True,
        drop_empty: bool = True,
        index_cols: Optional[list[str]] = None,
        autoindex_name: str = "QSPRID",
        random_state: int | None = None,
    ):
        """Construct QSPRdata, also apply transformations of output property if
        specified.

        Args:
            name (str): data name, used in saving the data
            target_props (list[TargetProperty | dict]): target properties, names
                should correspond with target columnname in df
            df (pd.DataFrame, optional): input dataframe containing smiles and target
                property. Defaults to None.
            smiles_col (str, optional): name of column in df containing SMILES.
                Defaults to "SMILES".
            add_rdkit (bool, optional): if true, column with rdkit molecules will be
                added to df. Defaults to False.
            store_dir (str, optional): directory for saving the output data.
                Defaults to '.'.
            overwrite (bool, optional): if already saved data at output dir if should
                be overwritten. Defaults to False.
            n_jobs (int, optional): number of parallel jobs. If <= 0, all available
                cores will be used. Defaults to 1.
            chunk_size (int, optional): chunk size for parallel processing.
                Defaults to 50.
            drop_invalids (bool, optional): if true, invalid SMILES will be dropped.
                Defaults to True.
            drop_empty (bool, optional): if true, rows with empty target property will
be removed.
            index_cols (list[str], optional): columns to be used as index in the
                dataframe. Defaults to `None` in which case a custom ID will be
                generated.
            autoindex_name (str): Column name to use for automatically generated IDs.
            random_state (int, optional): random state for splitting the data.

        Raises:
            `ValueError`: Raised if threshold given with non-classification task.
        """
        super().__init__(
            name,
            df,
            smiles_col,
            add_rdkit,
            store_dir,
            overwrite,
            n_jobs,
            chunk_size,
            False,
            index_cols,
            autoindex_name,
            random_state,
        )
        # load names of descriptors to use as training features
        self.featureNames = self.getFeatureNames()
        # load target properties
        self.setTargetProperties(target_props, drop_empty)
        self.feature_standardizer = None
        # populate feature matrix and target property array
        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None
        if drop_invalids:
            self.dropInvalids()
        self.restoreTrainingData()
        logger.info(
            f"Dataset '{self.name}' created for target "
            f"targetProperties: '{self.targetProperties}'."
        )

    def __setstate__(self, state):
        super().__setstate__(state)
        self.restoreTrainingData()

    @staticmethod
    def fromTableFile(name: str, filename: str, sep: str = "\t", *args, **kwargs):
        r"""Create QSPRDataset from table file (i.e. CSV or TSV).

        Args:
            name (str): name of the data set
            filename (str): path to the table file
            sep (str, optional): separator in the table file. Defaults to "\t".
            *args: additional arguments for QSPRDataset constructor
            **kwargs: additional keyword arguments for QSPRDataset constructor
        Returns:
            QSPRDataset: `QSPRDataset` object
        """
        return QSPRDataset(
            name,
            df=pd.read_table(filename, sep=sep),
            *args,  # noqa: B026 # FIXME: this is a bug in flake8...
            **kwargs,
        )

    @staticmethod
    def fromSDF(name: str, filename: str, smiles_prop: str, *args, **kwargs):
        """Create QSPRDataset from SDF file.

        It is currently not implemented for QSPRDataset, but you can convert from
        'MoleculeTable' with the 'fromMolTable' method.

        Args:
            name (str): name of the data set
            filename (str): path to the SDF file
            smiles_prop (str): name of the property in the SDF file containing SMILES
            *args: additional arguments for QSPRDataset constructor
            **kwargs: additional keyword arguments for QSPRDataset constructor
        """
        raise NotImplementedError(
            f"SDF loading not implemented for {QSPRDataset.__name__}, yet. You can "
            "convert from 'MoleculeTable' with 'fromMolTable'."
        )

    def setTargetProperties(
        self,
        target_props: list[TargetProperty],
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
        # check target properties validity
        assert isinstance(target_props, list), (
            "target_props should be a list of TargetProperty objects or dictionaries "
            "initialize TargetProperties from."
        )
        if isinstance(target_props[0], dict):
            assert all(isinstance(d, dict) for d in target_props), (
                "target_props should be a list of TargetProperty objects or "
                "dictionaries to initialize TargetProperties from, not a mix."
            )
            self.targetProperties = TargetProperty.fromList(target_props)
        else:
            assert all(isinstance(d, TargetProperty) for d in target_props), (
                "target_props should be a list of TargetProperty objects or "
                "dictionaries to initialize TargetProperties from, not a mix."
            )
            self.targetProperties = target_props
        assert all(
            prop in self.df.columns for prop in self.targetPropertyNames
        ), "Not all target properties in dataframe columns."
        # transform target properties
        for target_prop in self.targetProperties:
            if target_prop.transformer is not None:
                transformed_prop = f"{target_prop.name}_transformed"
                self.transform(
                    [target_prop.name],
                    target_prop.transformer,
                    addAs=[transformed_prop],
                )
                target_prop.name = transformed_prop
            if target_prop.imputer is not None:
                self.imputeProperties([target_prop.name], target_prop.imputer)
        # drop rows with missing smiles/no target property for any of
        # the target properties
        if drop_empty:
            self.dropEmptySmiles()
            self.dropEmptyProperties(self.targetPropertyNames)
        # convert classification targets to integers
        for target_prop in self.targetProperties:
            if target_prop.task.isClassification():
                self.makeClassification(target_prop)

    @property
    def hasFeatures(self):
        """Check whether the currently selected set of features is not empty."""
        return True if (self.featureNames and len(self.featureNames) > 0) else False

    def getFeatureNames(self) -> list[str]:
        """Get current feature names for this data set.

        Returns:
            list[str]: list of feature names
        """
        features = None if not self.hasDescriptors else self.getDescriptorNames()
        if self.descriptorCalculators:
            features = []
            for calc in self.descriptorCalculators:
                prefix = calc.getPrefix()
                for descset in calc.descSets:
                    features.extend(
                        [f"{prefix}_{descset}_{x}" for x in descset.descriptors]
                    )
        return features

    def restoreTrainingData(self):
        """Restore training data from the data frame.

        If the data frame contains a column 'Split_IsTrain',
        the data will be split into training and independent sets. Otherwise, the
            independent set will
        be empty. If descriptors are available, the resulting training matrices will
            be featurized.
        """
        # split data into training and independent sets if saved previously
        if "Split_IsTrain" in self.df.columns:
            self.y = self.df.query("Split_IsTrain")[self.targetPropertyNames]
            self.y_ind = self.df.loc[
                ~self.df.index.isin(self.y.index), self.targetPropertyNames
            ]
        else:
            self.y = self.df[self.targetPropertyNames]
            self.y_ind = self.df.loc[
                ~self.df.index.isin(self.y.index), self.targetPropertyNames
            ]
        self.X = self.y.drop(self.y.columns, axis=1)
        self.X_ind = self.y_ind.drop(self.y_ind.columns, axis=1)
        self.featurizeSplits(shuffle=False)

    def makeRegression(self, target_property: TargetProperty | str):
        """Switch to regression task using the given target property.

        Args:
            target_property (str): name of the target property to use for regression
        """
        if isinstance(target_property, str):
            target_property = self.getTargetProperties(
                [target_property], original_names=True
            )[0]
        target_property.name = target_property.originalName
        target_property.task = TargetTasks.REGRESSION
        del target_property.th
        self.restoreTrainingData()

    def makeClassification(
        self, target_property: TargetProperty | str, th: Optional[list[float]] = None
    ):
        """Switch to classification task using the given threshold values.

        Args:
            target_property (TargetProperty): Target property to use for classification
                or name of the target property.
            th (list[float], optional): list of threshold values. If not provided, the
                values will be inferred from th specified in TargetProperty.
                Defaults to None.
        """
        if th is not None:
            assert (
                isinstance(th, list) or th == "precomputed"
            ), "Threshold values should be provided as a list of floats."
            if isinstance(th, list):
                assert (
                    len(th) > 0
                ), "Threshold values should be provided as a list of floats."

        if isinstance(target_property, str):
            target_property = self.getTargetProperties(
                [target_property], original_names=True
            )[0]

        # check if the column only has nan values
        if self.df[target_property.name].isna().all():
            logger.debug(
                f"Target property {target_property.name}"
                " is all nan, assuming predictor."
            )
            return target_property

        # if no threshold values provided, use the ones specified in the TargetProperty
        if th is None:
            assert hasattr(target_property, "th"), (
                "Target property does not have a threshold attribute and "
                "no threshold specified in function args."
            )
            th = target_property.th

        new_prop = f"{target_property.originalName}_class"

        if th == "precomputed":
            self.df[new_prop] = self.df[target_property.originalName]
            assert all(
                value is None
                or (type(value) in (int, bool))
                or (isinstance(value, float) and value.is_integer())
                for value in self.df[new_prop]
            ), "Precomputed classification target must be integers or booleans."
            nClasses = len(self.df[new_prop].dropna().unique())
            target_property.task = (
                TargetTasks.MULTICLASS
                if nClasses > 2  # noqa: PLR2004
                else TargetTasks.SINGLECLASS
            )
            target_property.th = th
            target_property.nClasses = nClasses
            target_property.name = new_prop
        else:
            assert len(th) > 0, "Threshold list must contain at least one value."
            if len(th) > 1:
                assert len(th) > 3, (  # noqa: PLR2004c
                    "For multi-class classification, "
                    "set more than 3 values as threshold."
                )
                assert max(self.df[target_property.originalName]) <= max(th), (
                    "Make sure final threshold value is not smaller "
                    "than largest value of property"
                )
                assert min(self.df[target_property.originalName]) >= min(th), (
                    "Make sure first threshold value is not larger "
                    "than smallest value of property"
                )
                self.df[f"{new_prop}_intervals"] = pd.cut(
                    self.df[target_property.originalName], bins=th, include_lowest=True
                ).astype(str)
                self.df[new_prop] = LabelEncoder().fit_transform(
                    self.df[f"{new_prop}_intervals"]
                )
            else:
                self.df[new_prop] = self.df[target_property.originalName] > th[0]
            target_property.task = (
                TargetTasks.SINGLECLASS if len(th) == 1 else TargetTasks.MULTICLASS
            )
            target_property.th = th
            target_property.name = new_prop
        self.restoreTrainingData()
        logger.info("Target property converted to classification.")
        return target_property

    def searchWithIndex(
        self, index: pd.Index, name: str | None = None
    ) -> "MoleculeTable":
        ret = super().searchWithIndex(index, name)
        return QSPRDataset.fromMolTable(ret, self.targetProperties, name=ret.name)

    @staticmethod
    def fromMolTable(
        mol_table: MoleculeTable,
        target_props: list[TargetProperty | dict],
        name=None,
        **kwargs,
    ) -> "QSPRDataset":
        """Create QSPRDataset from a MoleculeTable.

        Args:
            mol_table (MoleculeTable): MoleculeTable to use as the data source
            target_props (list): list of target properties to use
            name (str, optional): name of the data set. Defaults to None.
            kwargs: additional keyword arguments to pass to the constructor

        Returns:
            QSPRDataset: created data set
        """
        name = mol_table.name if name is None else name
        kwargs["store_dir"] = (
            mol_table.baseDir if "store_dir" not in kwargs else kwargs["store_dir"]
        )
        kwargs["random_state"] = (
            mol_table.randomState
            if "random_state" not in kwargs
            else kwargs["random_state"]
        )
        kwargs["n_jobs"] = (
            mol_table.nJobs if "n_jobs" not in kwargs else kwargs["n_jobs"]
        )
        kwargs["chunk_size"] = (
            mol_table.chunkSize if "chunk_size" not in kwargs else kwargs["chunk_size"]
        )
        kwargs["smiles_col"] = (
            mol_table.smilesCol if "smiles_col" not in kwargs else kwargs["smiles_col"]
        )
        kwargs["index_cols"] = (
            mol_table.indexCols if "index_cols" not in kwargs else kwargs["index_cols"]
        )
        ds = QSPRDataset(
            name,
            target_props,
            mol_table.getDF(),
            **kwargs,
        )
        ds.descriptors = mol_table.descriptors
        return ds

    def addCustomDescriptors(
        self,
        calculator: "CustomDescriptorsCalculator",  # noqa: F821
        recalculate: bool = False,
        featurize: bool = True,
        **kwargs,
    ):
        """Add custom descriptors to the data set.

        If descriptors are already present, they will be recalculated if `recalculate`
        is `True`.

        Args:
            calculator (CustomDescriptorsCalculator): calculator instance to use for
                descriptor calculation
            recalculate (bool, optional): whether to recalculate descriptors if they
                are already present. Defaults to `False`.
            featurize (bool, optional): whether to featurize the data set splits
                after adding descriptors. Defaults to `True`.
            kwargs: additional keyword arguments to pass to the calculator
        """
        super().addCustomDescriptors(calculator, recalculate, **kwargs)
        self.featurize(update_splits=featurize)

    def filter(self, table_filters: list[Callable]):
        """Filter the data set using the given filters.

        Args:
            table_filters (list[Callable]): list of filters to apply
        """
        super().filter(table_filters)
        self.restoreTrainingData()
        self.featurize()

    def addDescriptors(
        self,
        calculator: "MoleculeDescriptorsCalculator",  # noqa: F821
        recalculate: bool = False,
        featurize: bool = True,
    ):
        """Add descriptors to the data set.

        If descriptors are already present, they will be recalculated if `recalculate`
        is `True`. Featurization will be performed after adding descriptors if
        `featurize` is `True`. Featurazation converts current data matrices to pure
        numeric matrices of selected descriptors (features).

        Args:
            calculator (MoleculeDescriptorsCalculator): calculator instance to use for
                descriptor calculation
            recalculate (bool, optional): whether to recalculate descriptors if they are
                already present. Defaults to `False`.
            featurize (bool, optional): whether to featurize the data set splits after
                adding descriptors. Defaults to `True`.
        """
        super().addDescriptors(calculator, recalculate)
        self.featurize(update_splits=featurize)

    def featurize(self, update_splits=True):
        self.featureNames = self.getFeatureNames()
        if update_splits:
            self.featurizeSplits(shuffle=False)

    def saveSplit(self):
        """Save split data to the managed data frame."""
        if self.X is not None:
            self.df["Split_IsTrain"] = self.df.index.isin(self.X.index)
        else:
            logger.debug("No split data available. Skipping split data save.")

    def save(self, save_split: bool = True):
        """Save the data set to file and serialize metadata.

        Args:
            save_split (bool): whether to save split data to the managed data frame.
        """
        if save_split:
            self.saveSplit()
        super().save()

    def split(self, split: "DataSplit", featurize: bool = False):
        """Split dataset into train and test set.

        You can either split tha data frame itself or you can set `featurize` to `True`
        if you want to use feature matrices instead of the raw data frame.

        Args:
            split (DataSplit) :
                split instance orchestrating the split
            featurize (bool):
                whether to featurize the data set splits after splitting.
                Defaults to `False`.

        """
        if (
            hasattr(split, "hasDataSet")
            and hasattr(split, "setDataSet")
            and not split.hasDataSet
        ):
            split.setDataSet(self)
        if hasattr(self.split, "setSeed") and hasattr(self.split, "getSeed"):
            if self.split.getSeed() is None:
                self.split.setSeed(self.randomState)
        # split the data into train and test
        folds = FoldsFromDataSplit(split)
        self.X, self.X_ind, self.y, self.y_ind, _, _ = next(
            folds.iterFolds(self, concat=True)
        )
        # select target properties
        logger.info("Total: train: %s test: %s" % (len(self.y), len(self.y_ind)))
        for prop in self.targetProperties:
            logger.info("Target property: %s" % prop.name)
            if prop.task == TargetTasks.SINGLECLASS:
                logger.info(
                    "    In train: active: %s not active: %s"
                    % (
                        sum(self.y[prop.name]),
                        len(self.y[prop.name]) - sum(self.y[prop.name]),
                    )
                )
                logger.info(
                    "    In test:  active: %s not active: %s\n"
                    % (
                        sum(self.y_ind[prop.name]),
                        len(self.y_ind[prop.name]) - sum(self.y_ind[prop.name]),
                    )
                )
            if prop.task == TargetTasks.MULTICLASS:
                logger.info("train: %s" % self.y[prop.name].value_counts())
                logger.info("test: %s\n" % self.y_ind[prop.name].value_counts())
                try:
                    assert np.all([x > 0 for x in self.y[prop.name].value_counts()])
                    assert np.all([x > 0 for x in self.y_ind[prop.name].value_counts()])
                except AssertionError as err:
                    logger.exception(
                        "All bins in multi-class classification "
                        "should contain at least one sample"
                    )
                    raise err

                if self.y[prop.name].dtype.name == "category":
                    self.y[prop.name] = self.y[prop.name].cat.codes
                    self.y_ind[prop.name] = self.y_ind[prop.name].cat.codes
        # convert splits to features if required
        if featurize:
            self.featurizeSplits(shuffle=False)

    def loadDescriptorsToSplits(
        self, shuffle: bool = True, random_state: Optional[int] = None
    ):
        """Load all available descriptors into the train and test splits.

        If no descriptors are available, an exception will be raised.

        args:
            shuffle (bool): whether to shuffle the training and test sets
            random_state (int): random state for shuffling

        Raises:
            ValueError: if no descriptors are available
        """
        if not self.hasDescriptors:
            raise ValueError(
                "No descriptors available. Cannot load descriptors to splits."
            )
        descriptors = self.getDescriptors()
        if self.X_ind is not None and self.y_ind is not None:
            self.X = descriptors.loc[self.X.index, :]
            self.y = self.df.loc[self.X.index, self.targetPropertyNames]
            self.X_ind = descriptors.loc[self.X_ind.index, :]
            self.y_ind = self.df.loc[self.y_ind.index, self.targetPropertyNames]
        else:
            self.X = descriptors
            self.featureNames = self.getDescriptorNames()
            self.y = self.df.loc[descriptors.index, self.targetPropertyNames]
            self.X_ind = descriptors.loc[~self.X.index.isin(self.X.index), :]
            self.y_ind = self.df.loc[self.X_ind.index, self.targetPropertyNames]
        if shuffle:
            self.shuffle(random_state)
        # make sure no extra data is present in the splits
        mask_train = self.X.index.isin(self.df.index)
        mask_test = self.X_ind.index.isin(self.df.index)
        if mask_train.sum() != len(self.X):
            logger.warning(
                "Some items will be removed from the training set because "
                f"they no longer exist in the data set: {self.X.index[~mask_train]}"
            )
        if mask_test.sum() != len(self.X_ind):
            logger.warning(
                "Some items will be removed from the test set because "
                f"they no longer exist in the data set: {self.X_ind.index[~mask_test]}"
            )
        self.X = self.X.loc[mask_train, :]
        self.X_ind = self.X_ind.loc[mask_test, :]
        self.y = self.y.loc[self.X.index, :]
        self.y_ind = self.y_ind.loc[self.X_ind.index, :]

    def shuffle(self, random_state: Optional[int] = None):
        self.X = self.X.sample(frac=1, random_state=random_state or self.randomState)
        self.X_ind = self.X_ind.sample(
            frac=1, random_state=random_state or self.randomState
        )
        self.y = self.y.loc[self.X.index, :]
        self.y_ind = self.y_ind.loc[self.X_ind.index, :]
        # self.df = self.df.loc[self.X.index, :]

    def featurizeSplits(self, shuffle: bool = True, random_state: Optional[int] = None):
        """If the data set has descriptors, load them into the train and test splits.

        If no descriptors are available, remove all features from
        the splits They will become zero length along the feature axis (columns), but
        will retain their original length along the sample axis (rows). This is useful
        for the case where the data set has no descriptors, but the user wants to retain
        train and test splits.

        shuffle (bool): whether to shuffle the training and test sets
        random_state (int): random state for shuffling
        """
        if self.hasDescriptors and self.hasFeatures:
            self.loadDescriptorsToSplits(
                shuffle=shuffle, random_state=random_state or self.randomState
            )
            self.X = self.X.loc[:, self.featureNames]
            self.X_ind = self.X_ind.loc[:, self.featureNames]
        else:
            if self.X is not None and self.X_ind is not None:
                self.X = self.X.loc[self.X.index, :]
                self.X_ind = self.X_ind.loc[self.X_ind.index, :]
            else:
                self.X = self.df.loc[self.df.index, :]
                self.X_ind = self.df.loc[~self.df.index.isin(self.X.index), :]
            self.X = self.X.drop(self.X.columns, axis=1)
            self.X_ind = self.X_ind.drop(self.X_ind.columns, axis=1)
            if shuffle:
                self.shuffle(random_state or self.randomState)
        # make sure no extra data is present in the splits
        mask_train = self.X.index.isin(self.df.index)
        mask_test = self.X_ind.index.isin(self.df.index)
        if mask_train.sum() != len(self.X):
            logger.warning(
                "Some items will be removed from the training set because "
                f"they no longer exist in the data set: {self.X.index[~mask_train]}"
            )
        if mask_test.sum() != len(self.X_ind):
            logger.warning(
                "Some items will be removed from the test set because "
                f"they no longer exist in the data set: {self.X_ind.index[~mask_test]}"
            )
        self.X = self.X.loc[mask_train, :]
        self.X_ind = self.X_ind.loc[mask_test, :]

    def fillMissing(self, fill_value: float, columns: Optional[list[str]] = None):
        """Fill missing values in the data set with a given value.

        Args:
            fill_value (float): value to fill missing values with
            columns (list[str], optional): columns to fill missing values in.
                Defaults to None.
        """
        filled = False
        for desc in self.descriptors:
            desc.fillMissing(fill_value, columns)
            filled = True
        if not filled:
            logger.warning("Missing values filled with %s" % fill_value)
        else:
            self.featurize()

    def filterFeatures(self, feature_filters: list[Callable]):
        """Filter features in the data set.

        Args:
            feature_filters (list[Callable]): list of feature filter functions that take
                X feature matrix and y target vector as arguments
        """
        if not self.hasFeatures:
            raise ValueError("No features to filter")
        if self.X.shape[1] == 1:
            logger.warning("Only one feature present. Skipping feature filtering.")
            return
        else:
            for featurefilter in feature_filters:
                self.X = featurefilter(self.X, self.y)
            # update features
            self.featureNames = self.X.columns.to_list()
            if self.X_ind is not None:
                self.X_ind = self.X_ind[self.featureNames]
            logger.info(f"Selected features: {self.featureNames}")
            # update descriptor calculator
            for calc in self.descriptorCalculators:
                prefix = calc.getPrefix()
                calc.keepDescriptors(
                    [x for x in self.featureNames if x.startswith(prefix)]
                )

    def setFeatureStandardizer(self, feature_standardizer):
        """Set feature standardizer.

        Args:
            feature_standardizer (SKLearnStandardizer | BaseEstimator): feature
                standardizer
        """
        if not hasattr(feature_standardizer, "toFile"):
            feature_standardizer = SKLearnStandardizer(feature_standardizer)
        self.feature_standardizer = feature_standardizer

    def addFeatures(
        self,
        feature_calculators: list["DescriptorsCalculator"] | None = None,  # noqa: F821
        recalculate: bool = False,
    ):
        """Add features to the data set.

        Args:
            feature_calculators (List[DescriptorsCalculator], optional): list of
                feature calculators to add. Defaults to None.
            recalculate (bool): if True, recalculate features even if they are already
                present in the data set. Defaults to False.
        """
        if feature_calculators is not None:
            for calc in feature_calculators:
                # we avoid isinstance() here to avoid circular imports
                if calc.__class__.__name__ == "MoleculeDescriptorsCalculator":
                    self.addDescriptors(calc, recalculate=recalculate, featurize=False)
                else:
                    raise ValueError("Unknown feature calculator type: %s" % type(calc))
            self.featurize()

    def dropInvalids(self):
        ret = super().dropInvalids()
        self.featurize()
        return ret

    def reset(self):
        """Reset the data set. Splits will be removed and all descriptors will be
        moved to the training data. Feature standardization and molecule
        standardization and molecule filtering are not affected.
        """
        if self.featureNames is not None:
            self.featureNames = self.getDescriptorNames()
            self.X = None
            self.X_ind = None
            self.y = None
            self.y_ind = None
            self.loadDescriptorsToSplits(shuffle=False)

    def prepareDataset(
        self,
        smiles_standardizer: str | Callable | None = "chembl",
        datafilters: list = [RepeatsFilter(keep=True)],
        split=None,
        feature_calculators: list | None = None,
        feature_filters: list | None = None,
        feature_standardizer: Optional[SKLearnStandardizer] = None,
        feature_fill_value: float = np.nan,
        recalculate_features: bool = False,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """Prepare the dataset for use in QSPR model.

        Arguments:
            smiles_standardizer (str | Callable): either `chembl`, `old`, or a
                partial function that reads and standardizes smiles. If `None`, no
                standardization will be performed. Defaults to `chembl`.
            datafilters (list of datafilter obj): filters number of rows from dataset
            split (datasplitter obj): splits the dataset into train and test set
            feature_calculators (list[DescriptorsCalculator]): calculate features using
                different information from the data set
            feature_filters (list of feature filter objs): filters features
            feature_standardizer (SKLearnStandardizer or sklearn.base.BaseEstimator):
                standardizes and/or scales features
            recalculate_features (bool): recalculate features even if they are already
                present in the file
            feature_fill_value (float): value to fill missing values with.
                Defaults to `numpy.nan`
            shuffle (bool): whether to shuffle the created training and test sets
            random_state (int): random state for shuffling
        """
        # reset everything
        self.reset()
        # apply sanitization and standardization
        if smiles_standardizer is not None:
            self.standardizeSmiles(smiles_standardizer)
        # calculate features
        if feature_calculators is not None:
            self.addFeatures(feature_calculators, recalculate=recalculate_features)
        # apply data filters
        if datafilters is not None:
            self.filter(datafilters)
        # Replace any NaN values in featureNames by 0
        # FIXME: this is not very good, we should probably add option to do custom
        # data imputation here or drop rows with NaNs
        if feature_fill_value is not None:
            self.fillMissing(feature_fill_value)
        # shuffle the data
        if shuffle:
            self.shuffle(random_state or self.randomState)
        # split dataset
        if split is not None:
            self.split(split)
        # apply feature filters on training set
        if feature_filters and self.hasDescriptors:
            self.filterFeatures(feature_filters)
        elif not self.hasDescriptors:
            logger.warning("No descriptors present, feature filters will be skipped.")
        # set feature standardizers
        if feature_standardizer:
            self.setFeatureStandardizer(feature_standardizer)

    def checkFeatures(self):
        """Check consistency of features and descriptors."""
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "X and y have different number of rows: "
                f"{self.X.shape[0]} != {self.y.shape[0]}"
            )
        elif self.X.shape[0] == 0:
            raise ValueError("X has no rows.")

    def fitFeatureStandardizer(self):
        """Fit the feature standardizers on the training set.

        Returns:
            X (pd.DataFrame): standardized training set
        """
        if self.hasDescriptors:
            X = self.getDescriptors()
            if self.featureNames is not None:
                X = X[self.featureNames]
            return apply_feature_standardizer(self.feature_standardizer, X, fit=True)[0]

    def getFeatures(
        self,
        inplace: bool = False,
        concat: bool = False,
        raw: bool = False,
        ordered: bool = False,
    ):
        """Get the current feature sets (training and test) from the dataset.

        This method also applies any feature standardizers that have been set on the
        dataset during preparation.

        Args:
            inplace (bool): If `True`, the created feature matrices will be saved to the
                dataset object itself as 'X' and 'X_ind' attributes. Note that this will
                overwrite any existing feature matrices and if the data preparation
                workflow changes, these are not kept up to date. Therefore, it is
                recommended to generate new feature sets after any data set changes.
            concat (bool): If `True`, the training and test feature matrices will be
                concatenated into a single matrix. This is useful for training models
                that do not require separate training and test sets (i.e. the final
                optimized models).
            raw (bool): If `True`, the raw feature matrices will be returned without
                any standardization applied.
            ordered (bool):
                If `True`, the returned feature matrices will be ordered
                according to the original order of the data set. This is only relevant
                if `concat` is `True`.
        """
        self.checkFeatures()

        if concat:
            if len(self.X.columns) != 0:
                df_X = pd.concat(
                    [self.X[self.featureNames], self.X_ind[self.featureNames]], axis=0
                )
                df_X_ind = None
            else:
                df_X = pd.concat([self.X, self.X_ind], axis=0)
                df_X_ind = None
        elif len(self.X.columns) != 0:
            df_X = self.X[self.featureNames]
            df_X_ind = self.X_ind[self.featureNames]
        else:
            df_X = self.X
            df_X_ind = self.X_ind

        X = df_X.values
        X_ind = df_X_ind.values if df_X_ind is not None else None
        if not raw and self.feature_standardizer:
            X, self.feature_standardizer = apply_feature_standardizer(
                self.feature_standardizer, df_X, fit=True
            )
            if X_ind is not None and X_ind.shape[0] > 0:
                X_ind, _ = apply_feature_standardizer(
                    self.feature_standardizer, df_X_ind, fit=False
                )

        X = pd.DataFrame(X, index=df_X.index, columns=df_X.columns)
        if X_ind is not None:
            X_ind = pd.DataFrame(X_ind, index=df_X_ind.index, columns=df_X_ind.columns)

        if inplace:
            self.X = X
            self.X_ind = X_ind

        if ordered and concat:
            X = X.loc[self.df.index, :]
        return (X, X_ind) if not concat else X

    def getTargetPropertiesValues(self, concat: bool = False, ordered: bool = False):
        """Get the response values (training and test) for the set target property.

        Args:
            concat (bool): if `True`, return concatenated training and validation set
                target properties
            ordered (bool): if `True`, return the target properties in the original
                order of the data set. This is only relevant if `concat` is `True`.
        Returns:
            `tuple` of (train_responses, test_responses) or `pandas.DataFrame` of all
            target property values
        """
        if concat:
            ret = pd.concat(
                [self.y, self.y_ind] if self.y_ind is not None else [self.y]
            )
            return ret.loc[self.df.index, :] if ordered else ret
        else:
            return self.y, self.y_ind if self.y_ind is not None else self.y

    def getTargetProperties(self, names: list, original_names: bool = False):
        """Get the target properties with the given names.

        Args:
            names (list[str]): name of the target properties
            original_names (bool): if `True`, use the original names of the target
                properties

        Returns:
            `TargetProperty`: target property with the given name
        """
        return TargetProperty.selectFromList(
            self.targetProperties, names, original_names=original_names
        )

    @property
    def targetPropertyNames(self):
        """Get the names of the target properties."""
        return TargetProperty.getNames(self.targetProperties)

    @property
    def targetPropertyOriginalNames(self):
        """Get the original names of the target properties."""
        return TargetProperty.getOriginalNames(self.targetProperties)

    @property
    def isMultiTask(self):
        """Check if the dataset contains multiple target properties.

        Returns:
            `bool`: `True` if the dataset contains multiple target properties
        """
        return len(self.targetProperties) > 1

    @property
    def nTasks(self):
        """Get the number of target properties in the dataset."""
        return len(self.targetProperties)

    def dropTask(self, task):
        """Drop the given task from the dataset.

        Args:
            task (str): name of the task to drop
        """
        assert task in self.targetPropertyNames, f"Task {task} not found in dataset."
        assert (
            len(self.targetProperties) > 1
        ), "Cannot drop task from single-task dataset."
        self.targetProperties = [tp for tp in self.targetProperties if tp.name != task]
        self.restoreTrainingData()

    def addTask(self, task: TargetProperty | dict):
        """Add a task to the dataset.

        Args:
            task (TargetProperty): name of the task to add
        """
        if isinstance(task, dict):
            task = TargetProperty.fromDict(task)

        assert (
            task.name not in self.targetPropertyNames
        ), f"Task {task} already exists in dataset."
        assert task.name in self.df.columns, f"Task {task} not found in dataset."

        self.targetProperties.append(task)
        self.restoreTrainingData()

    def iterFolds(
        self,
        split: "DataSplit",
        concat: bool = False,
    ) -> Generator[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame | pd.Series,
            pd.DataFrame | pd.Series,
            list[int],
            list[int],
        ],
        None,
        None,
    ]:
        """Iterate over the folds of the dataset.

        Args:
            split (DataSplit):
                split instance orchestrating the split
            concat (bool):
                whether to concatenate the training and test feature matrices

        Yields:
            tuple:
                training and test feature matrices and target vectors
                for each fold
        """
        self.checkFeatures()
        folds = FoldsFromDataSplit(split, self.feature_standardizer)
        return folds.iterFolds(self, concat=concat)

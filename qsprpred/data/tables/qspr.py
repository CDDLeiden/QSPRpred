import json
import os
from copy import deepcopy
from typing import Callable, ClassVar, Generator, Optional

import numpy as np
import pandas as pd
from mlchemad.applicability_domains import (
    ApplicabilityDomain as MLChemADApplicabilityDomain,
)
from sklearn.preprocessing import LabelEncoder

from .mol import MoleculeTable
from ..descriptors.sets import DescriptorSet
from ...data.processing.applicability_domain import ApplicabilityDomain, MLChemADWrapper
from ...data.processing.data_filters import RepeatsFilter
from ...data.processing.feature_standardizers import (
    SKLearnStandardizer,
    apply_feature_standardizer,
)
from ...data.sampling.folds import FoldsFromDataSplit
from ...logs import logger
from ...tasks import TargetProperty, TargetTasks


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
        X_ind_outliers (np.ndarray/pd.DataFrame) : m x n Feature matrix for outliers
            in independent set, where m is the number of samples and n is the number of
            features.
        y_ind_outliers (np.ndarray/pd.DataFrame) : m-l label array for outliers in
            independent set, where m is the number of samples and equals to row of
            X_ind_outliers, and l is the number of types.
        featureNames (list of str) : feature names
        featureStandardizer (SKLearnStandardizer) : feature standardizer
        applicabilityDomain (ApplicabilityDomain) : applicability domain
    """

    _notJSON: ClassVar = [*MoleculeTable._notJSON, "X", "X_ind", "y", "y_ind"]

    def __init__(
        self,
        name: str,
        target_props: list[TargetProperty | dict] | None = None,
        df: Optional[pd.DataFrame] = None,
        smiles_col: str = "SMILES",
        add_rdkit: bool = False,
        store_dir: str = ".",
        overwrite: bool = False,
        n_jobs: int | None = 1,
        chunk_size: int | None = None,
        drop_invalids: bool = True,
        drop_empty: bool = True,
        index_cols: Optional[list[str]] = None,
        autoindex_name: str = "QSPRID",
        random_state: int | None = None,
        store_format: str = "pkl",
    ):
        """Construct QSPRdata, also apply transformations of output property if
                specified.

        Args:
            name (str):
                data name, used in saving the data
            target_props (list[TargetProperty | dict] | None):
                target properties, names
                should correspond with target columnname in df. If `None`, target
                properties will be inferred if this data set has been saved
                previously. Defaults to `None`.
            df (pd.DataFrame, optional):
                input dataframe containing smiles and target
                property. Defaults to None.
            smiles_col (str, optional):
                name of column in df containing SMILES.
                Defaults to "SMILES".
            add_rdkit (bool, optional):
                if true, column with rdkit molecules will be
                added to df. Defaults to False.
            store_dir (str, optional):
                directory for saving the output data.
                Defaults to '.'.
            overwrite (bool, optional):
                if already saved data at output dir if should
                be overwritten. Defaults to False.
            n_jobs (int, optional):
                number of parallel jobs. If <= 0, all available
                cores will be used. Defaults to 1.
            chunk_size (int, optional):
                chunk size for parallel processing.
                Defaults to 50.
            drop_invalids (bool, optional):
                if true, invalid SMILES will be dropped.
                Defaults to True.
            drop_empty (bool, optional):
                if true, rows with empty target property will  be removed.
            index_cols (list[str], optional):
                columns to be used as index in the
                dataframe. Defaults to `None` in which case a custom ID will be
                generated.
            autoindex_name (str):
                Column name to use for automatically generated IDs.
            random_state (int, optional):
                random state for splitting the data.
            store_format (str, optional):
                format to use for storing the data ('pkl' or 'csv').

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
            store_format,
        )
        # load target properties if not specified and file exists
        if target_props is None and os.path.exists(self.metaFile):
            meta = json.load(open(self.metaFile, "r"))
            target_props = meta["py/state"]["targetProperties"]
            target_props = [
                TargetProperty.fromJSON(json.dumps(x)) for x in target_props
            ]
        elif target_props is None:
            raise ValueError(
                "Target properties must be specified for a new QSPRDataset."
            )
        # load names of descriptors to use as training features
        self.featureNames = self.getFeatureNames()
        self.featureStandardizer = None
        self.applicabilityDomain = None
        # populate feature matrix and target properties
        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None
        self.targetProperties = []
        self.setTargetProperties(target_props, drop_empty)
        self.chunkSize = chunk_size
        if drop_invalids:
            self.dropInvalids()
            self.chunkSize = chunk_size
        logger.info(
            f"Dataset '{self.name}' created for "
            f"target Properties: '{self.targetProperties}'. "
            f"Number of samples: {len(self.df)}. "
            f"Chunk size: {self.chunkSize}. "
            f"Number of CPUs: {self.nJobs}."
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

    def resetTargetProperty(self, prop: TargetProperty | str):
        """Reset target property to its original value.

        Args:
            prop (TargetProperty | str): target property to reset
        """
        if isinstance(prop, str):
            prop = self.getTargetProperties([prop])[0]
        if f"{prop.name}_original" in self.df.columns:
            self.df[prop.name] = self.df[f"{prop.name}_original"]
        # save original values for next reset
        self.df[f"{prop.name}_original"] = self.df[prop.name]
        self.restoreTrainingData()

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
        self.targetProperties = []
        for prop in target_props:
            self.setTargetProperty(prop, drop_empty)

    @property
    def hasFeatures(self):
        """Check whether the currently selected set of features is not empty."""
        return True if (self.featureNames and len(self.featureNames) > 0) else False

    def getFeatureNames(self) -> list[str]:
        """Get current feature names for this data set.

        Returns:
            list[str]: list of feature names
        """
        if not self.hasDescriptors():
            return []
        features = []
        for ds in self.descriptors:
            features.extend(ds.getDescriptorNames(active_only=True))
        return features

    def restoreTrainingData(self):
        """Restore training data from the data frame.

        If the data frame contains a column 'Split_IsTrain',
        the data will be split into training and independent sets. Otherwise, the
        independent set will be empty. If descriptors are available, the resulting
        training matrices will be featurized.
        """
        logger.debug("Restoring training data...")
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
        logger.debug("Training data restored.")
        logger.debug(f"Training features shape: {self.X.shape}")
        logger.debug(f"Test set features shape: {self.X_ind.shape}")
        logger.debug(f"Training labels shape: {self.y.shape}")
        logger.debug(f"Test set labels shape: {self.y_ind.shape}")
        logger.debug(f"Training features indices: {self.X.index}")
        logger.debug(f"Test set features indices: {self.X_ind.index}")
        logger.debug(f"Training labels indices: {self.y.index}")
        logger.debug(f"Test set labels indices: {self.y_ind.index}")

    def makeRegression(self, target_property: str):
        """Switch to regression task using the given target property.

        Args:
            target_property (str): name of the target property to use for regression
        """
        target_property = self.getTargetProperties([target_property])[0]
        self.resetTargetProperty(target_property)
        target_property.task = TargetTasks.REGRESSION
        if hasattr(target_property, "th"):
            del target_property.th
        self.restoreTrainingData()
        logger.info("Target property converted to regression.")

    def makeClassification(
        self,
        target_property: str,
        th: Optional[list[float]] = None,
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
        self.resetTargetProperty(target_property)
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
        if th == "precomputed":
            assert all(
                value is None
                or (type(value) in (int, bool))
                or (isinstance(value, float) and value.is_integer())
                for value in self.df[prop_name]
            ), "Precomputed classification target must be integers or booleans."
            n_classes = len(self.df[prop_name].dropna().unique())
            target_property.task = (
                TargetTasks.MULTICLASS
                if n_classes > 2  # noqa: PLR2004
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
                assert max(self.df[prop_name]) <= max(th), (
                    "Make sure final threshold value is not smaller "
                    "than largest value of property"
                )
                assert min(self.df[prop_name]) >= min(th), (
                    "Make sure first threshold value is not larger "
                    "than smallest value of property"
                )
                self.df[f"{prop_name}_intervals"] = pd.cut(
                    self.df[prop_name], bins=th, include_lowest=True
                ).astype(str)
                self.df[prop_name] = LabelEncoder().fit_transform(
                    self.df[f"{prop_name}_intervals"]
                )
            else:
                self.df[prop_name] = self.df[prop_name] > th[0]
            target_property.task = (
                TargetTasks.SINGLECLASS if len(th) == 1 else TargetTasks.MULTICLASS
            )
            target_property.th = th
        self.restoreTrainingData()
        logger.info(f"Target property '{prop_name}' converted to classification.")

    def searchWithIndex(
        self, index: pd.Index, name: str | None = None
    ) -> "MoleculeTable":
        ret = super().searchWithIndex(index, name)
        ret = QSPRDataset.fromMolTable(ret, self.targetProperties, name=ret.name)
        ret.featureStandardizer = self.featureStandardizer
        ret.featurize()
        return ret

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
        kwargs["store_format"] = (
            mol_table.storeFormat
            if "store_format" not in kwargs
            else kwargs["store_format"]
        )
        ds = QSPRDataset(
            name,
            target_props,
            mol_table.getDF(),
            **kwargs,
        )
        ds.descriptors = mol_table.descriptors
        ds.featureNames = mol_table.getDescriptorNames()
        ds.loadDescriptorsToSplits()
        return ds

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
        descriptors: list[DescriptorSet],
        recalculate: bool = False,
        featurize: bool = True,
        *args,
        **kwargs,
    ):
        """Add descriptors to the data set.

        If descriptors are already present, they will be recalculated if `recalculate`
        is `True`. Featurization will be performed after adding descriptors if
        `featurize` is `True`. Featurization converts current data matrices to pure
        numeric matrices of selected descriptors (features).

        Args:
            descriptors (list[DescriptorSet]): list of descriptor sets to add
            recalculate (bool, optional): whether to recalculate descriptors if they are
                already present. Defaults to `False`.
            featurize (bool, optional): whether to featurize the data set splits after
                adding descriptors. Defaults to `True`.
            *args: additional positional arguments to pass to each descriptor set
            **kwargs: additional keyword arguments to pass to each descriptor set
        """
        super().addDescriptors(descriptors, recalculate, *args, **kwargs)
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
        elif "Split_IsTrain" in self.df.columns:
            self.df.drop("Split_IsOutlier", axis=1, inplace=True)
        super().save()

    def split(self, split: "DataSplit", featurize: bool = False):
        """Split dataset into train and test set.

        You can either split tha data frame itself or you can set `featurize` to `True`
        if you want to use feature matrices instead of the raw data frame.

        Args:
            split (DataSplit):
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
        if hasattr(split, "setSeed") and hasattr(split, "getSeed"):
            if split.getSeed() is None:
                split.setSeed(self.randomState)
        # split the data into train and test
        folds = FoldsFromDataSplit(split)
        self.X, self.X_ind, self.y, self.y_ind, _, _ = next(
            folds.iterFolds(self, concat=True)
        )
        # select target properties
        logger.info("Total: train: %s test: %s" % (len(self.y), len(self.y_ind)))
        logger.debug(f"First index train: {self.y.index[0]}")
        logger.debug(f"First index test: {self.y_ind.index[0]}")
        logger.debug(f"Last index train: {self.y.index[-1]}")
        logger.debug(f"Last index test: {self.y_ind.index[-1]}")
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
        if "Split_IsOutlier" in self.df.columns:
            self.df = self.df.drop("Split_IsOutlier", axis=1)
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
        the splits. They will become zero length along the feature axis (columns), but
        will retain their original length along the sample axis (rows). This is useful
        for the case where the data set has no descriptors, but the user wants to retain
        train and test splits.

        shuffle (bool): whether to shuffle the training and test sets
        random_state (int): random state for shuffling
        """
        if self.hasDescriptors() and self.hasFeatures:
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
            for ds in self.descriptors:
                to_keep = [
                    x
                    for x in ds.getDescriptorNames(active_only=False)
                    if x in self.featureNames
                ]
                ds.keepDescriptors(to_keep)

    def setFeatureStandardizer(self, feature_standardizer):
        """Set feature standardizer.

        Args:
            feature_standardizer (SKLearnStandardizer | BaseEstimator): feature
                standardizer
        """
        if not hasattr(feature_standardizer, "toFile"):
            feature_standardizer = SKLearnStandardizer(feature_standardizer)
        self.featureStandardizer = feature_standardizer

    def addFeatures(
        self,
        feature_calculators: list[DescriptorSet],
        recalculate: bool = False,
    ):
        """Add features to the data set.

        Args:
            feature_calculators (list[DescriptorSet]): list of
                feature calculators to add. Defaults to None.
            recalculate (bool): if True, recalculate features even if they are already
                present in the data set. Defaults to False.
        """
        self.addDescriptors(
            feature_calculators, recalculate=recalculate, featurize=False
        )
        self.featurize()

    def dropInvalids(self):
        ret = super().dropInvalids()
        self.restoreTrainingData()
        return ret

    def reset(self):
        """Reset the data set. Splits will be removed and all descriptors will be
        moved to the training data. Molecule
        standardization and molecule filtering are not affected.
        """
        if self.featureNames is not None:
            self.featureNames = self.getDescriptorNames()
            self.X = None
            self.X_ind = None
            self.y = None
            self.y_ind = None
            self.featureStandardizer = None
            self.applicabilityDomain = None
            self.loadDescriptorsToSplits(shuffle=False)

    def prepareDataset(
        self,
        smiles_standardizer: str | Callable | None = "chembl",
        data_filters: list | None = (RepeatsFilter(keep=True),),
        split=None,
        feature_calculators: list["DescriptorSet"] | None = None,
        feature_filters: list | None = None,
        feature_standardizer: SKLearnStandardizer | None = None,
        feature_fill_value: float = np.nan,
        applicability_domain: ApplicabilityDomain
        | MLChemADApplicabilityDomain
        | None = None,
        drop_outliers: bool = False,
        recalculate_features: bool = False,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        """Prepare the dataset for use in QSPR model.

        Arguments:
            smiles_standardizer (str | Callable): either `chembl`, `old`, or a
                partial function that reads and standardizes smiles. If `None`, no
                standardization will be performed. Defaults to `chembl`.
            data_filters (list of datafilter obj): filters number of rows from dataset
            split (datasplitter obj): splits the dataset into train and test set
            feature_calculators (list[DescriptorSet]): descriptor sets to add to the data set
            feature_filters (list of feature filter objs): filters features
            feature_standardizer (SKLearnStandardizer or sklearn.base.BaseEstimator):
                standardizes and/or scales features
            feature_fill_value (float): value to fill missing values with.
                Defaults to `numpy.nan`
            applicability_domain (applicabilityDomain obj): attaches an
                applicability domain calculator to the dataset and fits it on
                the training set
            drop_outliers (bool): whether to drop samples that are outside the
                applicability domain from the test set, if one is attached.
            recalculate_features (bool): recalculate features even if they are already
                present in the file
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
        if data_filters is not None:
            self.filter(data_filters)
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
        if feature_filters and self.hasDescriptors():
            self.filterFeatures(feature_filters)
        elif not self.hasDescriptors():
            logger.warning("No descriptors present, feature filters will be skipped.")
        # set feature standardizers
        if feature_standardizer:
            self.setFeatureStandardizer(feature_standardizer)
        # set applicability domain and fit it on the training set
        if applicability_domain:
            self.setApplicabilityDomain(applicability_domain)
        # drop outliers from test set based on applicability domain
        if drop_outliers:
            self.dropOutliers()

    def checkFeatures(self):
        """Check consistency of features and descriptors."""
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "X and y have different number of rows: "
                f"{self.X.shape[0]} != {self.y.shape[0]}"
            )
        elif self.X.shape[0] == 0:
            raise ValueError("X has no rows.")

    def getFeatures(
        self,
        inplace: bool = False,
        concat: bool = False,
        raw: bool = False,
        ordered: bool = False,
        refit_standardizer: bool = True,
    ):
        """Get the current feature sets (training and test) from the dataset.

        This method also applies any feature standardizers that have been set on the
        dataset during preparation. Outliers are dropped from the test set if they are
        present, unless `concat` is `True`.

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
            refit_standardizer (bool): If `True`, the feature standardizer will be
                refit on the training set upon this call. If `False`, the previously
                fitted standardizer will be used. Defaults to `True`. Use `False` if
                this dataset is used for prediction only and the standardizer has
                been initialized already.
        """
        self.checkFeatures()
        # get feature matrices using feature names
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
        # convert to numpy arrays and standardize
        X = df_X.values
        X_ind = df_X_ind.values if df_X_ind is not None else None
        if not raw and self.featureStandardizer:
            X, self.featureStandardizer = apply_feature_standardizer(
                self.featureStandardizer,
                df_X,
                fit=True if refit_standardizer else False,
            )
            if X_ind is not None and X_ind.shape[0] > 0:
                X_ind, _ = apply_feature_standardizer(
                    self.featureStandardizer, df_X_ind, fit=False
                )
        # convert to data frames and make sure column order is correct
        X = pd.DataFrame(X, index=df_X.index, columns=df_X.columns)
        if X_ind is not None:
            X_ind = pd.DataFrame(X_ind, index=df_X_ind.index, columns=df_X_ind.columns)
        # drop outliers from test set
        if "Split_IsOutlier" in self.df.columns and not concat:
            if X_ind is not None:
                X_ind = X_ind.loc[~self.df["Split_IsOutlier"], :]
        # replace original feature matrices if inplace
        if inplace:
            self.X = X
            self.X_ind = X_ind
        # order if concatenating
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
            if self.y_ind is not None and "Split_IsOutlier" in self.df.columns:
                y_ind = self.y_ind.loc[~self.df["Split_IsOutlier"], :]
            else:
                y_ind = self.y_ind
            return self.y, y_ind if y_ind is not None else self.y

    def getTargetProperties(self, names: list) -> list[TargetProperty]:
        """Get the target properties with the given names.

        Args:
            names (list[str]): name of the target properties

        Returns:
            list[TargetProperty]: list of target properties
        """
        return [tp for tp in self.targetProperties if tp.name in names]

    @property
    def targetPropertyNames(self):
        """Get the names of the target properties."""
        return TargetProperty.getNames(self.targetProperties)

    @property
    def isMultiTask(self):
        """Check if the dataset contains multiple target properties.

        Returns:
            `bool`: `True` if the dataset contains multiple target properties
        """
        return len(self.targetProperties) > 1

    @property
    def nTargetProperties(self):
        """Get the number of target properties in the dataset."""
        return len(self.targetProperties)

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
        self.targetProperties = [tp for tp in self.targetProperties if tp.name != name]
        self.restoreTrainingData()

    def dropEmptyProperties(self, names: list[str]):
        super().dropEmptyProperties(names)
        self.restoreTrainingData()

    def transformProperties(self, targets: list[str], transformer: Callable):
        """Transform the target properties using the given transformer.

        Args:
            targets (list[str]): list of target properties names to transform
            transformer (Callable): transformer function
            add_as (list[str] | None, optional): list of names to add the transformed
                target properties as. If `None`, the original target properties will be
                overwritten. Defaults to `None`.
        """
        super().transformProperties(targets, transformer)
        self.restoreTrainingData()

    def imputeProperties(self, names: list[str], imputer: Callable):
        super().imputeProperties(names, imputer)
        self.restoreTrainingData()

    def setTargetProperty(self, prop: TargetProperty | dict, drop_empty: bool = True):
        """Add a target property to the dataset.

        Args:
            prop (TargetProperty):
                name of the target property to add
            drop_empty (bool):
                whether to drop rows with empty target property values. Defaults to
                `True`.
        """
        logger.debug(f"Adding target property '{prop}' to dataset.")
        # deep copy the property to avoid modifying the original
        prop = deepcopy(prop)
        if isinstance(prop, dict):
            prop = TargetProperty.fromDict(prop)
        if prop.name in self.targetPropertyNames:
            logger.warning(
                f"Property '{prop}' already exists in dataset. It will be reset."
            )
        assert prop.name in self.df.columns, f"Property {prop} not found in data set."
        # add the target property to the list
        self.targetProperties.append(prop)
        # restore original values if they were transformed
        self.resetTargetProperty(prop)
        # impute the property
        if prop.imputer is not None:
            self.imputeProperties([prop.name], prop.imputer)
        # transform the property
        if prop.transformer is not None:
            self.transformProperties([prop.name], prop.transformer)
        # drop rows with missing smiles/no target property for any of
        # the target properties
        if drop_empty:
            self.dropEmptyProperties([prop.name])
        # convert classification targets to integers
        if prop.task.isClassification():
            self.makeClassification(prop.name, prop.th)

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
        folds = FoldsFromDataSplit(split, self.featureStandardizer)
        return folds.iterFolds(self, concat=concat)

    def setApplicabilityDomain(
        self, applicability_domain: ApplicabilityDomain | MLChemADApplicabilityDomain
    ):
        """Set the applicability domain calculator.

        Args:
            applicability_domain (ApplicabilityDomain | MLChemADApplicabilityDomain):
                applicability domain calculator instance
        """
        if isinstance(applicability_domain, MLChemADApplicabilityDomain):
            self.applicabilityDomain = MLChemADWrapper(applicability_domain)
        else:
            self.applicabilityDomain = applicability_domain

    def dropOutliers(self):
        """Drop outliers from the test set based on the applicability domain."""
        if self.applicabilityDomain is None:
            raise ValueError(
                "No applicability domain calculator attached to the data set."
            )
        X, X_ind = self.getFeatures()
        if X_ind.shape[0] == 0:
            logger.warning(
                "No test samples available, skipping outlier removal from test set."
            )
            return
        # check if X or X_ind contain any nan values
        if X.isna().any().any() or X_ind.isna().any().any():
            logger.warning(
                "Feature matrix contains NaN values. "
                "Please fill them before applying outlier removal."
                "Outliers will not be dropped."
            )
            return
        # fit applicability domain on the training set
        self.applicabilityDomain.fit(X)
        mask = self.applicabilityDomain.contains(X_ind)
        if not mask.sum().any():
            logger.warning(
                "All samples in the test set are outside the applicability domain,"
                "outliers will not be dropped."
            )
            return
        self.df["Split_IsOutlier"] = False
        self.df.loc[X_ind.index, "Split_IsOutlier"] = ~mask["in_domain"]

        logger.info(
            f"Marked {(~mask).sum().sum()} samples from the test set as outlier."
        )

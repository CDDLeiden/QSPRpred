import copy
import itertools
import os
import shutil
import tempfile
from typing import Iterable

import pandas as pd
from mlchemad.applicability_domains import TopKatApplicabilityDomain
from sklearn.preprocessing import StandardScaler

from ...data import RandomSplit, QSPRDataset
from ...data.descriptors.fingerprints import (
    AtomPairFP,
    AvalonFP,
    LayeredFP,
    MaccsFP,
    MorganFP,
    PatternFP,
    RDKitFP,
    RDKitMACCSFP,
    TopologicalFP,
)
from ...data.descriptors.sets import (
    RDKitDescs,
    DrugExPhyschem,
    PredictorDesc,
    TanimotoDistances,
)
from ...data.processing.data_filters import RepeatsFilter
from ...data.processing.feature_filters import HighCorrelationFilter, LowVarianceFilter
from ...data.processing.feature_standardizers import SKLearnStandardizer
from ...models import SklearnModel
from ...tasks import TargetTasks


class PathMixIn:
    """Mix-in class that provides paths to test files and directories and handles their
    creation and deletion.

    Attributes:
        generatedPath (str):
            path to the directory where generated files are stored, this directory is
            created before and cleared after each test

    """
    def setUpPaths(self):
        """Create the directories that are used for testing."""
        self.generatedPath = tempfile.mkdtemp(prefix="qsprpred_test_")
        self.clearGenerated()
        if not os.path.exists(self.generatedPath):
            os.makedirs(self.generatedPath)

    def tearDown(self):
        """Remove all files and directories that are used for testing."""
        self.clearGenerated()

    def clearGenerated(self):
        """Remove the directories that are used for testing."""
        if os.path.exists(self.generatedPath):
            shutil.rmtree(self.generatedPath)


class DataSetsPathMixIn(PathMixIn):
    """Mix-in class that provides a small and large testing data set and some common
    preparation settings to use in tests."""
    def setUpPaths(self):
        """Create the directories that are used for testing."""
        super().setUpPaths()
        self.inputBasePath = f"{os.path.dirname(__file__)}/test_files"
        self.inputDataPath = f"{self.inputBasePath}/data/"
        self.generatedDataPath = f"{self.generatedPath}/datasets"
        if not os.path.exists(self.generatedDataPath):
            os.makedirs(self.generatedDataPath)

    @staticmethod
    def getDefaultPrep():
        """Return a dictionary with default preparation settings."""
        return {
            "feature_calculators": [MorganFP(radius=2, nBits=128)],
            "split": RandomSplit(test_fraction=0.1),
            "feature_standardizer": StandardScaler(),
            "feature_filters": [LowVarianceFilter(0.05),
                                HighCorrelationFilter(0.8)],
        }

    @classmethod
    def getAllDescriptors(cls):
        """Return a list of (ideally) all available descriptor sets. For now
        they need to be added manually to the list below.

        TODO: would be nice to create the list automatically by implementing a
        descriptor set registry that would hold all installed descriptor sets.

        Returns:
            list: `list` of `DescriptorCalculator` objects
        """
        descriptor_sets = [
            RDKitDescs(),
            DrugExPhyschem(),
            PredictorDesc(
                SklearnModel.fromFile(
                    f"{os.path.dirname(__file__)}/test_files/test_predictor/"
                    f"RFC_SINGLECLASS/RFC_SINGLECLASS_meta.json"
                )
            ),
            TanimotoDistances(
                list_of_smiles=["C", "CC", "CCC"],
                fingerprint_type=MorganFP(radius=2, nBits=128),
                radius=2,
                nBits=128,
            ),
            AtomPairFP(nBits=128),
            AvalonFP(nBits=128),
            LayeredFP(nBits=128),
            MaccsFP(),
            MorganFP(radius=2, nBits=128),
            PatternFP(nBits=128),
            RDKitFP(nBits=128),
            RDKitMACCSFP(),
            TopologicalFP(nBits=128),
        ]

        return descriptor_sets

    @classmethod
    def getDefaultCalculatorCombo(cls):
        """
        Makes a list of default descriptor calculators that can be used in tests.
        It creates a calculator with only morgan fingerprints and rdkit descriptors,
        but also one with them both to test behaviour with multiple descriptor sets.
        Override this method if you want to test with other descriptor sets and
        calculator combinations.

        Returns:
            list: `list` of created `DescriptorCalculator` objects
        """
        feature_sets = [
            MorganFP(radius=3, nBits=128),
            RDKitDescs(),
        ]
        mol_descriptor_calculators = list(
            itertools.combinations(feature_sets, 1)
        ) + list(itertools.combinations(feature_sets, 2))
        return mol_descriptor_calculators

    @classmethod
    def getDataPrepGrid(cls):
        """Return a list of many possible combinations of descriptor calculators,
        splits, feature standardizers, feature filters and data filters. Again, this is
        not exhaustive, but should cover a lot of cases.

        Returns:
            grid:
                a generator that yields tuples of all possible combinations as stated
                above, each tuple is defined as: (descriptor_calculator, split,
                feature_standardizer, feature_filters, data_filters)
        """
        # get the feature calculators
        descriptor_calculators = cls.getDefaultCalculatorCombo()
        # lists with common preparation settings
        splits = [None, RandomSplit(test_fraction=0.1)]
        feature_standardizers = [None, StandardScaler()]
        feature_filters = [None, HighCorrelationFilter(0.9)]
        applicability_domains = [None, TopKatApplicabilityDomain()]
        data_filters = [
            None,
            RepeatsFilter(),
            # CategoryFilter(
            # FIXME: this needs to be made more general and not specific to one dataset
            #     name="moka_ionState7.4",
            #     values=["cationic"]
            # ),
        ]
        # All combinations of the above preparation settings (passed to prepareDataset)
        return (
            # deep copy to avoid conflicts cayed by operating on one instance twice
            copy.deepcopy(combo) for combo in itertools.product(
                descriptor_calculators,
                splits,
                feature_standardizers,
                feature_filters,
                data_filters,
                applicability_domains,
            )
        )

    @classmethod
    def getPrepCombos(cls):
        """
        Return a list of all possible preparation combinations as generated by
        `getDataPrepGrid` as well as their names. The generated list can be used
        to parameterize tests with the given named combinations.

        Returns:
            list: `list` of `list`s of all possible combinations of preparation
        """
        def get_name(obj: object):
            """
            Get the name of a data preparation object,
            or its class name if it is not a string.

            Args:
                obj: the object to get name for

            Returns:
                str: the generated name of the object
            """
            return (
                str(None)
                if obj is None
                else obj.__class__.__name__
                if (not isinstance(obj, SKLearnStandardizer))
                else str(obj)
            )

        def get_name_list(obj: Iterable | object):
            """
            Parse an generator of data preparation objects and return a string.
            Note that the method proceeds recursively so nested iterables are also
            parsed.

            Args:
                obj: the object or an generator of objects to get name for

            Returns:
                str: the generated name of the object or a list of objects
            """
            if isinstance(obj, Iterable):
                return "_".join([get_name_list(i) for i in obj])
            else:
                return get_name(obj)

        ret = [2 * [get_name_list(x)] + list(x) for x in cls.getDataPrepGrid()]
        return ret

    def getBigDF(self):
        """Get a large data frame for testing purposes.

        Returns:
            pd.DataFrame: a `pandas.DataFrame` containing the dataset
        """
        return pd.read_csv(f"{self.inputDataPath}/test_data_large.tsv", sep="\t")

    def getSmallDF(self):
        """Get a small data frame for testing purposes.

        Returns:
            pd.DataFrame: a `pandas.DataFrame` containing the dataset
        """
        return pd.read_csv(f"{self.inputDataPath}/test_data.tsv", sep="\t").sample(10)

    def createLargeTestDataSet(
        self,
        name="QSPRDataset_test_large",
        target_props=[{
            "name": "CL",
            "task": TargetTasks.REGRESSION
        }],
        preparation_settings=None,
        random_state=42,
        n_jobs=1,
        chunk_size=None,
    ):
        """Create a large dataset for testing purposes.

        Args:
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            random_state (int): random state to use for splitting and shuffling
            preparation_settings (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        return self.createTestDataSetFromFrame(
            self.getBigDF(),
            name=name,
            target_props=target_props,
            prep=preparation_settings,
            random_state=random_state,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )

    def createSmallTestDataSet(
        self,
        name="QSPRDataset_test_small",
        target_props=[{
            "name": "CL",
            "task": TargetTasks.REGRESSION
        }],
        preparation_settings=None,
        random_state=42,
    ):
        """Create a small dataset for testing purposes.

        Args:
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            random_state (int): random state to use for splitting and shuffling
            preparation_settings (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        return self.createTestDataSetFromFrame(
            self.getSmallDF(),
            name=name,
            target_props=target_props,
            random_state=random_state,
            prep=preparation_settings,
        )

    def createTestDataSetFromFrame(
        self,
        df,
        name="QSPRDataset_test",
        target_props=[{
            "name": "CL",
            "task": TargetTasks.REGRESSION
        }],
        random_state=None,
        prep=None,
        n_jobs=1,
        chunk_size=None,
    ):
        """Create a dataset for testing purposes from the given data frame.

        Args:
            df (pd.DataFrame): data frame containing the dataset
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            random_state (int): random state to use for splitting and shuffling
            prep (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        ret = QSPRDataset(
            name,
            target_props=target_props,
            df=df,
            store_dir=self.generatedDataPath,
            random_state=random_state,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )
        if prep:
            ret.prepareDataset(**prep)
        return ret

    def createLargeMultitaskDataSet(
        self,
        name="QSPRDataset_multi_test",
        target_props=[
            {
                "name": "HBD",
                "task": TargetTasks.MULTICLASS,
                "th": [-1, 1, 2, 100]
            },
            {
                "name": "CL",
                "task": TargetTasks.REGRESSION
            },
        ],
        preparation_settings=None,
        random_state=42,
    ):
        """Create a large dataset for testing purposes.

        Args:
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            preparation_settings (dict): dictionary containing preparation settings
            random_state (int): random state to use for splitting and shuffling

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        return self.createTestDataSetFromFrame(
            self.getBigDF(),
            name=name,
            target_props=target_props,
            random_state=random_state,
            prep=preparation_settings,
        )

    def validate_split(self, dataset):
        """Check if the split has the data it should have after splitting."""
        self.assertTrue(dataset.X is not None)
        self.assertTrue(dataset.X_ind is not None)
        self.assertTrue(dataset.y is not None)
        self.assertTrue(dataset.y_ind is not None)


class ModelDataSetsPathMixIn(DataSetsPathMixIn):
    """This class sets up the datasets for the model tests."""
    def setUpPaths(self):
        """Set up the test environment."""
        super().setUpPaths()
        self.generatedModelsPath = f"{self.generatedPath}/models/"
        if not os.path.exists(self.generatedModelsPath):
            os.makedirs(self.generatedModelsPath)

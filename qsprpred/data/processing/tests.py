import copy
import itertools
from unittest import skipIf

import numpy as np
import pandas as pd
from mlchemad.applicability_domains import KNNApplicabilityDomain
from parameterized import parameterized
from rdkit.Chem import Mol
from sklearn.preprocessing import StandardScaler

from .mol_processor import MolProcessor
from ..descriptors.fingerprints import MorganFP
from ..descriptors.sets import DataFrameDescriptorSet
from ... import TargetTasks
from ...data import QSPRDataset
from ...data.processing.applicability_domain import MLChemADWrapper
from ...data.processing.data_filters import CategoryFilter, RepeatsFilter
from ...data.processing.feature_filters import (
    BorutaFilter,
    HighCorrelationFilter,
    LowVarianceFilter,
)
from ...data.processing.feature_standardizers import SKLearnStandardizer
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn, PathMixIn


class TestDataFilters(DataSetsPathMixIn, QSPRTestCase):
    """Small tests to only check if the data filters work on their own.

    The tests here should be used to check for all their specific parameters and
    edge cases."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testCategoryFilter(self):
        """Test the category filter, which drops specific values from dataset
        properties."""
        remove_cation = CategoryFilter(name="moka_ionState7.4", values=["cationic"])
        df_anion = remove_cation(self.getBigDF())
        self.assertTrue((df_anion["moka_ionState7.4"] == "cationic").sum() == 0)

        only_cation = CategoryFilter(
            name="moka_ionState7.4", values=["cationic"], keep=True
        )
        df_cation = only_cation(self.getBigDF())
        self.assertTrue((df_cation["moka_ionState7.4"] != "cationic").sum() == 0)

    def testRepeatsFilter(self):
        """Test the duplicate filter, which drops rows with identical descriptors
        from dataset."""
        descriptor_names = [f"Descriptor_{i}" for i in range(3)]
        df = pd.DataFrame(
            data=np.array(
                [
                    ["C", 1, 2, 1, 1],
                    ["CC", 1, 2, 2, 2],
                    ["CCC", 1, 2, 3, 3],
                    ["C", 1, 2, 1, 4],
                    ["C", 1, 2, 1, 5],
                    ["CC", 1, 2, 2, 6],  # 3rd "descriptor" is length of SMILES
                ]
            ),
            columns=["SMILES", *descriptor_names, "Year"],
        )
        # only warnings
        df_copy = copy.deepcopy(df)
        dup_filter1 = RepeatsFilter(keep=True)
        df_copy = dup_filter1(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), len(df))
        self.assertTrue(df_copy.equals(df))
        # drop duplicates
        df_copy = copy.deepcopy(df)
        dup_filter2 = RepeatsFilter(keep=False)
        df_copy = dup_filter2(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), 1)  # only CCC has one occurence
        self.assertTrue(df_copy.equals(df.iloc[[2]]))
        # keep first, by year
        df_copy = copy.deepcopy(df)
        dup_filter3 = RepeatsFilter(keep="first", timecol="Year")
        df_copy = dup_filter3(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), 3)  # three unique SMILES
        self.assertTrue(df_copy.equals(df.iloc[[0, 1, 2]]))  # keep first by year

        # check with additional columns
        df_copy = copy.deepcopy(df)
        df_copy["proteinid"] = ["A", "B", "B", "B", "B", "B"]
        dup_filter4 = RepeatsFilter(additional_cols=["proteinid"])
        df_copy = dup_filter4(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), 2)  # C (protein A, idx 0) and CCC are unique,
        # but C (protein B, idx 3) is a duplicate
        # of C (protein B, idx 4) and is dropped

    def testConsistency(self):
        dataset = self.createLargeTestDataSet()
        remove_cation = CategoryFilter(name="moka_ionState7.4", values=["cationic"])
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() > 0)
        dataset.filter([remove_cation])
        self.assertEqual(len(dataset.getDF()), len(dataset.getFeatures(concat=True)))
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() == 0)


class TestFeatureFilters(PathMixIn, QSPRTestCase):
    """Tests to check if the feature filters work on their own.

    Note: This also tests the `DataframeDescriptorSet`,
    as it is used to add test descriptors.
    """

    def setUp(self):
        """Set up the small test Dataframe."""
        super().setUp()
        self.nCPU = 2  # just to test parallel processing
        self.chunkSize = 2
        self.setUpPaths()
        descriptors = [
            "Descriptor_F1",
            "Descriptor_F2",
            "Descriptor_F3",
            "Descriptor_F4",
            "Descriptor_F5",
        ]
        self.df_descriptors = pd.DataFrame(
            data=np.array(
                [
                    [1, 4, 2, 6, 2],
                    [1, 8, 4, 2, 4],
                    [1, 4, 3, 2, 5],
                    [1, 8, 4, 9, 8],
                    [1, 4, 2, 3, 9],
                    [1, 8, 4, 7, 12],
                ]
            ),
            columns=descriptors,
        )
        self.df = pd.DataFrame(
            data=np.array([["C", 1], ["C", 2], ["C", 3], ["C", 4], ["C", 5], ["C", 6]]),
            columns=["SMILES", "y"],
        )
        self.dataset = QSPRDataset(
            "TestFeatureFilters",
            target_props=[{"name": "y", "task": TargetTasks.REGRESSION}],
            df=self.df,
            store_dir=self.generatedPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.df_descriptors["QSPRID"] = self.dataset.getProperty(
            self.dataset.idProp
        ).values
        self.df_descriptors.set_index("QSPRID", inplace=True, drop=True)
        self.dataset.addDescriptors([DataFrameDescriptorSet(self.df_descriptors)])
        self.descriptors = self.dataset.featureNames

    def recalculateWithMultiIndex(self):
        self.dataset.dropDescriptors(self.dataset.descriptorSets)
        self.df_descriptors["ID_COL1"] = (
            self.dataset.getProperty(self.dataset.idProp)
            .apply(lambda x: x.split("_")[0])
            .to_list()
        )
        self.df_descriptors["ID_COL2"] = (
            self.dataset.getProperty(self.dataset.idProp)
            .apply(lambda x: x.split("_")[1])
            .to_list()
        )
        self.dataset.addProperty("ID_COL1", self.df_descriptors["ID_COL1"].values)
        self.dataset.addProperty("ID_COL2", self.df_descriptors["ID_COL2"].values)
        self.dataset.addDescriptors(
            [
                DataFrameDescriptorSet(
                    self.df_descriptors,
                    ["ID_COL1", "ID_COL2"],
                )
            ]
        )

    # def testDefaultDescriptorAdd(self):
    #     """Test adding without index columns."""
    #     # TODO: issue 88 needs to be solved for this to work
    #     self.dataset.nJobs = 1
    #     df_new = self.dataset.getFeatures(concat=True).copy()
    #     calc = DataFrameDescriptorSet(df_new, suffix="new_df_desc")
    #     self.dataset.addDescriptors([calc])

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def testLowVarianceFilter(self, use_index_cols):
        """Test the low variance filter, which drops features with a variance below
        a threshold."""
        if use_index_cols:
            self.recalculateWithMultiIndex()
        self.dataset.filterFeatures([LowVarianceFilter(0.01)])
        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[1:])

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def testHighCorrelationFilter(self, use_index_cols):
        """Test the high correlation filter, which drops features with a correlation
        above a threshold."""
        if use_index_cols:
            self.recalculateWithMultiIndex()
        self.dataset.filterFeatures([HighCorrelationFilter(0.8)])
        # check if correct columns selected and values still original
        self.descriptors.pop(2)
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors)
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    @skipIf(
        int(np.__version__.split(".")[1]) >= 24,
        "numpy 1.24.0 not compatible with boruta",
    )
    def testBorutaFilter(self, use_index_cols):
        """Test the Boruta filter, which removes the features which are statistically as
        relevant as random features."""
        if use_index_cols:
            self.recalculateWithMultiIndex()
        self.dataset.filterFeatures([BorutaFilter()])
        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[-1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[-1:])


class TestFeatureStandardizer(DataSetsPathMixIn, QSPRTestCase):
    """Test the feature standardizer."""

    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([MorganFP(radius=3, nBits=128)])

    def testFeaturesStandardizer(self):
        """Test the feature standardizer fitting, transforming and serialization."""
        scaler = SKLearnStandardizer.fromFit(self.dataset.X, StandardScaler())
        scaled_features = scaler(self.dataset.X)
        scaler.toFile(f"{self.generatedPath}/test_scaler.json")
        scaler_fromfile = SKLearnStandardizer.fromFile(
            f"{self.generatedPath}/test_scaler.json"
        )
        scaled_features_fromfile = scaler_fromfile(self.dataset.X)
        self.assertIsInstance(scaled_features, np.ndarray)
        self.assertEqual(scaled_features.shape, (len(self.dataset), 128))
        self.assertEqual(
            np.array_equal(scaled_features, scaled_features_fromfile), True
        )


def getCombos():
    return list(
        itertools.product(
            [1, None],
            [50, None],
            [None, ["fu", "CL"], ["SMILES"]],
            [True, False],
            [None, [1, 2]],
            [None, {"a": 1}],
        )
    )


class TestMolProcessor(DataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    class TestingProcessor(MolProcessor):
        def __call__(self, mols, props, *args, **kwargs):
            assert "QSPRID" in props, "QSPRID not in props"
            result = []
            for mol in mols:
                result.append((mol, *props.keys(), *args, *kwargs.keys()))
            return np.array(result)

        @property
        def supportsParallel(self):
            return True

        @property
        def requiredProps(self) -> list[str]:
            return ["QSPRID"]

    @parameterized.expand([["_".join([str(i) for i in x]), *x] for x in getCombos()])
    def testMolProcess(self, _, n_jobs, chunk_size, props, add_rdkit, args, kwargs):
        dataset = self.createLargeTestDataSet()
        dataset.nJobs = n_jobs
        dataset.chunkSize = chunk_size
        self.assertTrue(dataset.nJobs is not None)
        self.assertTrue(dataset.chunkSize is not None)
        self.assertTrue(dataset.nJobs > 0)
        self.assertTrue(dataset.chunkSize > 0)
        result = dataset.processMols(
            self.TestingProcessor(),
            add_props=props,
            as_rdkit=add_rdkit,
            proc_args=args,
            proc_kwargs=kwargs,
        )
        expected_props = (
            [*props, "QSPRID", "SMILES"]
            if props is not None
            else dataset.getProperties()
        )
        expected_props = set(expected_props)
        expected_args = set(args) if args is not None else set()
        expected_kwargs = set(kwargs) if kwargs is not None else set()
        expected_cols = (
            len(expected_props) + len(expected_args) + len(expected_kwargs) + 1
        )
        for item in result:
            if dataset.nJobs > 1:
                self.assertTrue(item.shape[0] <= dataset.chunkSize)
            else:
                self.assertTrue(item.shape[0] == len(dataset))
            self.assertEqual(
                item.shape[1],
                expected_cols,
            )
            for prop in expected_props:
                self.assertIn(prop, item)
            if add_rdkit:
                self.assertIsInstance(item[0, 0], Mol)
            else:
                self.assertIsInstance(item[0, 0], str)


class testApplicabilityDomain(DataSetsPathMixIn, QSPRTestCase):
    """Test the applicability domain."""

    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([MorganFP(radius=3, nBits=1000)])

    def testApplicabilityDomain(self):
        """Test the applicability domain fitting, transforming and serialization."""
        ad = MLChemADWrapper(
            KNNApplicabilityDomain(dist="jaccard", scaling=None, alpha=0.95)
        )
        ad.fit(self.dataset.X)
        self.assertIsInstance(ad.contains(self.dataset.X), pd.DataFrame)
        filtered_dataset = ad.filter(self.dataset.X)
        self.assertIsInstance(filtered_dataset, pd.DataFrame)

        ad.toFile(f"{self.generatedPath}/test_ad.json")
        ad_fromfile = MLChemADWrapper.fromFile(f"{self.generatedPath}/test_ad.json")
        self.assertIsInstance(ad_fromfile.contains(self.dataset.X), pd.DataFrame)
        filtered_dataset_fromfile = ad_fromfile.filter(self.dataset.X)
        self.assertIsInstance(filtered_dataset_fromfile, pd.DataFrame)

import copy

import numpy as np
import pandas as pd
from mlchemad.applicability_domains import KNNApplicabilityDomain
from sklearn.preprocessing import StandardScaler

from ... import TargetTasks
from ...data import QSPRDataset
from ...data.descriptors.calculators import (
    CustomDescriptorsCalculator,
    MoleculeDescriptorsCalculator,
)
from ...data.descriptors.sets import DataFrameDescriptorSet, FingerprintSet
from ...data.processing.applicability_domain import MLChemAD
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
    """Tests to check if the feature filters work on their own."""
    def setUp(self):
        """Set up the small test Dataframe."""
        super().setUp()
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
            target_props=[{
                "name": "y",
                "task": TargetTasks.REGRESSION
            }],
            df=self.df,
            store_dir=self.generatedPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.df_descriptors.index = self.dataset.df.index
        calculator = CustomDescriptorsCalculator(
            [DataFrameDescriptorSet(self.df_descriptors)]
        )
        self.dataset.addCustomDescriptors(calculator)
        self.descriptors = self.dataset.featureNames

    def testLowVarianceFilter(self):
        """Test the low variance filter, which drops features with a variance below
        a threshold."""
        self.dataset.filterFeatures([LowVarianceFilter(0.01)])
        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[1:])

    def testHighCorrelationFilter(self):
        """Test the high correlation filter, which drops features with a correlation
        above a threshold."""
        self.dataset.filterFeatures([HighCorrelationFilter(0.8)])
        # check if correct columns selected and values still original
        self.descriptors.pop(2)
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors)
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors)

    def testBorutaFilter(self):
        """Test the Boruta filter, which removes the features which are statistically as
        relevant as random features."""
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
        self.dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]
            )
        )

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
        self.assertEqual(scaled_features.shape, (len(self.dataset), 1000))
        self.assertEqual(
            np.array_equal(scaled_features, scaled_features_fromfile), True
        )


class testApplicabilityDomain(DataSetsPathMixIn, QSPRTestCase):
    """Test the applicability domain."""
    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]
            )
        )

    def testApplicabilityDomain(self):
        """Test the applicability domain fitting, transforming and serialization."""
        ad = MLChemAD(
            KNNApplicabilityDomain(
                dist="rogerstanimoto", scaling=None, hard_threshold=0.75
            )
        )
        ad.fit(self.dataset.X)
        self.assertIsInstance(ad.contains(self.dataset.X), pd.DataFrame)
        filtered_dataset = ad.filter(self.dataset.X)
        self.assertIsInstance(filtered_dataset, pd.DataFrame)

        ad.toFile(f"{self.generatedPath}/test_ad.json")
        ad_fromfile = MLChemAD.fromFile(f"{self.generatedPath}/test_ad.json")
        self.assertIsInstance(ad_fromfile.contains(self.dataset.X), pd.DataFrame)
        filtered_dataset_fromfile = ad_fromfile.filter(self.dataset.X)
        self.assertIsInstance(filtered_dataset_fromfile, pd.DataFrame)

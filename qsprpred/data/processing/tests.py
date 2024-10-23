import copy
import itertools
from unittest import skipIf

import numpy as np
import pandas as pd
from mlchemad.applicability_domains import KNNApplicabilityDomain as KNNAD
from parameterized import parameterized
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from ... import TargetTasks
from ...data.processing.applicability_domain import (
    KNNApplicabilityDomain,
    MLChemADWrapper,
)
from ...data.processing.data_filters import CategoryFilter, RepeatsFilter
from ...data.processing.feature_filters import (
    BorutaFilter,
    HighCorrelationFilter,
    LowVarianceFilter,
)
from ...data.processing.feature_standardizers import SKLearnStandardizer
from ...data.pipelines.pipeline import QSPRPipeline
from ...data.tables.qspr import QSPRTable
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn, PathMixIn
from ..descriptors.fingerprints import MorganFP
from ..descriptors.sets import DataFrameDescriptorSet
from ..storage.interfaces.stored_mol import StoredMol
from .mol_processor import MolProcessor


class TestDataFilters(DataSetsPathMixIn, QSPRTestCase):
    """Small tests to only check if the data filters work on their own.

    The tests here should be used to check for all their specific parameters and
    edge cases."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([MorganFP(radius=2, nBits=20)])

    def testCategoryFilter(self):
        """Test the category filter, which drops specific values from dataset
        properties."""
        self.assertTrue(
            (self.dataset.getDF()["moka_ionState7.4"] == "cationic").sum() > 0
        )
        
        # Test with keep=False
        remove_cation = CategoryFilter(
            prop=self.dataset.getDF()["moka_ionState7.4"], values=["cationic"]
        )
        filtered_df, _ = remove_cation.transform(self.dataset.getDF())
        self.assertTrue((filtered_df["moka_ionState7.4"] == "cationic").sum() == 0)

        # Test with keep=True
        only_cation = CategoryFilter(
            prop=self.dataset.getDF()["moka_ionState7.4"],
            values=["cationic"],
            keep=True
        )
        filtered_df, _ = only_cation.transform(self.dataset.getDF())
        self.assertTrue((filtered_df["moka_ionState7.4"] != "cationic").sum() == 0)

    def testRepeatsFilter(self):
        """Test the duplicate filter, which drops rows with identical descriptors
        from dataset."""
        ## check assumptions about the test data
        # check that the descriptor rows 0, 3 and 5 are identical
        descriptors = self.dataset.getDescriptors()
        self.assertTrue(np.array_equal(descriptors.iloc[0], descriptors.iloc[3]))
        self.assertTrue(np.array_equal(descriptors.iloc[0], descriptors.iloc[5]))
        
        # check all other rows are unique
        self.assertEqual(
            len(descriptors.drop_duplicates(keep=False)), len(descriptors)-3
        )

        ## test the filter
        # only warnings
        warn_reps = RepeatsFilter(keep=True)
        filtered_df, _ = warn_reps.transform(descriptors)
        self.assertEqual(len(filtered_df), len(descriptors))
        self.assertTrue(filtered_df.equals(descriptors))
        
        # drop duplicates
        drop_reps = RepeatsFilter(keep=False)
        filtered_df, _ = drop_reps.transform(descriptors)
        self.assertEqual(len(filtered_df), len(descriptors)-3)
        
        # keep first, by year
        keep_first = RepeatsFilter(
            keep="first", timecol=self.dataset.getDF()["Year of first disclosure"]
        )
        filtered_df, _ = keep_first.transform(descriptors)
        self.assertEqual(len(filtered_df), len(descriptors)-2)
        self.assertIn(descriptors.iloc[0].name, filtered_df.index)

        # check with additional columns 
        proteinid = ["A", "B", "B", "A", "B", "B", "B", "B", "B"]
        drop_reps_protein = RepeatsFilter(
            keep=False,
            additional_cols={"proteinid": pd.Series(proteinid, index=descriptors.index)}
        )
        filtered_df, _ = drop_reps_protein.transform(descriptors)
        self.assertEqual(len(filtered_df), len(descriptors)-2)

    def testConsistency(self):
        dataset = self.createLargeTestDataSet()
        remove_cation = CategoryFilter(prop=dataset.getDF()["moka_ionState7.4"], values=["cationic"])
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() > 0)
        dataset.filter([remove_cation])
        self.assertEqual(len(dataset.getDF()), len(dataset.getFeatures(concat=True)[0]))
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
        self.df = pd.DataFrame(
            data=np.array([["C", 1], ["C", 2], ["C", 3], ["C", 4], ["C", 5], ["C", 6]]),
            columns=["SMILES", "y"],
        )
        self.dataset = QSPRTable.fromDF(
            "TestFeatureFilters",
            target_props=[{
                "name": "y",
                "task": TargetTasks.REGRESSION
            }],
            df=self.df,
            path=self.generatedPath,
        )
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
        self.df_descriptors[self.dataset.idProp] = list(
            self.dataset.getProperty(self.dataset.idProp)
        )
        self.df_descriptors.set_index(self.dataset.idProp, inplace=True, drop=True)
        self.dataset.addDescriptors([DataFrameDescriptorSet(self.df_descriptors)])
        self.descriptors = self.dataset.featureNames

    def recalculateWithMultiIndex(self):
        self.dataset.dropDescriptorSets(self.dataset.descriptorSets, full_removal=True)
        self.df_descriptors["ID_COL1"] = (
            self.dataset.getProperty(self.dataset.idProp
                                    ).apply(lambda x: x.split("_")[0]).to_list()
        )
        self.df_descriptors["ID_COL2"] = (
            self.dataset.getProperty(self.dataset.idProp
                                    ).apply(lambda x: x.split("_")[-1]).to_list()
        )
        self.dataset.addProperty("ID_COL1", self.df_descriptors["ID_COL1"].values)
        self.dataset.addProperty("ID_COL2", self.df_descriptors["ID_COL2"].values)
        self.dataset.addDescriptors(
            [DataFrameDescriptorSet(
                self.df_descriptors,
                ["ID_COL1", "ID_COL2"],
            )]
        )

    def testDefaultDescriptorAdd(self):
        """Test adding without index columns."""
        self.dataset.nJobs = 1
        df_new = self.dataset.getFeatures(concat=True)[0].copy()
        calc = DataFrameDescriptorSet(df_new, suffix="new_df_desc")
        self.dataset.addDescriptors([calc])

    @parameterized.expand([
        (True, ),
        (False, ),
    ])
    def testLowVarianceFilter(self, use_index_cols):
        """Test the low variance filter, which drops features with a variance below
        a threshold."""
        if use_index_cols:
            self.recalculateWithMultiIndex()
        self.dataset.applyPipeline(LowVarianceFilter(0.01), inplace=True)
        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[1:])

    @parameterized.expand([
        (True, ),
        (False, ),
    ])
    def testHighCorrelationFilter(self, use_index_cols):
        """Test the high correlation filter, which drops features with a correlation
        above a threshold."""
        if use_index_cols:
            self.recalculateWithMultiIndex()
        self.dataset.applyPipeline(HighCorrelationFilter(0.8), inplace=True)
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
    def testBorutaFilter(self, use_index_cols):
        """Test the Boruta filter, which removes the features which are statistically as
        relevant as random features."""
        if use_index_cols:
            self.recalculateWithMultiIndex()
        self.dataset.applyPipeline(BorutaFilter(), inplace=True)
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
        scaler = SKLearnStandardizer(StandardScaler())
        scaled_features, _ = scaler.fitTransform(self.dataset.X)
        scaler.toFile(f"{self.generatedPath}/test_scaler.json")
        scaler_fromfile = SKLearnStandardizer.fromFile(
            f"{self.generatedPath}/test_scaler.json"
        )
        scaled_features_fromfile, _ = scaler_fromfile.transform(self.dataset.X)
        self.assertIsInstance(scaled_features, pd.DataFrame)
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
            [None, {
                "a": 1
            }],
        )
    )


class TestMolProcessor(DataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    class TestingProcessor(MolProcessor):
        def __init__(self, id_prop):
            self.id_prop = id_prop

        def __call__(self, mols, *args, **kwargs):
            result = []
            for mol in mols:
                if not isinstance(mol, Chem.Mol):
                    assert self.id_prop in mol.props
                    result.append((mol, mol.props, {"args": args}, {"kwargs": kwargs}))
                else:
                    result.append((mol, None, {"args": args}, {"kwargs": kwargs}))
            return np.array(result)

        @property
        def supportsParallel(self):
            return True

        @property
        def requiredProps(self) -> list[str]:
            return [self.id_prop]

    @parameterized.expand([["_".join([str(i) for i in x]), *x] for x in getCombos()])
    def testMolProcess(self, _, n_jobs, chunk_size, props, add_rdkit, args, kwargs):
        dataset = self.createLargeTestDataSet()
        dataset.storage.nJobs = n_jobs
        dataset.storage.chunkSize = chunk_size
        self.assertTrue(dataset.storage.nJobs is not None)
        self.assertTrue(dataset.storage.nJobs > 0)
        result = dataset.processMols(
            self.TestingProcessor(dataset.idProp),
            add_props=props,
            proc_args=args,
            proc_kwargs=kwargs,
            mol_type="rdkit" if add_rdkit else "mol",
        )
        expected_props = (
            [*props, dataset.idProp] if props is not None else dataset.getProperties()
        )
        expected_props = set(expected_props)
        expected_args = set(args) if args is not None else set()
        expected_kwargs = set(kwargs) if kwargs is not None else set()
        for item in result:
            if dataset.storage.chunkSize is not None:
                self.assertTrue(item.shape[0] <= dataset.storage.chunkSize)
            if add_rdkit:
                self.assertIsInstance(item[0, 0], Chem.Mol)
            else:
                self.assertIsInstance(item[0, 0], StoredMol)
            if not add_rdkit:
                self.assertEqual(len(expected_props), len(item[0, 1]))
                for prop in expected_props:
                    self.assertIn(prop, item[0, 1])
            self.assertEqual(len(expected_args), len(item[0, 2]["args"]))
            self.assertEqual(len(expected_kwargs), len(item[0, 3]["kwargs"]))


class TestApplicabilityDomain(DataSetsPathMixIn, QSPRTestCase):
    """Test the applicability domain."""
    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([MorganFP(radius=3, nBits=1000)])

    def testApplicabilityDomain(self):
        """Test the applicability domain fitting, transforming and serialization."""
        ad = MLChemADWrapper(KNNAD(dist="jaccard", scaling=None, alpha=0.95))
        ad.fit(self.dataset.X)
        self.assertIsInstance(ad.contains(self.dataset.X), pd.Series)

        ad.toFile(f"{self.generatedPath}/test_ad.json")
        ad_fromfile = MLChemADWrapper.fromFile(f"{self.generatedPath}/test_ad.json")
        self.assertIsInstance(ad_fromfile.contains(self.dataset.X), pd.Series)

    def testContinousAD(self):
        """Test the applicability domain for continuous data."""
        ad = KNNApplicabilityDomain(dist="euclidean", scaling="standard", alpha=0.95)
        ad.fit(self.dataset.X)

        with self.assertRaises(ValueError):
            ad.contains(ad.contains(self.dataset.X))

        self.assertIsInstance(ad.transform(self.dataset.X), pd.Series)

        ad.threshold = 0.3
        ad.direction = "<"
        self.assertIsInstance(ad.contains(self.dataset.X), pd.Series)

        ad.toFile(f"{self.generatedPath}/test_ad.json")
        MLChemADWrapper.fromFile(f"{self.generatedPath}/test_ad.json")

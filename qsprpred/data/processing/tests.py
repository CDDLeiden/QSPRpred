import copy
import itertools
from unittest import skipIf

import numpy as np
import pandas as pd
from mlchemad.applicability_domains import KNNApplicabilityDomain as KNNAD
from parameterized import parameterized
from rdkit import Chem
from sklearn.preprocessing import StandardScaler, Binarizer

from ... import TargetTasks
from ...data.processing.applicability_domain import (
    KNNApplicabilityDomain,
    MLChemAD,
)
from ...data.processing.data_filters import CategoryFilter, RepeatsFilter
from ...data.processing.feature_filters import (
    BorutaFilter,
    HighCorrelationFilter,
    LowVarianceFilter,
)
from ...data.processing.feature_standardizers import SKLearnStandardizer
from .pipeline import DatasetPipeline, Pipeline, DummyStep
from ...data.tables.qspr import QSPRTable
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn, PathMixIn
from ..descriptors.fingerprints import MorganFP
from ..descriptors.sets import DataFrameDescriptorSet, RandomDescs
from ..storage.interfaces.stored_mol import StoredMol
from .mol_processor import MolProcessor
from ...data.sampling.splits import RandomSplit

#####-----------------Test MolProcessor-----------------#####
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

#####-----------------Test Applicability Domain-----------------#####
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
        ad = MLChemAD(KNNAD(dist="jaccard", scaling=None, alpha=0.95))
        ad.fit(self.dataset.getDescriptors())
        self.assertIsInstance(ad.contains(self.dataset.getDescriptors()), pd.Series)

        ad.toFile(f"{self.generatedPath}/test_ad.json")
        ad_fromfile = MLChemAD.fromFile(f"{self.generatedPath}/test_ad.json")
        self.assertIsInstance(ad_fromfile.contains(self.dataset.getDescriptors()), pd.Series)

    def testContinousAD(self):
        """Test the applicability domain for continuous data."""
        ad = KNNApplicabilityDomain(dist="euclidean", scaling="standard", alpha=0.95)
        ad.fit(self.dataset.getDescriptors())

        with self.assertRaises(ValueError):
            ad.contains(ad.contains(self.dataset.getDescriptors()))

        self.assertIsInstance(ad.transform(self.dataset.getDescriptors()), pd.Series)

        ad.threshold = 0.3
        ad.direction = "<"
        self.assertIsInstance(ad.contains(self.dataset.getDescriptors()), pd.Series)

        ad.toFile(f"{self.generatedPath}/test_ad.json")
        MLChemAD.fromFile(f"{self.generatedPath}/test_ad.json")

#####-----------------Test Pipeline-----------------#####
class TestPipeline(DataSetsPathMixIn, QSPRTestCase):
    """Test the dataset pipeline."""
    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.setUpPaths()
        dataset = self.createSmallTestDataSet(self.__class__.__name__)
        dataset.addDescriptors([RandomDescs(n=10, seed=42)])
        targets = dataset.getTargets()
        descriptors = dataset.getDescriptors()
        train_idx, test_idx = next(dataset.split(RandomSplit(seed=42)))
        self.X_train, self.X_test = descriptors.loc[train_idx], descriptors.loc[test_idx]
        self.y_train, self.y_test = targets.loc[train_idx], targets.loc[test_idx]

    def testApply(self):
        """Test the pipeline apply method."""        
        def checkOutput(
            input_data: tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None],
            pipeline: Pipeline
        ):
            """Check if the output of the pipeline is the same type and shape as the input."""
            output_data = pipeline.apply(*input_data)
            for df_in, df_out in zip(input_data, output_data):
                if df_in is not None:
                    self.assertIsInstance(df_out, pd.DataFrame)
                    self.assertEqual(df_in.shape, df_out.shape)
                else:
                    self.assertIsNone(df_out)
                    
        # Test apply with a dummy step on init and None inputs
        pipeline = Pipeline(steps={"dummy_step": DummyStep()}, seed=42)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(pipeline.randomState, 42)
        self.assertEqual(len(pipeline.steps), 1)
        self.assertIn("dummy_step", pipeline.steps)
        
        checkOutput((self.X_train, None, None, None), pipeline)
        checkOutput((self.X_train, self.y_train, None, None), pipeline)
        checkOutput((self.X_train, None, self.X_test, None), pipeline)
        checkOutput((self.X_train, self.y_train, self.X_test, self.y_test), pipeline)
        
        # Test apply with more than one step
        pipeline = Pipeline(
            steps={
                "dummy_step1": DummyStep(),
                "dummy_step2": DummyStep(),
            },
            seed=42
        )
        checkOutput((self.X_train, self.y_train, self.X_test, self.y_test), pipeline)
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(["dummy_step1", "dummy_step2"], list(pipeline.steps.keys()))
        
        # test apply with no steps
        pipeline = Pipeline(steps={}, seed=42)
        self.assertEqual(len(pipeline.steps), 0)
        checkOutput((self.X_train, self.y_train, self.X_test, self.y_test), pipeline)
        
        # test setting fit argument and fitted state
        pipeline = Pipeline(steps={"scaler": SKLearnStandardizer(Binarizer())})
        self.assertFalse(pipeline.steps["scaler"].fitted)
        self.assertFalse(pipeline.fitted)
        with self.assertRaises(ValueError):
            # test that the pipeline raises an error if fit is not called
            pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test, fit=False)
        
        pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertTrue(pipeline.steps["scaler"].fitted)
        self.assertTrue(pipeline.fitted)
        
        # test if the input data is not modified
        pipeline = Pipeline(steps={"scaler": SKLearnStandardizer(StandardScaler())})
        X_train_copy = self.X_train.copy()
        X_test_copy = self.X_test.copy()
        y_train_copy = self.y_train.copy()
        y_test_copy = self.y_test.copy()
        pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertTrue(self.X_train.equals(X_train_copy))
        self.assertTrue(self.X_test.equals(X_test_copy))
        self.assertTrue(self.y_train.equals(y_train_copy))
        self.assertTrue(self.y_test.equals(y_test_copy))
        
    def testApplyWithFixedSteps(self):
        """Test the pipeline apply method with fixed steps."""
        pipeline = Pipeline(steps={"scaler": SKLearnStandardizer(Binarizer())}, fixed=["scaler"])
        self.assertEqual(pipeline.fixed, ["scaler"])
        
        pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertFalse(pipeline.steps["scaler"].fitted)
        self.assertTrue(pipeline.fitted)
    
    def testApplyWithFitOn(self):
        """Test the pipeline apply method with fit_on argument."""
        # test with fit_on="test"
        test_means = self.X_test.mean().to_list()
        pipeline = Pipeline(
            steps={"scaler": SKLearnStandardizer(StandardScaler())},
            fit_on={"scaler": "test"}
        )
        self.assertEqual(pipeline.fitOn, {"scaler": "test"})
        pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertListEqual(
            pipeline.steps["scaler"].scaler.mean_.tolist(), test_means
        )
        
        # test with fit_on="both"
        all_means = pd.concat([self.X_train, self.X_test]).mean().to_list()
        pipeline = Pipeline(
            steps={"scaler": SKLearnStandardizer(StandardScaler())},
            fit_on={"scaler": "both"}
        )
        self.assertEqual(pipeline.fitOn, {"scaler": "both"})
        pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertListEqual(
            pipeline.steps["scaler"].scaler.mean_.tolist(), all_means
        )
        
    def testApplyWithApplyTo(self):
        """Test the pipeline apply method with apply_to argument."""
        # test with apply_to="train"
        pipeline = Pipeline(
            steps={"scaler": SKLearnStandardizer(StandardScaler())},
            apply_to={"scaler": "train"}
        )
        self.assertEqual(pipeline.applyTo, {"scaler": "train"})
        X_train_out, _, X_test_out, _ = pipeline.apply(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        self.assertTrue(X_test_out.equals(self.X_test))
        self.assertFalse(X_train_out.equals(self.X_train))
        
        # test with apply_to="test"
        pipeline = Pipeline(
            steps={"scaler": SKLearnStandardizer(StandardScaler())},
            apply_to={"scaler": "test"}
        )
        self.assertEqual(pipeline.applyTo, {"scaler": "test"})
        X_train_out, _, X_test_scaled, _ = pipeline.apply(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        self.assertFalse(X_test_scaled.equals(self.X_test))
        self.assertTrue(X_train_out.equals(self.X_train))

    def testAddStep(self):
        """Test the pipeline add step method."""
        pipeline = Pipeline(steps={"dummy_step_1": DummyStep()})
        self.assertEqual(len(pipeline.steps), 1)
        _ = pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        pipeline.addStep("dummy_step_2", DummyStep())
        self.assertEqual(len(pipeline.steps), 2)
        self.assertListEqual(
            list(pipeline.steps.keys()), ["dummy_step_1", "dummy_step_2"]
        )
        _ = pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        
    
    def testRemoveStep(self):
        """Test the pipeline remove step method."""
        pipeline = Pipeline(steps={"dummy_step_1": DummyStep(), "dummy_step_2": DummyStep()})
        self.assertEqual(len(pipeline.steps), 2)
        _ = pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)

        # remove a step
        pipeline.removeStep("dummy_step_1")
        self.assertEqual(len(pipeline.steps), 1)
        self.assertIn("dummy_step_2", pipeline.steps)
        _ = pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
    
    def testOrderSteps(self):
        """Test the pipeline order steps method."""
        pipeline = Pipeline(steps={
            "dummy_step_1": DummyStep(),
            "dummy_step_2": DummyStep(),
            "dummy_step_3": DummyStep(),
        })
        self.assertEqual(len(pipeline.steps), 3)
        _ = pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)

        # order steps
        pipeline.orderSteps(["dummy_step_3", "dummy_step_1", "dummy_step_2"])
        self.assertListEqual(
            list(pipeline.steps.keys()), ["dummy_step_3", "dummy_step_1", "dummy_step_2"]
        )
        _ = pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
    
    def testSkipping(self):
        """Test the pipeline skipping steps."""
        pipeline = Pipeline(steps={
            "dummy_step_1": DummyStep(),
            "scaler": SKLearnStandardizer(StandardScaler()),
            "dummy_step_2": DummyStep(),
        }, skip=["scaler"])
        self.assertEqual(len(pipeline.steps), 3)
        self.assertIn("scaler", pipeline.skip)
        X_train_out, _, _, _ = pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertTrue(X_train_out.equals(self.X_train))
        
        pipeline.addSkip("dummy_step_1")
        self.assertIn("dummy_step_1", pipeline.skip)
        self.assertIn("scaler", pipeline.skip)
        
        pipeline.removeSkip("dummy_step_1")
        self.assertNotIn("dummy_step_1", pipeline.skip)
        self.assertIn("scaler", pipeline.skip)

    def testSerialization(self):
        """Test the pipeline serialization."""
        pipeline = Pipeline(steps={"dummy_step": DummyStep()}, seed=42)
        pipeline.toFile(f"{self.generatedPath}/test_pipeline.json")
        pipeline_fromfile = Pipeline.fromFile(f"{self.generatedPath}/test_pipeline.json")
        self.assertIsInstance(pipeline_fromfile, Pipeline)
        self.assertEqual(pipeline_fromfile.randomState, pipeline.randomState)
        self.assertEqual(len(pipeline_fromfile.steps), len(pipeline.steps))
        self.assertIn("dummy_step", pipeline_fromfile.steps)
        self.assertIsInstance(pipeline_fromfile.steps["dummy_step"], DummyStep)
    
class TestDatasetPipeline(DataSetsPathMixIn, QSPRTestCase):
    """Test the dataset pipeline."""

    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([MorganFP(radius=2, nBits=20)])

    def testApply(self):
        """Test the dataset pipeline apply method."""
        # TODO
        pass

#####-----------------Test Pipeline Steps-----------------#####
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
            prop="moka_ionState7.4",
            values=["cationic"],
            data_set=self.dataset,
        )
        filtered_df, _ = remove_cation.transform(self.dataset.getDF())
        self.assertTrue((filtered_df["moka_ionState7.4"] == "cationic").sum() == 0)

        # Test with keep=True
        only_cation = CategoryFilter(
            prop="moka_ionState7.4",
            values=["cationic"],
            data_set=self.dataset,
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
        remove_cation = CategoryFilter(
            prop="moka_ionState7.4",
            values=["cationic"],
            data_set=dataset
        )
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() > 0)
        dataset.filter([remove_cation])
        self.assertEqual(len(dataset.getDF()), len(dataset.getDescriptors()))
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
        # create example dataset
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
        self.descriptor_names = [
            "Descriptor_F1",
            "Descriptor_F2",
            "Descriptor_F3",
            "Descriptor_F4",
            "Descriptor_F5",
        ]
        # create example descriptors and add them to the dataset
        self.example_descriptors = pd.DataFrame(
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
            columns=[
                "Descriptor_F1",
                "Descriptor_F2",
                "Descriptor_F3",
                "Descriptor_F4",
                "Descriptor_F5",
            ],
        )
        self.example_descriptors[self.dataset.idProp] = list(
            self.dataset.getProperty(self.dataset.idProp)
        )
        self.example_descriptors.set_index(self.dataset.idProp, inplace=True, drop=True)
        self.dataset.addDescriptors([DataFrameDescriptorSet(self.example_descriptors)])

    def recalculateWithMultiIndex(self):
        """Change the dataset to have a multi-index."""
        self.dataset.dropDescriptorSets(self.dataset.descriptorSets, full_removal=True)
        self.example_descriptors["ID_COL1"] = (
            self.dataset.getProperty(self.dataset.idProp
                                    ).apply(lambda x: x.split("_")[0]).to_list()
        )
        self.example_descriptors["ID_COL2"] = (
            self.dataset.getProperty(self.dataset.idProp
                                    ).apply(lambda x: x.split("_")[-1]).to_list()
        )
        self.dataset.addProperty("ID_COL1", self.example_descriptors["ID_COL1"].values)
        self.dataset.addProperty("ID_COL2", self.example_descriptors["ID_COL2"].values)
        self.dataset.addDescriptors(
            [DataFrameDescriptorSet(
                self.example_descriptors,
                ["ID_COL1", "ID_COL2"],
            )]
        )

    def testDefaultDescriptorAdd(self):
        """Test adding without index columns."""
        self.dataset.nJobs = 1
        df_new = self.dataset.getDescriptors().copy()
        calc = DataFrameDescriptorSet(df_new, suffix="new_df_desc")
        self.dataset.addDescriptors([calc])

    @parameterized.expand([
        (True, ),
        (False, ),
    ])
    def testLowVarianceFilter(self, use_index_cols):
        """Test the low variance filter, which drops features with a variance below
        a threshold.
        
        Args:
            use_index_cols (bool): If True, a multi-index is used for the dataset.
        """
        if use_index_cols:
            self.recalculateWithMultiIndex()

        pipeline = DatasetPipeline(
            steps={
                "low_var_filter": LowVarianceFilter(0.01),
            }
        )
        X, y = next(pipeline.apply(self.dataset))
        # check if first column (no variance) is dropped
        self.assertListEqual(X.columns.tolist(), self.dataset.getDescriptorNames()[1:])
        # check y is still the same
        self.assertListEqual(y.columns.tolist(), self.dataset.getTargets().columns.tolist())

    @parameterized.expand([
        (True, ),
        (False, ),
    ])
    def testHighCorrelationFilter(self, use_index_cols):
        """Test the high correlation filter, which drops features with a correlation
        above a threshold."""
        if use_index_cols:
            self.recalculateWithMultiIndex()

        pipeline = DatasetPipeline(
            steps={
                "high_corr_filter": HighCorrelationFilter(0.8),
            }
        )
        X, y = next(pipeline.apply(self.dataset))
        # check if "Descriptor_F3" (correlated to "Descriptor_F2") is dropped
        desc_to_keep = self.dataset.getDescriptorNames()
        desc_to_keep.remove("DataFrame_Descriptor_F3")
        self.assertListEqual(X.columns.tolist(), desc_to_keep)
        # check y is still the same
        self.assertListEqual(y.columns.tolist(), self.dataset.getTargets().columns.tolist())


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
        pipeline = DatasetPipeline(
            steps={
                "boruta_filter": BorutaFilter(),
            }
        )
        X, y = next(pipeline.apply(self.dataset))
        # check if only "Descriptor_F5" is kept (increases with target)
        self.assertListEqual(X.columns.tolist(), self.dataset.getDescriptorNames()[-1:])
        # check y is still the same
        self.assertListEqual(y.columns.tolist(), self.dataset.getTargets().columns.tolist())


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
        scaled_features, _ = scaler.fitTransform(self.dataset.getDescriptors())
        scaler.toFile(f"{self.generatedPath}/test_scaler.json")
        scaler_fromfile = SKLearnStandardizer.fromFile(
            f"{self.generatedPath}/test_scaler.json"
        )
        scaled_features_fromfile, _ = scaler_fromfile.transform(self.dataset.getDescriptors())
        self.assertIsInstance(scaled_features, pd.DataFrame)
        self.assertEqual(scaled_features.shape, (len(self.dataset), 128))
        self.assertEqual(
            np.array_equal(scaled_features, scaled_features_fromfile), True
        )
import itertools

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
from .feature_transformers import SklearnStep
from .pipeline import DatasetPipeline, Pipeline, DummyStep, Shuffle
from ...data.tables.qspr import QSPRTable
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn
from ...utils.testing.check_mixins import StepCheckMixIn
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
        self.dataset.addDescriptors([MorganFP(radius=3, nBits=100)])

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
        """Create a small test dataset with random descriptors."""
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
        pipeline = Pipeline(steps={"scaler": SklearnStep(Binarizer())})
        self.assertFalse(pipeline.steps["scaler"].fitted)
        self.assertFalse(pipeline.fitted)
        with self.assertRaises(ValueError):
            # test that the pipeline raises an error if fit is not called
            pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test, fit=False)
        
        pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertTrue(pipeline.steps["scaler"].fitted)
        self.assertTrue(pipeline.fitted)
        
        # test if the input data is not modified
        pipeline = Pipeline(steps={"scaler": SklearnStep(StandardScaler())})
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
        pipeline = Pipeline(steps={"scaler": SklearnStep(Binarizer())}, fixed=["scaler"])
        self.assertEqual(pipeline.fixed, ["scaler"])
        
        pipeline.apply(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertFalse(pipeline.steps["scaler"].fitted)
        self.assertTrue(pipeline.fitted)
    
    def testApplyWithFitOn(self):
        """Test the pipeline apply method with fit_on argument."""
        # test with fit_on="test"
        test_means = self.X_test.mean().to_list()
        pipeline = Pipeline(
            steps={"scaler": SklearnStep(StandardScaler())},
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
            steps={"scaler": SklearnStep(StandardScaler())},
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
            steps={"scaler": SklearnStep(StandardScaler())},
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
            steps={"scaler": SklearnStep(StandardScaler())},
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
            "scaler": SklearnStep(StandardScaler()),
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
        
        # test if apply works after serialization
        X_train, y_train, X_test, y_test = pipeline.apply(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        X_train_fromfile, y_train_fromfile, X_test_fromfile, y_test_fromfile = pipeline_fromfile.apply(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        self.assertIsInstance(X_train_fromfile, pd.DataFrame)
        self.assertIsInstance(y_train_fromfile, pd.DataFrame)
        self.assertIsInstance(X_test_fromfile, pd.DataFrame)
        self.assertIsInstance(y_test_fromfile, pd.DataFrame)
        self.assertTrue(X_train.equals(X_train_fromfile))
        self.assertTrue(y_train.equals(y_train_fromfile))
        self.assertTrue(X_test.equals(X_test_fromfile))
        self.assertTrue(y_test.equals(y_test_fromfile))
    
class TestDatasetPipeline(DataSetsPathMixIn, QSPRTestCase):
    """Test the dataset pipeline."""

    def setUp(self):
        """Create a small test dataset for the dataset pipeline."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([RandomDescs(n=10, seed=42)])

    def testApply(self):
        """Test the dataset pipeline apply method."""
        pipeline = DatasetPipeline(steps={"dummy_step": DummyStep()})
        self.assertIsInstance(pipeline, DatasetPipeline)
        self.assertEqual(len(pipeline.steps), 1)
        self.assertIn("dummy_step", pipeline.steps)

        # Test apply with a dummy step
        X, y = next(pipeline.apply(self.dataset))
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.DataFrame)
        self.assertEqual(X.shape[0], len(self.dataset))
        self.assertEqual(y.shape[0], len(self.dataset))

        # test apply with a data split
        X_train, y_train, X_test, y_test = next(
            pipeline.apply(self.dataset, split=RandomSplit(seed=42))
        )
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.DataFrame)
        self.assertEqual(X_train.shape[0] + X_test.shape[0], len(self.dataset))
        self.assertEqual(y_train.shape[0] + y_test.shape[0], len(self.dataset))
        self.assertEqual(X_train.shape[1], self.dataset.getDescriptors().shape[1])
        self.assertEqual(y_train.shape[1], self.dataset.getTargets().shape[1])
        
        # Test apply with no steps
        pipeline = DatasetPipeline(steps={})
        self.assertEqual(len(pipeline.steps), 0)
        X, y = next(pipeline.apply(self.dataset))
        self.assertTrue(X.equals(self.dataset.getDescriptors()))
        self.assertTrue(y.equals(self.dataset.getTargets()))
        
        # test pipeline with feature calculators
        pipeline = DatasetPipeline(
            feature_calculators=[MorganFP(radius=2, nBits=20)],
            steps={"dummy_step": DummyStep()}
        )
        X, y = next(pipeline.apply(self.dataset))
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.DataFrame)
        self.assertEqual(X.shape[0], len(self.dataset))
        self.assertEqual(y.shape[0], len(self.dataset))
        self.assertEqual(X.shape[1], 10 + 20)  # 10 random descriptors + 20 MorganFP bits
        self.assertEqual(y.shape[1], self.dataset.getTargets().shape[1])
        
        # test if an error is raised if the dataset misses expected descriptors
        self.dataset.dropDescriptorSets(["RandomDesc(10)"])
        self.assertEqual(len(pipeline.originalfeatureNames), 10 + 20)
        with self.assertRaises(AssertionError):
            print(self.dataset.getDescriptors().columns.tolist())
            _ = next(pipeline.apply(self.dataset, fit=False))
        
        
    def testSerialization(self):
        """Test the dataset pipeline serialization."""
        pipeline = DatasetPipeline(
            feature_calculators=[MorganFP(radius=2, nBits=20)],
            steps={
                "dummy_step": DummyStep(),
            }
        )
        pipeline.toFile(f"{self.generatedPath}/test_dataset_pipeline.json")
        pipeline_fromfile = DatasetPipeline.fromFile(
            f"{self.generatedPath}/test_dataset_pipeline.json"
        )
        self.assertIsInstance(pipeline_fromfile, DatasetPipeline)
        self.assertEqual(len(pipeline_fromfile.steps), len(pipeline.steps))
        self.assertIn("dummy_step", pipeline_fromfile.steps)
        self.assertIsInstance(pipeline_fromfile.steps["dummy_step"], DummyStep)
        self.assertEqual(
            len(pipeline_fromfile.feature_calculators),
            len(pipeline.feature_calculators)
        )
        
        X, y = next(pipeline.apply(self.dataset))
        X_fromfile, y_fromfile = next(pipeline_fromfile.apply(self.dataset))
        self.assertIsInstance(X_fromfile, pd.DataFrame)
        self.assertIsInstance(y_fromfile, pd.DataFrame)
        self.assertTrue(X.equals(X_fromfile))
        self.assertTrue(y.equals(y_fromfile))

#####-----------------Test Pipeline Steps-----------------#####
class TestDummyStep(QSPRTestCase, StepCheckMixIn):
    """Test the dummy step"""
    def setUp(self):
        """Create a small test dataset with random descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([RandomDescs(n=10, seed=42)])

    def testDummyStep(self):
        """Test the dummy step."""
        X_out, y_out = self.checkStep(DummyStep(), self.dataset)
        self.assertTrue(X_out.equals(self.dataset.getDescriptors()))
        self.assertTrue(y_out.equals(self.dataset.getTargets()))

class TestShuffle(QSPRTestCase, StepCheckMixIn):
    """Test the shuffle step in the pipeline."""
    def setUp(self):
        """Create a small test dataset with random descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([RandomDescs(n=10, seed=42)])

    def testShuffle(self):
        """Test the shuffle step."""
        X_out, y_out = self.checkStep(Shuffle(42), self.dataset)
        
        # check if the output of the step is a shuffled version of the input
        self.assertEqual(X_out.shape, self.dataset.getDescriptors().shape)
        self.assertEqual(y_out.shape, self.dataset.getTargets().shape)
        self.assertFalse(X_out.equals(self.dataset.getDescriptors()))
        self.assertTrue(
            X_out.sort_values(by=self.dataset.idProp).equals(
                self.dataset.getDescriptors().sort_values(by=self.dataset.idProp)
            )
        )
        
        # check if the random state is set correctly
        X_out_same, y_out_same = self.checkStep(Shuffle(42), self.dataset)
        self.assertTrue(X_out.equals(X_out_same))
        self.assertTrue(y_out.equals(y_out_same))
        X_out_diff, y_out_diff = self.checkStep(Shuffle(43), self.dataset)
        self.assertFalse(X_out.equals(X_out_diff))
        self.assertFalse(y_out.equals(y_out_diff))

class TestDataFilters(QSPRTestCase, StepCheckMixIn):
    """Test the data filters, which filter the dataset based on properties."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([MorganFP(radius=2, nBits=20)])

    def testCategoryFilter(self):
        """Test the category filter that drops values from a dataset property."""
        self.assertTrue(
            (self.dataset.getDF()["moka_ionState7.4"] == "cationic").sum() > 0
        )
        
        # Test with keep=False
        remove_cation = CategoryFilter(
            prop="moka_ionState7.4",
            values=["cationic"],
            data_set=self.dataset,
        )
        X_filtered, _ = self.checkStep(remove_cation, self.dataset)
        filtered_df = self.dataset.getDF().loc[X_filtered.index]
        self.assertTrue((filtered_df["moka_ionState7.4"] == "cationic").sum() == 0)

        # Test with keep=True
        only_cation = CategoryFilter(
            prop="moka_ionState7.4",
            values=["cationic"],
            data_set=self.dataset,
            keep=True
        )
        X_filtered, _ = self.checkStep(only_cation, self.dataset)
        filtered_df = self.dataset.getDF().loc[X_filtered.index]
        self.assertTrue((filtered_df["moka_ionState7.4"] != "cationic").sum() == 0)

    def testRepeatsFilter(self):
        """Test the duplicate filter, which drops rows with identical descriptors
        from dataset."""
        ## check assumptions about the test data
        # check that the descriptor rows 0, 3 and 5 are identical
        descriptors = self.dataset.getDescriptors()
        self.assertTrue(np.array_equal(descriptors.iloc[0], descriptors.iloc[3]))
        self.assertTrue(np.array_equal(descriptors.iloc[0], descriptors.iloc[5]))
        self.assertEqual(
            len(descriptors.drop_duplicates(keep=False)), len(descriptors)-3
        )

        ## test the filter
        # only warnings
        warn_reps = RepeatsFilter(keep=True, data_set=self.dataset)
        X_filtered, _ = self.checkStep(warn_reps, self.dataset)
        self.assertEqual(len(X_filtered), len(descriptors))
        self.assertTrue(X_filtered.equals(descriptors))

        # drop duplicates
        drop_reps = RepeatsFilter(keep=False, data_set=self.dataset)
        X_filtered, _ = self.checkStep(drop_reps, self.dataset)
        self.assertEqual(len(X_filtered), len(descriptors)-3)
        
        # keep first, by year
        keep_first = RepeatsFilter(
            keep="first", timecol="Year of first disclosure", data_set=self.dataset
        )
        X_filtered, _ = self.checkStep(keep_first, self.dataset)
        self.assertEqual(len(X_filtered), len(descriptors)-2)
        self.assertIn(descriptors.iloc[0].name, X_filtered.index)

        # check with additional columns 
        proteinid = ["A", "B", "B", "A", "B", "B", "B", "B", "B"]
        self.dataset.addProperty("proteinid", pd.Series(proteinid, index=descriptors.index))
        drop_reps_protein = RepeatsFilter(
            keep=False,
            additional_cols=["proteinid"],
            data_set=self.dataset
        )
        X_filtered, _ = self.checkStep(drop_reps_protein, self.dataset)
        self.assertEqual(len(X_filtered), len(descriptors)-2)

    def testFilterMethodOfDataset(self):
        # TODO: Either this functionality should be removed or the test should be moved
        # to the dataset tests, as it is not a step in the pipeline.
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

class TestFeatureFilters(QSPRTestCase, StepCheckMixIn):
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

        low_var_filter = LowVarianceFilter(0.01)
        X, y = self.checkStep(low_var_filter, self.dataset)
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

        high_corr_filter = HighCorrelationFilter(0.8)
        X, y = self.checkStep(high_corr_filter, self.dataset)
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
        boruta_filter = BorutaFilter()
        X, y = self.checkStep(boruta_filter, self.dataset)
        # check if only "Descriptor_F5" is kept (increases with target)
        self.assertListEqual(X.columns.tolist(), self.dataset.getDescriptorNames()[-1:])
        # check y is still the same
        self.assertListEqual(y.columns.tolist(), self.dataset.getTargets().columns.tolist())
        
class TestFeatureTransformers(QSPRTestCase, StepCheckMixIn):
    """Test the sklearn step which wraps a sklearn transformer."""
    def setUp(self):
        """Create a small test dataset with random descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors([RandomDescs(n=10, seed=42)])

    def testSklearnStep(self):
        """Test the sklearn step."""
        X_out, y_out = self.checkStep(SklearnStep(StandardScaler()), self.dataset)
        
        # check if the output of the step is equal to directly applying a sklearn scaler
        X = self.dataset.getDescriptors()
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(X)
        self.assertTrue(np.allclose(X_out.values, X_transformed))
        self.assertTrue(y_out.equals(self.dataset.getTargets()))
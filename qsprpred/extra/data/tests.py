import itertools
import os
from typing import Callable, Type
from unittest import TestCase

import numpy as np
import pandas as pd
from parameterized import parameterized
from sklearn.preprocessing import StandardScaler

from ...data.data import TargetProperty
from ...data.interfaces import DataSplit
from ...data.tests import (
    CHUNK_SIZE,
    N_CPU,
    DataPrepTestMixIn,
    DataSetsMixIn,
    TestDescriptorInDataMixIn,
)
from ...data.utils.datasplitters import ClusterSplit, RandomSplit, ScaffoldSplit
from ...data.utils.descriptorcalculator import (
    DescriptorsCalculator,
    MoleculeDescriptorsCalculator,
)
from ...data.utils.descriptorsets import (
    DrugExPhyschem,
    FingerprintSet,
    MoleculeDescriptorSet,
    RDKitDescs,
)
from ...data.utils.feature_standardization import SKLearnStandardizer
from ...data.utils.featurefilters import HighCorrelationFilter, LowVarianceFilter
from ...extra.data.data import PCMDataSet
from ...extra.data.utils.descriptor_utils.msa_calculator import (
    MAFFT,
    BioPythonMSA,
    ClustalMSA,
)
from ...extra.data.utils.descriptorcalculator import ProteinDescriptorCalculator
from ...extra.data.utils.descriptorsets import (
    ExtendedValenceSignature,
    Mold2,
    Mordred,
    PaDEL,
    ProDec,
    ProteinDescriptorSet,
)
from ...models.tasks import TargetTasks
from .utils.datasplitters import LeaveTargetsOut, PCMSplit, TemporalPerTarget


class DataSetsMixInExtras(DataSetsMixIn):
    """MixIn class for testing data sets in extras."""
    def setUp(self):
        super().setUp()
        self.dataPathPCM = f"{os.path.dirname(__file__)}/test_files/data"

    @classmethod
    def getAllDescriptors(cls) -> list[MoleculeDescriptorSet]:
        """Return a list of all available molecule descriptor sets.

        Returns:
            list: list of `MoleculeDescriptorSet` objects
        """
        return [
            Mordred(),
            Mold2(),
            FingerprintSet(fingerprint_type="CDKFP", size=2048, search_depth=7),
            FingerprintSet(fingerprint_type="CDKExtendedFP"),
            FingerprintSet(fingerprint_type="CDKEStateFP"),
            FingerprintSet(
                fingerprint_type="CDKGraphOnlyFP", size=2048, search_depth=7
            ),
            FingerprintSet(fingerprint_type="CDKMACCSFP"),
            FingerprintSet(fingerprint_type="CDKPubchemFP"),
            FingerprintSet(fingerprint_type="CDKSubstructureFP", use_counts=False),
            FingerprintSet(fingerprint_type="CDKKlekotaRothFP", use_counts=True),
            FingerprintSet(fingerprint_type="CDKAtomPairs2DFP", use_counts=False),
            FingerprintSet(fingerprint_type="CDKSubstructureFP", use_counts=True),
            FingerprintSet(fingerprint_type="CDKKlekotaRothFP", use_counts=False),
            FingerprintSet(fingerprint_type="CDKAtomPairs2DFP", use_counts=True),
            PaDEL(),
            ExtendedValenceSignature(1),
        ]

    @staticmethod
    def getAllProteinDescriptors() -> list[ProteinDescriptorSet]:
        """Return a list of all available protein descriptor sets.

        Returns:
            list: list of `ProteinDescriptorSet` objects
        """

        return [
            ProDec(sets=["Zscale Hellberg"]),
            ProDec(sets=["Sneath"]),
        ]

    @classmethod
    def getDefaultCalculatorCombo(cls):
        mol_descriptor_calculators = super().getDefaultCalculatorCombo()
        feature_sets_pcm = [
            ProDec(sets=["Zscale Hellberg"]),
            ProDec(sets=["Sneath"]),
        ]
        protein_descriptor_calculators = [
            [ProteinDescriptorCalculator(combo, msa_provider=cls.getMSAProvider())]
            for combo in itertools.combinations(feature_sets_pcm, 1)
        ] + [
            [ProteinDescriptorCalculator(combo, msa_provider=cls.getMSAProvider())]
            for combo in itertools.combinations(feature_sets_pcm, 2)
        ]
        descriptor_calculators = (
            mol_descriptor_calculators + protein_descriptor_calculators
        )
        # make combinations of molecular and PCM descriptor calculators
        descriptor_calculators += [
            mol + prot for mol, prot in
            zip(mol_descriptor_calculators, protein_descriptor_calculators)
        ]

        return descriptor_calculators

    def getPCMDF(self) -> pd.DataFrame:
        """Return a test dataframe with PCM data.

        Returns:
            pd.DataFrame: dataframe with PCM data
        """
        return pd.read_csv(f"{self.dataPathPCM}/pcm_sample.csv")

    def getPCMTargetsDF(self) -> pd.DataFrame:
        """Return a test dataframe with PCM targets and their sequences.

        Returns:
            pd.DataFrame: dataframe with PCM targets and their sequences
        """
        return pd.read_csv(f"{self.dataPathPCM}/pcm_sample_targets.csv")

    def getPCMSeqProvider(
        self,
    ) -> Callable[[list[str]], tuple[dict[str, str], dict[str, dict]]]:
        """Return a function that provides sequences for given accessions.

        Returns:
            Callable[[list[str]], tuple[dict[str, str], dict[str, dict]]]:
                function that provides sequences for given accessions
        """
        df_seq = self.getPCMTargetsDF()
        mapper = {}
        kwargs_map = {}
        for i, row in df_seq.iterrows():
            mapper[row["accession"]] = row["Sequence"]
            kwargs_map[row["accession"]] = {
                "Classification": row["Classification"],
                "Organism": row["Organism"],
                "UniProtID": row["UniProtID"],
            }

        return lambda acc_keys: (
            {
                acc: mapper[acc]
                for acc in acc_keys
            },
            {
                acc: kwargs_map[acc]
                for acc in acc_keys
            },
        )

    def getMSAProvider(self):
        return ClustalMSA(out_dir=self.generatedDataPath)

    def createPCMDataSet(
        self,
        name: str = "QSPRDataset_test_pcm",
        target_props: TargetProperty | list[dict] = [
            {
                "name": "pchembl_value_Median",
                "task": TargetTasks.REGRESSION
            }
        ],
        target_imputer: Callable[[pd.Series], pd.Series] | None = None,
        preparation_settings: dict | None = None,
        protein_col: str = "accession",
    ):
        """Create a small dataset for testing purposes.

        Args:
            name (str, optional):
                name of the dataset. Defaults to "QSPRDataset_test".
            target_props (list[TargetProperty] | list[dict], optional):
                target properties.
            target_imputer (Callable[pd.Series, pd.Series] | None, optional):
                target imputer. Defaults to `None`.
            preparation_settings (dict | None, optional):
                preparation settings. Defaults to None.
            protein_col (str, optional):
                name of the column with protein accessions. Defaults to "accession".
        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        df = self.getPCMDF()
        ret = PCMDataSet(
            name,
            protein_col=protein_col,
            protein_seq_provider=self.getPCMSeqProvider(),
            target_props=target_props,
            df=df,
            store_dir=self.generatedDataPath,
            target_imputer=target_imputer,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        if preparation_settings:
            ret.prepareDataset(**preparation_settings)
        return ret


class TestDescriptorSetsExtra(DataSetsMixInExtras, TestCase):
    """Test descriptor sets with extra features.

    Attributes:
        dataset (QSPRDataset): dataset for testing, shuffled
    """
    def setUp(self):
        super().setUp()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.shuffle()

    def testMold2(self):
        """Test the Mold2 descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator([Mold2()])
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 777))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testPaDELDescriptors(self):
        """Test the PaDEL descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator([PaDEL()])
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1444))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    @parameterized.expand(
        [
            ("CDKFP", 1024),
            ("CDKExtendedFP", 1024),
            ("CDKGraphOnlyFP", 1024),
            ("CDKMACCSFP", 166),
            ("CDKPubchemFP", 881),
            ("CDKEStateFP", 79),
            ("CDKSubstructureFP", 307),
            ("CDKKlekotaRothFP", 4860),
            ("CDKAtomPairs2DFP", 780),
        ]
    )
    def testPaDELFingerprints(self, fp_type, nbits):
        desc_calc = MoleculeDescriptorsCalculator(
            [FingerprintSet(fingerprint_type=fp_type)]
        )
        dataset = self.createSmallTestDataSet(f"{self.__class__.__name__}_{fp_type}")
        dataset.addDescriptors(desc_calc)
        self.assertEqual(dataset.X.shape, (len(dataset), nbits))
        self.assertTrue(dataset.X.any().any())
        self.assertTrue(dataset.X.any().sum() > 1)

    def testMordred(self):
        """Test the Mordred descriptor calculator."""
        import mordred
        from mordred import descriptors

        desc_calc = MoleculeDescriptorsCalculator([Mordred()])
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(
            self.dataset.X.shape,
            (
                len(self.dataset),
                len(mordred.Calculator(descriptors).descriptors),
            ),
        )
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testExtendedValenceSignature(self):
        """Test the SMILES based signature descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator([ExtendedValenceSignature(1)])
        self.dataset.addDescriptors(desc_calc, recalculate=True)
        self.dataset.featurize()
        self.assertTrue(self.dataset.X.shape[1] > 0)
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)


class TestPCMDescriptorCalculation(DataSetsMixInExtras, TestCase):
    """Test the calculation of protein descriptors.

    Attributes:
        dataset (QSPRDataset): dataset for testing
        sampleDescSet (DescriptorSet): descriptor set for testing
        defaultMSA (BioPythonMSA): MSA provider for testing
    """
    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.dataset = self.createPCMDataSet(self.__class__.__name__)
        self.sampleDescSet = ProDec(sets=["Zscale Hellberg"])
        self.defaultMSA = self.getMSAProvider()

    @parameterized.expand([
        ("MAFFT", MAFFT),
        ("ClustalMSA", ClustalMSA),
    ])
    def testSerialization(self, _, msa_provider_cls: Type[BioPythonMSA]):
        """Test the serialization of dataset with data split.

        Args:
            msa_provider_cls (BioPythonMSA): MSA provider class
        """
        provider = msa_provider_cls(out_dir=self.generatedDataPath)
        dataset = self.createPCMDataSet(self.__class__.__name__)
        split = RandomSplit(test_fraction=0.2)
        calculator = ProteinDescriptorCalculator([self.sampleDescSet], provider)
        dataset.prepareDataset(
            split=split,
            feature_calculators=[calculator],
            feature_standardizer=StandardScaler(),
            feature_filters=[LowVarianceFilter(0.05),
                             HighCorrelationFilter(0.9)],
        )
        ndata = dataset.getDF().shape[0]
        self.validate_split(dataset)
        self.assertEqual(dataset.X_ind.shape[0], round(ndata * 0.2))
        test_ids = dataset.X_ind.index.values
        train_ids = dataset.y_ind.index.values
        dataset.save()
        # load dataset and test if all checks out after loading
        dataset_new = PCMDataSet.fromFile(dataset.storePath)
        self.assertIsInstance(dataset_new, PCMDataSet)
        self.validate_split(dataset_new)
        self.assertEqual(dataset.X_ind.shape[0], round(ndata * 0.2))
        self.assertEqual(
            len(dataset_new.descriptorCalculators), len(dataset_new.descriptors)
        )
        self.assertTrue(dataset_new.feature_standardizer)
        self.assertTrue(dataset_new.fold_generator.featureStandardizer)
        self.assertTrue(
            isinstance(
                dataset_new.fold_generator.featureStandardizer, SKLearnStandardizer
            ) and isinstance(dataset_new.feature_standardizer, SKLearnStandardizer)
        )
        self.assertTrue(len(dataset_new.featureNames) == len(self.sampleDescSet))
        self.assertTrue(all(mol_id in dataset_new.X_ind.index for mol_id in test_ids))
        self.assertTrue(all(mol_id in dataset_new.y_ind.index for mol_id in train_ids))
        # clear files and try saving again
        dataset_new.clearFiles()
        dataset_new.save()

    def testSwitching(self):
        """Test if the feature calculator can be switched to a new dataset."""
        dataset = self.createPCMDataSet(self.__class__.__name__)
        split = RandomSplit(test_fraction=0.5)
        calculator = ProteinDescriptorCalculator([self.sampleDescSet], self.defaultMSA)
        lv = LowVarianceFilter(0.05)
        hc = HighCorrelationFilter(0.9)
        dataset.prepareDataset(
            split=split,
            feature_calculators=[calculator],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )
        ndata = dataset.getDF().shape[0]
        self.assertEqual(len(dataset.descriptorCalculators), len(dataset.descriptors))
        self.assertEqual(
            dataset.X_ind.shape, (round(ndata * 0.5), len(self.sampleDescSet))
        )
        # create new dataset with different feature calculator
        dataset_next = self.createPCMDataSet(f"{self.__class__.__name__}_next")
        dataset_next.prepareDataset(
            split=split,
            feature_calculators=[calculator],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )
        self.assertEqual(
            len(dataset_next.descriptorCalculators), len(dataset_next.descriptors)
        )
        self.assertEqual(
            dataset_next.X_ind.shape, (round(ndata * 0.5), len(self.sampleDescSet))
        )
        self.assertEqual(
            dataset_next.X.shape, (round(ndata * 0.5), len(self.sampleDescSet))
        )

    def testWithMolDescriptors(self):
        """Test the calculation of protein and molecule descriptors."""
        protein_feature_calculator = ProteinDescriptorCalculator(
            desc_sets=[ProDec(sets=["Zscale Hellberg"])],
            msa_provider=self.defaultMSA,
        )
        mol_feature_calculator = MoleculeDescriptorsCalculator(
            desc_sets=[
                FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048),
                DrugExPhyschem(),
            ]
        )
        calcs = [protein_feature_calculator, mol_feature_calculator]
        self.dataset.prepareDataset(
            feature_calculators=calcs,
            feature_standardizer=StandardScaler(),
            split=RandomSplit(test_fraction=0.2),
        )
        # test if all descriptors are there
        expected_length = 0
        for calc in calcs:
            for descset in calc.descSets:
                expected_length += len(descset)
        self.assertEqual(self.dataset.X.shape[1], expected_length)
        # filter features and test if they are there after saving and loading
        self.dataset.filterFeatures(
            [LowVarianceFilter(0.05),
             HighCorrelationFilter(0.9)]
        )
        feats_left = self.dataset.X.shape[1]
        self.dataset.save()
        dataset_new = PCMDataSet.fromFile(self.dataset.storePath)
        self.assertEqual(dataset_new.X.shape[1], feats_left)

    @parameterized.expand([
        ("MAFFT", MAFFT),
        ("ClustalMSA", ClustalMSA),
    ])
    def testProDec(self, _, provider_class):
        provider = provider_class(out_dir=self.generatedDataPath)
        descset = ProDec(sets=["Zscale Hellberg"])
        protein_feature_calculator = ProteinDescriptorCalculator(
            desc_sets=[descset],
            msa_provider=provider,
        )
        self.dataset.addProteinDescriptors(calculator=protein_feature_calculator)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), len(descset)))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)


class TestPCMDataSetPreparation(DataSetsMixInExtras, DataPrepTestMixIn, TestCase):
    """Test the preparation of the PCMDataSet."""
    def fetchDataset(self, name: str) -> PCMDataSet:
        """Create a quick dataset with the given name.

        Args:
            name (str): Name of the dataset.

        Returns:
            PCMDataSet: The dataset.
        """
        return self.createPCMDataSet(name)

    @parameterized.expand(
        DataSetsMixInExtras.getPrepCombos()
    )  # add @skip("Not now...") below this line to skip these tests
    def testPrepCombinations(
        self,
        _,
        name: str,
        feature_calculators: list[DescriptorsCalculator],
        split: DataSplit,
        feature_standardizer: SKLearnStandardizer,
        feature_filter: Callable,
        data_filter: Callable,
    ):
        """Test the preparation of the dataset.

        Use different combinations of
        feature calculators, feature standardizers, feature filters and data filters.

        Args:
            name (str): Name of the dataset.
            feature_calculators (list[DescriptorsCalculator]):
                List of feature calculators.
            split (DataSplit): Splitting strategy.
            feature_standardizer (SKLearnStandardizer): Feature standardizer.
            feature_filter (Callable): Feature filter.
            data_filter (Callable): Data filter.
        """
        dataset = self.createPCMDataSet(name=name)
        self.checkPrep(
            dataset,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter,
            ["pchembl_value_Median"],
        )


class TestDescriptorsExtra(DataSetsMixInExtras, TestDescriptorInDataMixIn, TestCase):
    @parameterized.expand(
        [
            (
                f"{desc_set}",
                desc_set,
                [
                    {
                        "name": "CL",
                        "task": TargetTasks.REGRESSION
                    },
                    {
                        "name": "fu",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [0.3]
                    },
                ],
            ) for desc_set in DataSetsMixInExtras.getAllDescriptors()
        ]
    )
    def testDescriptorsExtraAll(
        self,
        _,
        desc_set: MoleculeDescriptorSet,
        target_props: list[dict | TargetProperty],
    ):
        """Test the calculation of extra descriptors with data preparation."""
        np.random.seed(42)
        dataset = self.createLargeTestDataSet(
            name=self.getDatSetName(desc_set, target_props), target_props=target_props
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )


class TestDescriptorsPCM(DataSetsMixInExtras, TestDescriptorInDataMixIn, TestCase):
    """Test the calculation of PCM descriptors with data preparation.

    Attributes:
        defaultMSA (MSAProvider): Default MSA provider.
    """
    def setUp(self):
        super().setUp()
        self.defaultMSA = self.getMSAProvider()

    def getCalculators(self, desc_sets):
        return [
            ProteinDescriptorCalculator(
                desc_sets=desc_sets, msa_provider=self.defaultMSA
            )
        ]

    @parameterized.expand(
        [
            (
                f"{desc_set}_{TargetTasks.MULTICLASS}",
                desc_set,
                [
                    {
                        "name": "pchembl_value_Median",
                        "task": TargetTasks.MULTICLASS,
                        "th": [2.0, 5.5, 6.5, 12.0],
                    }
                ],
            ) for desc_set in DataSetsMixInExtras.getAllProteinDescriptors()
        ] + [
            (
                f"{desc_set}_{TargetTasks.REGRESSION}",
                desc_set,
                [{
                    "name": "pchembl_value_Median",
                    "task": TargetTasks.REGRESSION
                }],
            ) for desc_set in DataSetsMixInExtras.getAllProteinDescriptors()
        ] + [
            (
                f"{desc_set}_Multitask",
                desc_set,
                [
                    {
                        "name": "pchembl_value_Median",
                        "task": TargetTasks.REGRESSION
                    },
                    {
                        "name": "pchembl_value_Mean",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [6.5],
                    },
                ],
            ) for desc_set in DataSetsMixInExtras.getAllProteinDescriptors()
        ]
    )
    def testDescriptorsPCMAll(self, _, desc_set, target_props):
        """Tests all available descriptor sets with data set preparation.

        Note that they are not checked with all possible settings
        and all possible preparations,
        but only with the default settings provided
        by `DataSetsMixIn.getDefaultPrep()`.
        The list itself is defined and configured
        by `DataSetsMixIn.getAllDescriptors()`,
        so if you need a specific descriptor tested, add it there.
        """
        np.random.seed(42)
        dataset = self.createPCMDataSet(
            name=f"{self.getDatSetName(desc_set, target_props)}_pcm",
            target_props=target_props,
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )


class TestPCMSplitters(DataSetsMixInExtras, TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = self.createPCMDataSet(f"{self.__class__.__name__}_test")
        self.dataset.addProteinDescriptors(
            calculator=ProteinDescriptorCalculator(
                desc_sets=[ProDec(sets=["Zscale Hellberg"])],
                msa_provider=self.getMSAProvider(),
            )
        )
        self.dataset.addDescriptors(
            calculator=MoleculeDescriptorsCalculator(desc_sets=[RDKitDescs()])
        )

    @parameterized.expand([(RandomSplit(), ), (ScaffoldSplit(), ), (ClusterSplit(), )])
    def test_PCMSplit(self, splitter):
        splitter = PCMSplit(splitter)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        train, test = train.index, test.index
        test_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[test]
        train_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[train]
        test_smiles = self.dataset.getProperty(self.dataset.smilesCol).loc[test]
        train_smiles = self.dataset.getProperty(self.dataset.smilesCol).loc[train]
        self.assertEqual(len(test_targets), len(test))
        self.assertEqual(len(train_targets), len(train))
        self.assertTrue(
            set(test_smiles.unique()).isdisjoint(set(train_smiles.unique()))
        )

    def test_LeaveTargetOut(self):
        target = self.dataset.getProteinKeys()[0:2]
        splitter = LeaveTargetsOut(targets=target)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        train, test = train.index, test.index
        test_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[test]
        train_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[train]
        self.assertEqual(len(test_targets), len(test))
        self.assertEqual(len(train_targets), len(train))
        self.assertTrue(
            set(test_targets.unique()).isdisjoint(set(train_targets.unique()))
        )

    def test_PerTargetTemporal(self):
        year_col = "Year"
        year = 2015
        splitter = TemporalPerTarget(
            year_col=year_col,
            split_years={key: year
                         for key in self.dataset.getProteinKeys()},
        )
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        self.assertTrue(self.dataset.getDF()[year_col].loc[train.index].max() <= year)
        self.assertTrue(self.dataset.getDF()[year_col].loc[test.index].min() > year)

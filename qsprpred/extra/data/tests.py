"""
tests

Created by: Martin Sicho
On: 12.05.23, 17:46
"""
import itertools
import os
from unittest import TestCase

import numpy as np
import pandas as pd
from parameterized import parameterized
from sklearn.preprocessing import StandardScaler

from qsprpred.data.tests import DataSetsMixIn, N_CPU, CHUNK_SIZE, DataPrepTestMixIn, TestDescriptorInDataMixIn
from qsprpred.data.utils.datasplitters import randomsplit, scaffoldsplit
from qsprpred.data.utils.descriptorcalculator import MoleculeDescriptorsCalculator
from qsprpred.data.utils.descriptorsets import FingerprintSet, DrugExPhyschem, rdkit_descs
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.data.utils.featurefilters import lowVarianceFilter, highCorrelationFilter
from qsprpred.extra.data.data import PCMDataset
from qsprpred.extra.data.utils.datasplitters import LeaveTargetsOut, TemporalPerTarget, \
    StratifiedPerTarget
from qsprpred.extra.data.utils.descriptor_utils.msa_calculator import ClustalMSA, MAFFT
from qsprpred.extra.data.utils.descriptorcalculator import ProteinDescriptorCalculator
from qsprpred.extra.data.utils.descriptorsets import Mordred, Mold2, PaDEL, ProDecDescriptorSet
from qsprpred.models.tasks import TargetTasks


class DataSetsMixInExtras(DataSetsMixIn):
    pcmdatapath = f'{os.path.dirname(__file__)}/test_files/data'
    qsprdatapath = f'{os.path.dirname(__file__)}/test_files/qspr/data'

    @classmethod
    def get_all_descriptors(cls):
        return [
            Mordred(),
            Mold2(),
            FingerprintSet(fingerprint_type="CDKFP", size=2048, searchDepth=7),
            FingerprintSet(fingerprint_type="CDKExtendedFP"),
            FingerprintSet(fingerprint_type="CDKEStateFP"),
            FingerprintSet(fingerprint_type="CDKGraphOnlyFP", size=2048, searchDepth=7),
            FingerprintSet(fingerprint_type="CDKMACCSFP"),
            FingerprintSet(fingerprint_type="CDKPubchemFP"),
            FingerprintSet(fingerprint_type="CDKSubstructureFP", useCounts=False),
            FingerprintSet(fingerprint_type="CDKKlekotaRothFP", useCounts=True),
            FingerprintSet(fingerprint_type="CDKAtomPairs2DFP", useCounts=False),
            FingerprintSet(fingerprint_type="CDKSubstructureFP", useCounts=True),
            FingerprintSet(fingerprint_type="CDKKlekotaRothFP", useCounts=False),
            FingerprintSet(fingerprint_type="CDKAtomPairs2DFP", useCounts=True),
            PaDEL(),
        ]

    @staticmethod
    def get_all_protein_descriptors():
        """Return a list of all available protein descriptor sets."""

        return [
            ProDecDescriptorSet(sets=["Zscale Hellberg"]),
            ProDecDescriptorSet(sets=["Sneath"]),  # TODO: add more?
        ]

    def get_desc_calculators(cls):
        mol_descriptor_calculators = super().get_desc_calculators()
        feature_sets_pcm = [
            ProDecDescriptorSet(sets=["Zscale Hellberg"]),
            ProDecDescriptorSet(sets=["Sneath"]),
        ]
        protein_descriptor_calculators = [
                                             [ProteinDescriptorCalculator(combo, msa_provider=cls.getMSAProvider())] for combo in
                                             itertools.combinations(
                                                 feature_sets_pcm, 1
                                             )] + [
                                             [ProteinDescriptorCalculator(combo, msa_provider=cls.getMSAProvider())] for combo in
                                             itertools.combinations(
                                                 feature_sets_pcm, 2
                                             )
                                         ]
        descriptor_calculators = mol_descriptor_calculators + protein_descriptor_calculators
        # make combinations of molecular and PCM descriptor calculators
        descriptor_calculators += [
            mol + prot for mol, prot in zip(mol_descriptor_calculators, protein_descriptor_calculators)
        ]

        return descriptor_calculators

    def getPCMDF(self):
        return pd.read_csv(f'{self.pcmdatapath}/pcm_sample.csv')

    def getPCMTargetsDF(self):
        return pd.read_csv(f'{self.pcmdatapath}/pcm_sample_targets.csv')

    def getPCMSeqProvider(self):
        df_seq = self.getPCMTargetsDF()
        map = dict()
        kwargs_map = dict()
        for i, row in df_seq.iterrows():
            map[row['accession']] = row['Sequence']
            kwargs_map[row['accession']] = {
                'Classification': row['Classification'],
                'Organism': row['Organism'],
                'UniProtID': row['UniProtID'],
            }

        return lambda acc_keys: ({acc: map[acc] for acc in acc_keys}, {acc: kwargs_map[acc] for acc in acc_keys})

    @classmethod
    def getMSAProvider(cls):
        return ClustalMSA(out_dir=cls.qsprdatapath)

    def create_pcm_dataset(self, name="QSPRDataset_test_pcm", target_props=[
        {"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}], target_imputer=None,
                           preparation_settings=None, proteincol='accession'):
        """Create a small dataset for testing purposes.

        Args:
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            target_imputer (Imputer): imputer to use for missing values of the target property
            preparation_settings (dict): dictionary containing preparation settings
            proteincol (str): name of the column containing the protein identifiers
        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        df = self.getPCMDF()
        ret = PCMDataset(
            name, proteincol=proteincol, proteinseqprovider=self.getPCMSeqProvider(), target_props=target_props, df=df,
            store_dir=self.qsprdatapath, target_imputer=target_imputer, n_jobs=N_CPU, chunk_size=CHUNK_SIZE)
        if preparation_settings:
            ret.prepareDataset(**preparation_settings)
        return ret

class TestDescriptorsetsExtra(DataSetsMixInExtras, TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)
        self.dataset.shuffle()

    def test_Mold2(self):
        """Test the Mold2 descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator([Mold2()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(
            self.dataset.X.shape,
            (len(self.dataset), 777))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def test_PaDEL_descriptors(self):
        """Test the PaDEL descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator([PaDEL()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(
            self.dataset.X.shape,
            (len(self.dataset), 1444))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    @parameterized.expand([
        ("CDKFP", 1024),
        ("CDKExtendedFP", 1024),
        ("CDKGraphOnlyFP", 1024),
        ("CDKMACCSFP", 166),
        ("CDKPubchemFP", 881),
        ("CDKEStateFP", 79),
        ("CDKSubstructureFP", 307),
        ('CDKKlekotaRothFP', 4860),
        ('CDKAtomPairs2DFP', 780)
    ])
    def test_PaDEL_fingerprints(self, fp_type, nbits):
        desc_calc = MoleculeDescriptorsCalculator([FingerprintSet(fingerprint_type=fp_type)])
        dataset = self.create_small_dataset(f"{self.__class__.__name__}_{fp_type}")
        dataset.addDescriptors(desc_calc)

        self.assertEqual(
            dataset.X.shape,
            (len(dataset), nbits))
        self.assertTrue(dataset.X.any().any())
        self.assertTrue(dataset.X.any().sum() > 1)

    def test_Mordred(self):
        """Test the Mordred descriptor calculator."""
        import mordred
        from mordred import descriptors as mordreddescriptors
        desc_calc = MoleculeDescriptorsCalculator([Mordred()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(
            self.dataset.X.shape,
            (len(self.dataset), len(mordred.Calculator(mordreddescriptors).descriptors)))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

class TestPCMDescriptorCalculation(DataSetsMixInExtras, TestCase):

    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.dataset = self.create_pcm_dataset(self.__class__.__name__)
        self.sample_descset = ProDecDescriptorSet(sets=["Zscale Hellberg"])
        self.default_msa_provider = self.getMSAProvider()

    @parameterized.expand(
        [
            ("MAFFT", MAFFT),
            ("ClustalMSA", ClustalMSA),
        ]
    )
    def test_serialization(self, _, msa_provider_cls):
        """Test the serialization of dataset with datasplit."""
        provider = msa_provider_cls(out_dir=self.qsprdatapath)
        dataset = self.create_pcm_dataset(self.__class__.__name__)
        split = randomsplit(test_fraction=0.2)
        calculator = ProteinDescriptorCalculator([self.sample_descset], provider)
        dataset.prepareDataset(
            split=split,
            feature_calculators=[calculator],
            feature_standardizer=StandardScaler(),
            feature_filters=[lowVarianceFilter(0.05), highCorrelationFilter(0.9)],
        )
        self.validate_split(dataset)
        self.assertEqual(dataset.X_ind.shape[0], 4)
        test_ids = dataset.X_ind.index.values
        train_ids = dataset.y_ind.index.values
        dataset.save()

        dataset_new = PCMDataset.fromFile(dataset.storePath)
        self.validate_split(dataset_new)
        self.assertEqual(dataset.X_ind.shape[0], 4)
        self.assertEqual(len(dataset_new.descriptorCalculators),len(dataset_new.descriptors))
        self.assertTrue(dataset_new.feature_standardizer)
        self.assertTrue(dataset_new.fold_generator.featureStandardizer)
        self.assertTrue(isinstance(dataset_new.fold_generator.featureStandardizer, SKLearnStandardizer) and isinstance(dataset_new.feature_standardizer, SKLearnStandardizer))
        self.assertTrue(len(dataset_new.featureNames) == len(self.sample_descset))
        self.assertTrue(all(mol_id in dataset_new.X_ind.index for mol_id in test_ids))
        self.assertTrue(all(mol_id in dataset_new.y_ind.index for mol_id in train_ids))

        dataset_new.clearFiles()
        dataset_new.save()
    def test_switching(self):
        """Test if the feature calculator can be switched to a new dataset."""
        dataset = self.create_pcm_dataset(self.__class__.__name__)
        split = randomsplit(test_fraction=0.5)
        calculator = ProteinDescriptorCalculator([self.sample_descset], self.default_msa_provider)
        lv = lowVarianceFilter(0.05)
        hc = highCorrelationFilter(0.9)

        dataset.prepareDataset(
            split=split,
            feature_calculators=[calculator],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan
        )
        self.assertEqual(len(dataset.descriptorCalculators), len(dataset.descriptors))
        self.assertEqual(dataset.X_ind.shape, (10, len(self.sample_descset)))

        # create new dataset with different feature calculator
        dataset_next = self.create_pcm_dataset(f"{self.__class__.__name__}_next")
        dataset_next.prepareDataset(
            split=split,
            feature_calculators=[calculator],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan
        )
        self.assertEqual(len(dataset_next.descriptorCalculators), len(dataset_next.descriptors))
        self.assertEqual(dataset_next.X_ind.shape, (10, len(self.sample_descset)))
        self.assertEqual(dataset_next.X.shape, (10, len(self.sample_descset)))

    def test_with_mol_descs(self):
        protein_feature_calculator = ProteinDescriptorCalculator(
            descsets=[ProDecDescriptorSet(sets=["Zscale Hellberg"])],
            msa_provider=self.default_msa_provider,
        )
        mol_feature_calculator = MoleculeDescriptorsCalculator(
            descsets=[FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048), DrugExPhyschem()]
        )
        calcs = [protein_feature_calculator, mol_feature_calculator]
        self.dataset.prepareDataset(
            feature_calculators=calcs,
            feature_standardizer=StandardScaler(),
            split=randomsplit(test_fraction=0.2),
        )

        expected_length = 0
        for calc in calcs:
            for descset in calc.descsets:
                expected_length += len(descset)
        self.assertEqual(self.dataset.X.shape[1], expected_length)

        # filter features and test if they are there after saving and loading
        self.dataset.filterFeatures([lowVarianceFilter(0.05), highCorrelationFilter(0.9)])
        feats_left = self.dataset.X.shape[1]
        self.dataset.save()
        dataset_new = PCMDataset.fromFile(self.dataset.storePath)
        self.assertEqual(dataset_new.X.shape[1], feats_left)

    @parameterized.expand(
        [
            ("MAFFT", MAFFT),
            ("ClustalMSA", ClustalMSA),
        ]
    )
    def test_ProDec(self, _, provider_class):
        provider = provider_class(out_dir=self.qsprdatapath)
        descset = ProDecDescriptorSet(sets=["Zscale Hellberg"])
        protein_feature_calculator = ProteinDescriptorCalculator(
            descsets=[descset],
            msa_provider=provider,
        )
        self.dataset.addProteinDescriptors(calculator=protein_feature_calculator)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), len(descset)))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

class TestPCMDataSetPreparation(DataSetsMixInExtras, DataPrepTestMixIn, TestCase):

    def fetchDataset(self, name):
        return self.create_pcm_dataset(name)

    @parameterized.expand(
        DataSetsMixInExtras.get_prep_combinations()
    )  # add @skip("Not now...") below this line to skip these tests
    def test_prep_combinations(
            self,
            _,
            name,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter
    ):
        dataset = self.create_pcm_dataset(name=name)
        self.check_prep(
            dataset,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter,
            ["pchembl_value_Median"]
        )

class TestDescriptorsExtra(DataSetsMixInExtras, TestDescriptorInDataMixIn, TestCase):

    @parameterized.expand(
        # [(f"{desc_set}_{TargetTasks.MULTICLASS}", desc_set,
        #                     [{"name": "CL", "task": TargetTasks.MULTICLASS, "th": [0, 1, 10, 1200]}])
        #                    for desc_set in DataSetsMixInExtras.get_all_descriptors()
        #  ] +
        # [(f"{desc_set}_{TargetTasks.REGRESSION}", desc_set,
        #                     [{"name": "CL", "task": TargetTasks.REGRESSION}])
        #                    for desc_set in DataSetsMixInExtras.get_all_descriptors()
        #  ] +
        [(f"{desc_set}", desc_set,
                            [{"name": "CL", "task": TargetTasks.REGRESSION},
                             {"name": "fu", "task": TargetTasks.SINGLECLASS, "th": [0.3]}])
                           for desc_set in DataSetsMixInExtras.get_all_descriptors()
         ])
    def test_descriptors_extras_all(self, _, desc_set, target_props):
        np.random.seed(42)

        dataset = self.create_large_dataset(
            name=self.get_ds_name(desc_set, target_props),
            target_props=target_props
        )

        self.check_desc_in_dataset(dataset, desc_set, self.get_default_prep(), target_props)

class TestDescriptorsPCM(DataSetsMixInExtras, TestDescriptorInDataMixIn, TestCase):

    def setUp(self):
        super().setUp()
        self.default_msa_provider = self.getMSAProvider()

    def get_calculators(self, desc_sets):
        return [ProteinDescriptorCalculator(descsets=desc_sets, msa_provider=self.default_msa_provider)]

    @parameterized.expand([(f"{desc_set}_{TargetTasks.MULTICLASS}", desc_set,
                            [{"name": "pchembl_value_Median", "task": TargetTasks.MULTICLASS,
                              "th": [5.0, 5.5, 6.5, 10.0]}])
                           for desc_set in DataSetsMixInExtras.get_all_protein_descriptors()] +
                          [(f"{desc_set}_{TargetTasks.REGRESSION}", desc_set,
                            [{"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}])
                           for desc_set in DataSetsMixInExtras.get_all_protein_descriptors()] +
                          [(f"{desc_set}_Multitask", desc_set,
                            [{"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION},
                             {"name": "pchembl_value_Mean", "task": TargetTasks.SINGLECLASS, "th": [6.5]}])
                           for desc_set in DataSetsMixInExtras.get_all_protein_descriptors()])
    def test_descriptors_pcm_all(self, _, desc_set, target_props):
        """Tests all available descriptor sets.

        Note that they are not checked with all possible settings and all possible preparations,
        but only with the default settings provided by `DataSetsMixIn.get_default_prep()`.
        The list itself is defined and configured by `DataSetsMixIn.get_all_descriptors()`,
        so if you need a specific descriptor tested, add it there.
        """
        np.random.seed(42)

        dataset = self.create_pcm_dataset(
            name=f"{self.get_ds_name(desc_set, target_props)}_pcm",
            target_props=target_props
        )

        self.check_desc_in_dataset(dataset, desc_set, self.get_default_prep(), target_props)


class TestSplitsPCM(DataSetsMixInExtras, TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = self.create_pcm_dataset(f"{self.__class__.__name__}_test")
        self.dataset.shuffle()
        self.dataset.addProteinDescriptors(
            calculator=ProteinDescriptorCalculator(
                descsets=[ProDecDescriptorSet(sets=["Zscale Hellberg"])],
                msa_provider=self.getMSAProvider()
            )
        )
        self.dataset.addDescriptors(
            calculator=MoleculeDescriptorsCalculator(
                descsets=[rdkit_descs()]
            )
        )

    def testLeaveTargetOut(self):
        target = self.dataset.getProteinKeys()[0:2]
        splitter = LeaveTargetsOut(dataset=self.dataset, targets=target)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        train, test = train.index, test.index
        test_targets = self.dataset.getProperty(self.dataset.proteincol).loc[test]
        train_targets = self.dataset.getProperty(self.dataset.proteincol).loc[train]
        self.assertEqual(len(test_targets), len(test))
        self.assertEqual(len(train_targets), len(train))
        self.assertTrue(set(test_targets.unique()).isdisjoint(set(train_targets.unique())))

    def testStratifiedPerTarget(self):
        randsplitter = randomsplit(0.2)
        splitter = StratifiedPerTarget(dataset=self.dataset, splitter=randsplitter)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        test_targets = self.dataset.getProperty(self.dataset.proteincol).loc[test.index]
        # check that all targets are present in the test set just once
        # (implied by the stratification on this particular dataset)
        self.assertEqual(len(test_targets), len(self.dataset.getProteinKeys()))

    def testPerTargetTemporal(self):
        year_col = "Year"
        year = 2015
        splitter = TemporalPerTarget(
            dataset=self.dataset,
            year_col=year_col,
            split_years={key: year for key in self.dataset.getProteinKeys()}
        )
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        self.assertTrue(self.dataset.getDF()[year_col].loc[train.index].max() <= year)
        self.assertTrue(self.dataset.getDF()[year_col].loc[test.index].min() > year)

    def testPerTargetScaffoldSplit(self):
        scaffsplit = scaffoldsplit()
        splitter = StratifiedPerTarget(dataset=self.dataset, splitter=scaffsplit)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        test_targets = self.dataset.getProperty(self.dataset.proteincol).loc[test.index]
        # check that all targets are present in the test set at least once, very crude check
        self.assertEqual(len(test_targets.unique()), len(self.dataset.getProteinKeys()))

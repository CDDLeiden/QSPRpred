from unittest import TestCase

from parameterized import parameterized

from qsprpred.data import ClusterSplit, RandomSplit, ScaffoldSplit, GBMTRandomSplit
from qsprpred.data.descriptors.sets import RDKitDescs
from qsprpred.extra.data.descriptors.sets import ProDec
from qsprpred.extra.data.sampling.splits import (
    LeaveTargetsOut,
    PCMSplit,
    TemporalPerTarget,
)
from qsprpred.extra.data.tables.pcm import PCMDataSet
from qsprpred.extra.data.utils.testing.path_mixins import DataSetsMixInExtras


class TestPCMSplitters(DataSetsMixInExtras, TestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.msaProvider = self.getMSAProvider(self.generatedDataPath)
        self.dataset = self.createPCMDataSet(f"{self.__class__.__name__}_test")
        self.dataset.addDescriptors([ProDec(["Zscale Hellberg"], self.msaProvider)])
        self.dataset.addDescriptors([RDKitDescs()])

    @parameterized.expand([(GBMTRandomSplit, ), (ScaffoldSplit, ), (ClusterSplit, )])
    def testPCMSplit(self, splitter):
        splitter = PCMSplit(splitter())

        train_index, test_index = next(self.dataset.split(splitter))
        test_targets = self.dataset.getProperty(self.dataset.proteinIDProp).loc[test_index]
        train_targets = self.dataset.getProperty(self.dataset.proteinIDProp).loc[train_index]
        test_smiles = self.dataset.getProperty(self.dataset.smilesProp).loc[test_index]
        train_smiles = self.dataset.getProperty(self.dataset.smilesProp).loc[train_index]
        self.assertEqual(len(test_targets), len(test_index))
        self.assertEqual(len(train_targets), len(train_index))
        self.assertTrue(
            set(test_smiles.unique()).isdisjoint(set(train_smiles.unique()))
        )

    def testPCMSplitRandomShuffle(self):
        seed = self.dataset.randomState
        self.dataset.save()
        splitter = PCMSplit(GBMTRandomSplit())
        train_index, test_index = next(self.dataset.split(splitter))
        train_order = train_index.tolist()
        test_order = test_index.tolist()
        print(test_order)
        # reload and check if orders are the same if we redo the split
        dataset = PCMDataSet.fromFile(self.dataset.metaFile)
        splitter = PCMSplit(GBMTRandomSplit())
        train_index, test_index = next(self.dataset.split(splitter))
        print(test_index.tolist())
        self.assertEqual(dataset.randomState, seed)
        self.assertListEqual(train_index.tolist(), train_order)
        self.assertListEqual(test_index.tolist(), test_order)

    def testLeaveTargetOut(self):
        target = self.dataset.getProteinKeys()[0:2]
        splitter = LeaveTargetsOut(targets=target)
        train_index, test_index = next(self.dataset.split(splitter))
        test_targets = self.dataset.getProperty(self.dataset.proteinIDProp).loc[test_index]
        train_targets = self.dataset.getProperty(self.dataset.proteinIDProp).loc[train_index]
        self.assertEqual(len(test_targets), len(test_index))
        self.assertEqual(len(train_targets), len(train_index))
        self.assertTrue(
            set(test_targets.unique()).isdisjoint(set(train_targets.unique()))
        )

    def testPerTargetTemporal(self):
        year_col = "Year"
        year = 2015
        splitter = TemporalPerTarget(
            time_prop=year_col,
            split_time={key: year
                         for key in self.dataset.getProteinKeys()},
        )
        train_index, test_index = next(self.dataset.split(splitter))
        self.assertTrue(self.dataset.getDF()[year_col].loc[train_index].max() <= year)
        self.assertTrue(self.dataset.getDF()[year_col].loc[test_index].min() > year)

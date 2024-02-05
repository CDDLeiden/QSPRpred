from typing import Callable
from unittest import TestCase

from parameterized import parameterized

from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.processing.feature_standardizers import SKLearnStandardizer
from qsprpred.data.processing.applicability_domain import ApplicabilityDomain
from qsprpred.data.sampling.splits import DataSplit
from qsprpred.extra.data.tables.pcm import PCMDataSet
from qsprpred.extra.data.utils.testing.path_mixins import DataSetsMixInExtras
from qsprpred.utils.testing.check_mixins import DataPrepCheckMixIn


class TestPCMDataSetPreparation(DataSetsMixInExtras, DataPrepCheckMixIn, TestCase):
    """Test the preparation of the PCMDataSet."""

    def setUp(self):
        super().setUp()
        super().setUpPaths()

    def fetchDataset(self, name: str) -> PCMDataSet:
        """Create a quick dataset with the given name.

        Args:
            name (str): Name of the dataset.

        Returns:
            PCMDataSet: The dataset.
        """
        return self.createPCMDataSet(name)

    @parameterized.expand(DataSetsMixInExtras.getPrepCombos())
    def testPrepCombinations(
        self,
        _,
        name: str,
        feature_calculators: list[DescriptorSet],
        split: DataSplit,
        feature_standardizer: SKLearnStandardizer,
        feature_filter: Callable,
        data_filter: Callable,
        applicability_domain: ApplicabilityDomain,
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
            applicability_domain (Callable): Applicability domain.
        """
        dataset = self.createPCMDataSet(name=name)
        self.checkPrep(
            dataset,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter,
            applicability_domain,
            ["pchembl_value_Median"],
        )

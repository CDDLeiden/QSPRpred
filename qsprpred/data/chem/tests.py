from rdkit import Chem

from ... import TargetTasks
from ...data import QSPRDataset
from ...data.chem.scaffolds import Murcko
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn


class TestScaffolds(DataSetsPathMixIn, QSPRTestCase):
    """Test calculation of scaffolds."""

    def setUp(self):
        """Create a small dataset."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)

    def testScaffoldAdd(self):
        """Test the adding and getting of scaffolds."""
        self.dataset.addScaffolds([Murcko()])
        scaffs = self.dataset.getScaffolds()
        self.assertEqual(scaffs.shape, (len(self.dataset), 1))
        self.dataset.addScaffolds([Murcko()], add_rdkit_scaffold=True, recalculate=True)
        scaffs = self.dataset.getScaffolds(includeMols=True)
        self.assertEqual(scaffs.shape, (len(self.dataset), 2))
        for mol in scaffs[f"Scaffold_{Murcko()}_RDMol"]:
            self.assertTrue(isinstance(mol, Chem.rdchem.Mol))


class TestStandardizers(DataSetsPathMixIn, QSPRTestCase):
    """Test the standardizers."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testInvalidFilter(self):
        """Test the invalid filter."""
        df = self.getSmallDF()
        orig_len = len(df)
        mask = [False] * orig_len
        mask[0] = True
        df.loc[mask, "SMILES"] = "C(C)(C)(C)(C)(C)(C)(C)(C)(C)"  # bad valence example
        dataset = QSPRDataset(
            "standardization_test_invalid_filter",
            df=df,
            target_props=[{"name": "CL", "task": TargetTasks.REGRESSION}],
            drop_invalids=False,
            drop_empty=False,
        )
        dataset.standardizeSmiles("chembl", drop_invalid=False)
        self.assertEqual(len(dataset), len(df))
        dataset.standardizeSmiles("chembl", drop_invalid=True)
        self.assertEqual(len(dataset), orig_len - 1)

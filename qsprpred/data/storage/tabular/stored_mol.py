from typing import Any, Optional

from rdkit import Chem

from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class TabularMol(StoredMol):
    """Simple implementation of a molecule that is stored in a tabular storage."""

    def __init__(
            self,
            mol_id: str,
            origin: str,
            smiles: str,
            parent: Optional["TabularMol"] = None,
            rd_mol: Chem.Mol | None = None,
            props: dict[str, Any] | None = None,
            representations: tuple["TabularMol", ...] | None = None,
    ):
        """Create a new molecule instance.

        Args:
            mol_id (str): identifier of the molecule
            smiles (str): SMILES of the molecule
            parent (TabularMol, optional): parent molecule
            rd_mol (Chem.Mol, optional): rdkit molecule object
            props (dict, optional): properties of the molecule
            representations (tuple, optional): representations of the molecule
        """
        self._origin = origin
        self._parent = parent
        self._id = mol_id
        self._smiles = smiles
        self._rd_mol = rd_mol
        self._props = props
        self._representations = representations

    @property
    def origin(self) -> str:
        return self._origin

    def as_rd_mol(self) -> Chem.Mol:
        """Get the rdkit molecule object.

        Returns:
            (Chem.Mol) rdkit molecule object
        """
        if self._rd_mol is None:
            self._rd_mol = Chem.MolFromSmiles(self.smiles)
        return self._rd_mol

    @property
    def parent(self) -> "TabularMol":
        """Get the parent molecule."""
        return self._parent

    @parent.setter
    def parent(self, parent: "TabularMol"):
        """Set the parent molecule."""
        self._parent = parent

    @property
    def id(self) -> str:
        """Get the identifier of the molecule."""
        return self._id

    @property
    def smiles(self) -> str:
        """Get the SMILES of the molecule."""
        return self._smiles

    @property
    def props(self) -> dict[str, Any] | None:
        """Get the row of the dataframe corresponding to this molecule."""
        return self._props

    @property
    def representations(self) -> list["TabularMol"] | None:
        """Get the representations of the molecule."""
        return self._representations

    @representations.setter
    def representations(self, representations: list["TabularMol"] | None):
        """Set the representations of the molecule."""
        self._representations = representations

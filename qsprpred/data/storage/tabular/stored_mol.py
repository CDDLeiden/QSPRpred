from typing import Any, Optional

from rdkit import Chem

from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class TabularMol(StoredMol):
    """Simple implementation of a molecule that is stored in a tabular storage."""

    def __init__(
            self,
            mol_id: str,
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
        self._parent = parent
        self._id = mol_id
        self._smiles = smiles
        self._rd_mol = rd_mol
        self._props = props
        self._representations = representations

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
        # sdfs = self.sdf()
        # ret = []
        # for sdf in sdfs:
        #     mol = Chem.MolFromMolBlock(
        #         sdf, strictParsing=False, sanitize=False, removeHs=False
        #     )
        #     properties = parse_sdf(next(sdf_to_lines(sdf.split("\n"))))
        #     for prop in properties:
        #         mol.SetProp(prop, properties[prop])
        #     ret.append(mol)
        # return ret

    # def to_file(self, directory, extension=".csv") -> str:
    #     """
    #     Write a minimal file containing the SMILES and the ID of the molecule.
    #     Used for ligrep (.csv is the preferred format).
    #     """
    #     filename = os.path.join(directory, self._id + extension)
    #     if not os.path.isfile(filename):
    #         with open(filename, "w") as f:
    #             f.write("SMILES,id\n")
    #             f.write(f"{self._smiles},{self._id}\n")
    #     return filename

    # def sdf(self) -> List[str] or None:
    #     """
    #     Get the SDF file for this molecule.
    #     """
    #     sdfs = self.parent._sdf[self.parent._sdf.id == self.id].sdf.values
    #     if len(sdfs) == 0:
    #         return None
    #     else:
    #         return [decode(sdf) for sdf in sdfs]

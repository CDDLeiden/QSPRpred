from typing import Literal

from papyrus_structure_pipeline import standardizer as Papyrus_standardizer
from papyrus_structure_pipeline.standardizer import StandardizationResult
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import FragmentParent

from .base import ChemStandardizer


class PapyrusStandardizer(ChemStandardizer):
    """Papyrus standardizer

    Uses Papyrus (>v05.6) standardization protecol to standardize SMILES.

    Béquignon, O.J.M., Bongers, B.J., Jespers, W. et al.
    Papyrus: a large-scale curated dataset aimed at bioactivity predictions.
    J Cheminform 15, 3 (2023). https://doi.org/10.1186/s13321-022-00672-x

    Attributes:
        settings (dict): Settings of the standardizer
    """
    def __init__(
        self,
        keep_stereo: bool = True,
        canonize: bool = True,
        mixture_handling: Literal["keep_largest", "filter", "keep"] = "keep_largest",
        remove_additional_salts: bool = True,
        remove_additional_metals: bool = True,
        filter_inorganic: bool = False,
        filter_non_small_molecule: bool = True,
        small_molecule_min_mw: float = 200,
        small_molecule_max_mw: float = 800,
        canonicalize_tautomer: bool = True,
        tautomer_max_tautomers: int = 2**32 - 1,
        extra_organic_atoms: list | None = None,
        extra_metals: list | None = None,
        extra_salts: list | None = None,
        uncharge: bool = True,
    ):
        """Initialize Papyrus standardizer

        Args:
            keep_stereo (bool, optional): Keep stereochemistry.
            canonize (bool, optional): Canonicalize SMILES.
            mixture_handling (Literal["keep_largest", "filter", "keep"], optional):
                How to handle mixtures. Defaults to "keep_largest".
            remove_additional_salts (bool, optional):
                Removes a custom set of fragments if present in the molecule object.
            remove_additional_metals (bool, optional):
                Removes metal fragments if present in the molecule object.
                Ignored if remove_additional_salts is set to False.
            filter_inorganic (bool, optional): Filter inorganic molecules.
            filter_non_small_molecule (bool, optional): Filter non-small molecules.
            small_molecule_min_mw (float, optional):
                Minimum molecular weight of small molecules.
            small_molecule_max_mw (float, optional):
                Maximum molecular weight of small molecules.
            canonicalize_tautomer (bool, optional): Canonicalize tautomers.
            tautomer_max_tautomers (int, optional):
                Maximum number of tautomers to consider by the tautomer search
                algorithm (<2^32).
            extra_organic_atoms (list, optional):
                Extra organic atoms to consider in addition to the default set
                (Papyrus_standardizer.ORGANIC_ATOMS).
            extra_metals (list, optional):
                Extra metals to consider in addition to the default set
                (Papyrus_standardizer.METALS).
            extra_salts (list, optional):
                Extra salts to consider in addition to the default set
                (Papyrus_standardizer.SALTS).
            uncharge (bool, optional): Uncharge molecules.
        """

        self._settings = {
            "keep_stereo": keep_stereo,
            "canonize": canonize,
            "remove_additional_salts": remove_additional_salts,
            "remove_additional_metals": remove_additional_metals,
            "filter_inorganic": filter_inorganic,
            "filter_non_small_molecule": filter_non_small_molecule,
            "canonicalize_tautomer": canonicalize_tautomer,
            "small_molecule_min_mw": small_molecule_min_mw,
            "small_molecule_max_mw": small_molecule_max_mw,
            "tautomer_allow_stereo_removal": not keep_stereo,
            "tautomer_max_tautomers": tautomer_max_tautomers,
            "extra_organic_atoms":
                (sorted(extra_organic_atoms) if extra_organic_atoms else []),
            "extra_metals": sorted(extra_metals) if extra_metals else [],
            "extra_salts": sorted(extra_salts) if extra_salts else [],
            "mixture_handling": mixture_handling,
            "uncharge": uncharge,
        }
        if self._settings["extra_organic_atoms"]:
            Papyrus_standardizer.ORGANIC_ATOMS.extend(
                self._settings["extra_organic_atoms"]
            )
        if self._settings["extra_metals"]:
            Papyrus_standardizer.METALS.extend(self._settings["extra_metals"])
        if self._settings["extra_salts"]:
            Papyrus_standardizer.SALTS.extend(self._settings["extra_salts"])

    def _fix_errors(
        self, mol: Chem.Mol, error: StandardizationResult
    ) -> Chem.Mol | None:
        """Attempts to fix mixture molecules by keeping the largest fragment.

        Args:
            mol (Chem.Mol): RDKit molecule object
            error (StandardizationResult): Error code

        Returns:
            Chem.Mol | None: Fixed molecule or None if molecule cannot be fixed
        """
        if (
            error == StandardizationResult.MIXTURE_MOLECULE and
            self._settings["mixture_handling"] == "keep_largest"
        ):
            mol = FragmentParent(mol)
            return mol
        return None

    def convertSMILES(self, smiles: str, verbose: bool = False) -> str | None:
        """Standardize SMILES using Papyrus standardization protocol.

        Args:
            smiles (str): SMILES to be standardized
            verbose (bool, optional): Print verbose output. Defaults to False.

        Returns:
            tuple[str | None, str]:
                a tuple where the first element is the standardized SMILES and the
                second element is the original SMILES
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        out = Papyrus_standardizer.standardize(
            mol,
            return_type=True,
            remove_additional_salts=self._settings["remove_additional_salts"],
            remove_additional_metals=self._settings["remove_additional_metals"],
            filter_mixtures=(
                False if self._settings["mixture_handling"] == "keep" else True
            ),
            filter_inorganic=self._settings["filter_inorganic"],
            filter_non_small_molecule=self._settings["filter_non_small_molecule"],
            small_molecule_min_mw=self._settings["small_molecule_min_mw"],
            small_molecule_max_mw=self._settings["small_molecule_max_mw"],
            canonicalize_tautomer=self._settings["canonicalize_tautomer"],
            tautomer_max_tautomers=self._settings["tautomer_max_tautomers"],
            tautomer_allow_stereo_removal=self.
            _settings["tautomer_allow_stereo_removal"],
            uncharge=self._settings["uncharge"],
        )
        results = list(out[1:])
        if StandardizationResult.CORRECT_MOLECULE not in results:
            mol = self._fix_errors(mol, results[-1])
            if not mol:
                if verbose:
                    print("SMILES rejected", smiles)
                    print("\tCause:", results)
                return None
            else:
                return self.convertSMILES(
                    Chem.MolToSmiles(
                        mol,
                        isomericSmiles=self._settings["keep_stereo"],
                        canonical=self._settings["canonize"],
                    )
                )
        else:
            return (
                Chem.MolToSmiles(
                    out[0],
                    canonical=self._settings["canonize"],
                    isomericSmiles=self._settings["keep_stereo"],
                ) if out[0] else None
            )

    @property
    def settings(self) -> dict:
        return self._settings

    def getID(self) -> str:
        """Get the ID of the standardizer.

        In this case, the ID is based on the settings of the standardizer.
        It starts with 'PapyrusStandardizer' followed by a tilde and
        the settings concatenated with a colon.

        Returns:
            str: ID of the standardizer
        """
        sorted_keys = sorted(self._settings.keys())
        return "PapyrusStandardizer~" + ":".join(
            [f"{key}={self._settings[key]!s}" for key in sorted_keys]
        )

    def fromSettings(self, settings: dict) -> "PapyrusStandardizer":
        """Create a Papyrus standardizer from settings.

        Args:
            settings (dict): settings of the standardizer

        Returns:
            PapyrusStandardizer: a Papyrus standardizer
        """
        return PapyrusStandardizer(**settings)

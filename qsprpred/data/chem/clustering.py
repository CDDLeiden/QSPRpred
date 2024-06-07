import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Mol
from rdkit.SimDivFilters import rdSimDivPickers

from .scaffolds import BemisMurckoRDKit, Scaffold
from .. import MoleculeTable
from ..descriptors.fingerprints import Fingerprint, MorganFP
from ...logs import logger

from qsprpred.data.processing.mol_processor import MolProcessorWithID


class MoleculeClusters(MolProcessorWithID, ABC):
    """
    Abstract base class for clustering molecules.

    Attributes:
        nClusters (int): number of clusters
    """
    
    def __call__(self, mols: list[str | Mol], props, *args, **kwargs):
        """
        Calculate the clusters for a list of  molecules.

        Args:
            mol (str | Mol): SMILES or RDKit molecule to calculate the cluster for.

        Returns:
            list of cluster index for each molecule
        """
        if isinstance(mols[0], Mol):
            mols = [Chem.MolToSmiles(mol) for mol in mols]

        clusters = self.get_clusters(mols)
        
        # map clusters to molecules
        output = np.array([-1]*len(mols))
        for cluster_idx, molecule_idxs in clusters.items():
            output[molecule_idxs] = cluster_idx
        
        return pd.Series(output, index=props[self.idProp])


    @abstractmethod
    def get_clusters(self, smiles_list: list[str]) -> dict:
        """
        Cluster molecules.

        Args:
            smiles_list (list): list of molecules to be clustered

        Returns:
            clusters (dict): dictionary of clusters, where keys are cluster indices
                and values are indices of molecules
        """

    def _set_nClusters(self, N: int) -> None:
        self.nClusters = self.nClusters if self.nClusters is not None else N // 10
        if self.nClusters < 10:
            self.nClusters = 10
            logger.warning(
                f"Number of initial clusters is too small to combine them well,\
                it has set to {self.nClusters}"
            )
            
    def supportsParallel(self) -> bool:
        return False
    
    @abstractmethod
    def __str__(self):
        pass


class RandomClusters(MoleculeClusters):
    """
    Randomly cluster molecules.

    Attributes:
        seed (int): random seed
        nClusters (int): number of clusters
        id_prop (str): name of the property to be used as ID
    """

    def __init__(
        self, seed: int = 42, n_clusters: int | None = None, id_prop: str | None = None
        ):
        super().__init__(id_prop=id_prop)
        self.seed = seed
        self.nClusters = n_clusters

    def get_clusters(self, smiles_list: list[str]) -> dict:
        """
        Cluster molecules.

        Args:
            smiles_list (list): list of molecules to be clustered

        Returns:
            clusters (dict): dictionary of clusters, where keys are cluster indices \
                and values are indices of molecules
        """

        self._set_nClusters(len(smiles_list))

        # Initialize clusters
        clusters = {i: [] for i in range(self.nClusters)}

        # Randomly assign each molecule to a cluster
        indices = np.random.RandomState(seed=self.seed).permutation(len(smiles_list))
        for i, index in enumerate(indices):
            clusters[i % self.nClusters].append(index)

        return clusters
    
    def __str__(self):
        return "RandomClusters"


class ScaffoldClusters(MoleculeClusters):
    """
    Cluster molecules based on scaffolds.

    Attributes:
        scaffold (Scaffold): scaffold generator
        id_prop (str): name of the property to be used as ID
    """

    def __init__(
        self, scaffold: Scaffold = BemisMurckoRDKit(), id_prop: str | None = None
    ):
        super().__init__(id_prop=id_prop)
        self.scaffold = scaffold

    def get_clusters(self, smiles_list: list[str]) -> dict:
        """
        Cluster molecules.

        Args:
            smiles_list (list): list of molecules to be clustered

        Returns:
            clusters (dict): dictionary of clusters, where keys are cluster indices
                and values are indices of molecules
        """

        # Generate scaffolds for each molecule
        mt = MoleculeTable(
            "scaffolds", pd.DataFrame({"SMILES": smiles_list}), n_jobs=os.cpu_count()
        )
        mt.addScaffolds([self.scaffold])
        scaffolds = (
            mt.getScaffolds([self.scaffold])
            .loc[mt.getDF().index, :]
            .iloc[:, 0]
            .tolist()
        )

        # Get unique scaffolds and initialize clusters
        unique_scaffolds = sorted(list(set(scaffolds)))
        clusters = {i: [] for i in range(len(unique_scaffolds))}

        # Cluster molecules based on scaffolds
        for i, scaffold in enumerate(scaffolds):
            clusters[unique_scaffolds.index(scaffold)].append(i)

        return clusters
    
    def __str__(self):
        return f"ScaffoldClusters_{self.scaffold}"


class FPSimilarityClusters(MoleculeClusters):
    def __init__(
        self,
        fp_calculator: Fingerprint = MorganFP(radius=3, nBits=2048),
        id_prop: str | None = None,
    ) -> None:
        super().__init__(id_prop=id_prop)
        self.fp_calculator = fp_calculator

    def get_clusters(self, smiles_list: list[str]) -> dict:
        """
        Cluster a list of SMILES strings based on molecular dissimilarity.

        Args:
            smiles_list (list): list of SMILES strings to be clustered

        Returns:
            clusters (dict): dictionary of clusters, where keys are cluster indices
                and values are indices of molecules
        """

        # Get fingerprints for each molecule
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        fps = self.fp_calculator.getDescriptors(
            mols, props={self.fp_calculator.idProp: [str(x) for x in range(len(mols))]}
        )

        # Convert np fingerprints to rdkit fingerprints
        fps = [
            DataStructs.cDataStructs.CreateFromBitString("".join(fp.astype(str)))
            for fp in fps
        ]

        # Get cluster centroids and initialize clusters
        centroid_indices = self._get_centroids(fps)
        clusters = {i: [] for i in range(len(centroid_indices))}

        # Cluster molecules based on centroids
        for i, fp in enumerate(fps):
            similarities = [
                DataStructs.FingerprintSimilarity(fp, fps[j]) for j in centroid_indices
            ]
            clusters[np.argmax(similarities)].append(i)

        return clusters

    @abstractmethod
    def _get_centroids(self, fps: list) -> list:
        pass


class FPSimilarityMaxMinClusters(FPSimilarityClusters):
    """
    Cluster molecules based on molecular fingerprint with MaxMin algorithm.

    Attributes:
        fp_calculator (FingerprintSet): fingerprint calculator
        nClusters (int): number of clusters
        seed (int): random seed
        initialCentroids (list): list of indices of initial cluster centroids
        id_prop (str): name of the property to be used as ID
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        seed: int | None = None,
        initial_centroids: list[str] | None = None,
        fp_calculator: Fingerprint = MorganFP(radius=3, nBits=2048),
        id_prop: str | None = None,
    ):
        super().__init__(fp_calculator=fp_calculator, id_prop=id_prop)
        self.nClusters = n_clusters
        self.seed = seed
        self.initialCentroids = initial_centroids

    def _get_centroids(self, fps: list) -> list:
        """
        Get cluster centroids with MaxMin algorithm.

        Args:
            fps (list): list of molecular fingerprints

        Returns:
            centroid_indices (list): list of indices of cluster centroids
        """
        self._set_nClusters(len(fps))
        picker = rdSimDivPickers.MaxMinPicker()
        self.centroid_indices = picker.LazyBitVectorPick(
            fps,
            len(fps),
            self.nClusters,
            firstPicks=self.initialCentroids if self.initialCentroids else [],
            seed=self.seed if self.seed is not None else -1,
        )

        return self.centroid_indices
    
    def __str__(self):
        return "FPSimilarityMaxMinClusters"


class FPSimilarityLeaderPickerClusters(FPSimilarityClusters):
    """
    Cluster molecules based on molecular fingerprint with LeaderPicker algorithm.

    Attributes:
        fp_calculator (FingerprintSet): fingerprint calculator
        similarity_threshold (float): similarity threshold
        id_prop (str): name of the property to be used as ID
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        fp_calculator: Fingerprint = MorganFP(radius=3, nBits=2048),
        id_prop: str | None = None,
    ):
        super().__init__(fp_calculator=fp_calculator, id_prop=id_prop)
        self.similarityThreshold = similarity_threshold
        self.fpCalculator = fp_calculator

    def _get_centroids(self, fps: list) -> list:
        """
        Get cluster centroids with LeaderPicker algorithm.
        """
        picker = rdSimDivPickers.LeaderPicker()
        self.centroid_indices = picker.LazyBitVectorPick(
            fps, len(fps), self.similarityThreshold
        )

        return self.centroid_indices
    
    def __str__(self):
        return "FPSimilarityLeaderPickerClusters"

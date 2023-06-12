"""
interfaces

Created by: Martin Sicho
On: 25.11.22, 13:36
"""
from abc import ABC, abstractmethod


class Fingerprint(ABC):
    """Base class for fingerprints."""
    def __call__(self, mols):
        """Actual call method.

        Args:
            mols: molecules to obtain the fingerprints of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self.getFingerprints(mols)

    @abstractmethod
    def settings(self):
        """Return settings of fingerprint."""

    @abstractmethod
    def __len__(self):
        """Return length of fingerprint."""

    @abstractmethod
    def getKey(self):
        """Return identifier of fingerprint."""


class Scorer(ABC):
    """Used to calculate customized scores."""
    def __init__(self, modifier=None):
        self.modifier = modifier

    @abstractmethod
    def getScores(self, mols, frags=None):
        """Return scores for the input molecules.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (list): `list` of scores for "mols"
        """

    def __call__(self, mols, frags=None):
        """Actual call method. Modifies the scores before returning them.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (DataFrame): a data frame with columns name 'VALID' and 'DESIRE'
                indicating the validity of the SMILES and the degree of desirability
        """
        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores):
        """Modify the scores with the given ScoreModifier.

        Args:
            scores:

        Returns:
            scores
        """
        if self.modifier:
            return self.modifier(scores)
        else:
            return scores

    @abstractmethod
    def getKey(self):
        pass

    def setModifier(self, modifier):
        self.modifier = modifier

    def getModifier(self):
        return self.modifier

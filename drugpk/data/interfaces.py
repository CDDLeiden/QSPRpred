from abc import ABC, abstractmethod

class datasplit(ABC):
    """
    Defines a function split a dataframe into train and test set

    """
    @abstractmethod
    def __call__(self, df, Xcol, ycol):
        """
        Split dataframe df into train and test set.
        Args:
            df: pandas dataframe to split
            Xcol: input column name, e.g. "SMILES"
            ycol: output column name, e.g. "Cl"
        Returns:
            X: training set input
            X_ind: test set input
            y: training set output
            y_ind: test set output
        """
        pass

class datafilter(ABC):
    """
        filter out some rows from a dataframe
    """
    @abstractmethod
    def __call__(self, df):
        """
        filter out some rows from a dataframe
        Args:
            df: pandas dataframe to filter
        Returns:
            df: filtered pandas dataframe
        """
        pass

class featurefilter(ABC):
    """
        filter out uninformative features from a dataframe
    """
    @abstractmethod
    def __call__(self, df):
        """
        filter out uninformative features from a dataframe
        Args:
            df: pandas dataframe to filter
        Returns:
            df: filtered pandas dataframe
        """
        pass

class Scorer(ABC):
    """
    Used to calculate customized scores.

    """

    def __init__(self, modifier=None):
        self.modifier = modifier

    @abstractmethod
    def getScores(self, mols, frags=None):
        """
        Returns scores for the input molecules.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (list): `list` of scores for "mols"
        """

        pass

    def __call__(self, mols, frags=None):
        """
        Actual call method. Modifies the scores before returning them.

        Args:
            mols: molecules to score
            frags: input fragments

        Returns:
            scores (DataFrame): a data frame with columns name 'VALID' and 'DESIRE' indicating the validity of the SMILES and the degree of desirability
        """

        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores):
        """
        Modify the scores with the given `ScoreModifier`.

        Args:
            scores:

        Returns:

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
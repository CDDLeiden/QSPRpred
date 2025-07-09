from abc import abstractmethod
import pandas as pd
from ...utils.serialization import JSONSerializable
from ...utils.interfaces.randomized import Randomized

class Step(JSONSerializable):
    """A data preprocessing step that can be applied to a dataset"""
    
    def __init__(self):
        """Initialize the step"""
        self._fitted = False
    
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the step to the dataset
        
        If the step requires fitting to the data, this method should be implemented.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame): training targets
        """
        self._fitted = True
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply the step to the dataset
        
        Note. the step should not modify the original data
        
        Args:
            X (pd.DataFrame): data to be transformed
            y (pd.DataFrame): target data to be transformed
        
        Returns:
            pd.DataFrame: transformed data
            pd.DataFrame: (transformed) target data
        """
        pass
    
    def fitTransform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Fit the step to the dataset and apply it
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame): training targets
            
        Returns:
            pd.DataFrame: transformed data
            pd.DataFrame: (transformed) target data
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    @property
    def fitted(self) -> bool:
        """Check if the step is fitted
        
        Returns:
            bool: True if the step is fitted, False otherwise
        """
        return self._fitted
        
    
class DummyStep(Step):
    """Dummy step that does nothing"""
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Just return the input data
        
        Args:
            X (pd.DataFrame): data to be transformed
            y (pd.DataFrame | None): target data to be transformed
        
        Returns:
            pd.DataFrame: unchanged data
            pd.DataFrame | None: unchanged target data
        """
        return X, y
    
class Shuffle(Step, Randomized):
    """Step that shuffles the data
    
    Attributes:
        randomState (int | None): Seed to randomize the shuffle.
    """
    
    def __init__(self, seed: int | None = None):
        """Initialize the shuffle step
        
        Args:
            seed (int | None): Seed to randomize the shuffle.
        """
        self._fitted = False
        self.seed = seed
    
    @property
    def randomState(self) -> int | None:
        """Get the random state for the object."""
        return self.seed
    
    @randomState.setter
    def randomState(self, seed: int | None):
        """Set the random state for the object.

        Args:
            seed (int | None):
                The seed to use to randomize the action. If `None`,
                a random seed is used instead of a fixed one.
        """
        self.seed = seed
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Shuffle the data
        
        Args:
            X (pd.DataFrame): data to be shuffled
            y (pd.DataFrame | None): target data to be shuffled
            
        Returns:
            pd.DataFrame: shuffled data
            pd.DataFrame | None: shuffled target data
        """
        X_shuffled = X.sample(frac=1, random_state=self.randomState)
        y_shuffled = y.loc[X_shuffled.index] if y is not None else None
        return X_shuffled, y_shuffled


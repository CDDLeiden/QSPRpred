from abc import ABC, abstractmethod
import pandas as pd
from ...utils.serialization import JSONSerializable
# from ..descriptors.sets import DescriptorSet
from qsprpred.data.sampling.splits import DataSplit
from typing import Generator

class Step(JSONSerializable):
    """A data preprocessing step that can be applied to a dataset"""
    
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the step to the dataset
        
        If the step requires fitting to the data, this method should be implemented.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame): training targets
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    
    def fitTransform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    
class DummyStep(Step):
    """Dummy step that does nothing"""
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Just return the input data"""
        return X, y

class SklearnStep(Step):
    """Step that wraps a scikit-learn transformer"""
    
    def __init__(self, transformer):
        self.transformer = transformer
    
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        self.transformer.fit(X, y)
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns, index=X.index), y

class Pipeline(ABC):
    """Pipeline class for data preprocessing steps
    
    Pipeline is a sequence of data preprocessing steps that can be applied to a dataset.
    
    Args:
        steps (dict[str, Step]): Dictionary of named steps in the pipeline
    """
    
    def __init__(self, steps: dict[str, Step]):
        self.steps = steps
    
    @abstractmethod
    def fitTransform(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
    

class QSPRPipeline(Pipeline):
    """Pipeline class for QSPR prediction
    
    QSPRPipeline is a sequence of data preprocessing steps that can be applied to a dataset.
    
    Args:
        steps (dict[str, Step]): Dictionary of named steps in the pipeline
    """
    def __init__(
        self,
        # feature_calculators: list[DescriptorSet] | None = None,
        steps: dict[str, Step] = {},
    ):
        super().__init__(steps)
        # self.feature_calculators = feature_calculators
        for name, step in steps.items():
            if not isinstance(step, Step):
                if hasattr(step, 'fit_transform'):
                    steps[name] = SklearnStep(step)
        self.originalfeatureNames = None
        self.featureNames = None
    
    def fitTransform(
        self, X: pd.DataFrame, y: None | pd.DataFrame = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.originalfeatureNames = X.columns
        for step in self.steps.values():
            X, y = step.fitTransform(X, y)
        self.featureNames = X.columns
        return X, y

    def transform(
        self, X: pd.DataFrame, y: None | pd.DataFrame = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # add NaN values for missing features
        missing_features = list(set(self.originalfeatureNames) - set(X.columns))
        X = pd.concat(
            [X, pd.DataFrame(0, index=X.index, columns=missing_features)], axis=1
        )
        X = X[self.originalfeatureNames]
        for step in self.steps.values():
            X, y = step.transform(X, y)
        return X, y
            
    def apply(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.DataFrame | None = None,
        fit: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None
    ]:
        """Apply the pipeline to the data
        
        If fit is True, the pipeline is fitted to the training data and 
        then applied to the train and test data. If fit is False, the pipeline is only
        applied to the data.

        Args:
            X_train (pd.DataFrame): training data to apply the pipeline to
            y_train (pd.DataFrame | None): training target data to apply the pipeline to
            X_test (pd.DataFrame | None): test data to apply the pipeline to
            y_test (pd.DataFrame | None): test target data to apply the pipeline to
            refit (bool): whether to fit the pipeline
        
        Returns:
            X_train (pd.DataFrame): transformed training data
            y_train (pd.DataFrame | None): transformed training targets
            X_test (pd.DataFrame | None): transformed test data
            y_test (pd.DataFrame | None): transformed test targets
        """
        if fit:
            X_train, y_train = self.fitTransform(X_train, y_train)
        else:
            X_train, y_train = self.transform(X_train, y_train)
        if X_test is not None:
            X_test, y_test = self.transform(X_test, y_test)
        return X_train, X_test, y_train, y_test
    
    
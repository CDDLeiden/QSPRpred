from abc import ABC, abstractmethod
import pandas as pd
from ...utils.serialization import JSONSerializable
from ..descriptors.sets import DescriptorSet
from qsprpred.data.sampling.splits import DataSplit
from typing import Generator
from qsprpred.data.tables.qspr import QSPRTable
from ...utils.interfaces.randomized import Randomized

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
    
class Shuffle(Step, Randomized):
    """Step that shuffles the data"""
    
    def __init__(self, seed: int | None = None):
        self.seed = seed
    
    @property
    def randomState(self) -> int:
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
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Shuffle the data"""
        X_shuffled = X.sample(frac=1, random_state=self.randomState)
        y_shuffled = y.loc[X_shuffled.index] if y is not None else None
        return X_shuffled, y_shuffled

class SklearnStep(Step):
    """Step that wraps a scikit-learn transformer"""
    
    def __init__(self, transformer):
        self.transformer = transformer
    
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        self.transformer.fit(X, y)
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns, index=X.index), y

class InvalidRemove(Step):
    """Step that removes rows containing NaN values in a specified column"""
    
    def __init__(self, features: list[str] | None = None):
        """Initialize the step with the columns to check for NaN values
        
        If no columns are specified, all columns are checked for NaN values.
        
        Args:
            features (list[str] | None): columns to check for NaN values
        """
        self.selected_features = features
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Remove rows containing NaN values in the specified columns"""
        if self.selected_features is None:
            self.selected_features = X.columns
        # only take selected features that are in the current data
        selected_features = list(set(self.selected_features) & set(X.columns))
        # print ids of removed rows
        # print(X[X.isnull().any(axis=1)].index)
        X = X.dropna(subset=selected_features)
        if y is not None:
            y = y.loc[X.index]
        return X, y
        

class BasePipeline(ABC):
    """Pipeline class for data preprocessing steps
    
    Pipeline is a sequence of data preprocessing steps that can be applied to a dataset.
    
    Args:
        steps (dict[str, Step]): Dictionary of named steps in the pipeline
    """
    @abstractmethod
    def fitTransform(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass
    

class Pipeline(BasePipeline, Randomized, JSONSerializable):
    """Pipeline class for QSPR prediction
    
    A sequence of data preprocessing steps that can be applied to a dataset.
    
    Args:
        steps (dict[str, Step]): Dictionary of named steps in the pipeline
        fixed (list[str]): List of step names that should not be fitted, only transformed
        fit_on (dict[str, str]): Settings for which data a step should be fitted on.
            Either 'train', 'test' or 'both', if not specified the step is fitted on
            the training data.
        apply_to (dict[str, str]): Settings for which data a step should be applied to.
            Either 'train', 'test' or 'both', if not specified the step is applied to 
            both.
        seed (int | None): Seed to randomize the pipeline
    """
    def __init__(
        self,
        steps: dict[str, Step] = {},
        fixed: list[str] = [],
        fit_on: dict[str, str] = {},
        apply_to: dict[str, str] = {},
        seed: int | None = None,
    ):
        self.steps = steps
        self.fixed = fixed
        self.fitOn = fit_on
        self.applyTo = apply_to
        for name, step in steps.items():
            if not isinstance(step, Step):
                if hasattr(step, 'fit_transform'):
                    steps[name] = SklearnStep(step)
        self.originalfeatureNames = None
        self.featureNames = None
        self.randomState = seed

    @property
    def randomState(self) -> int:
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
    
    def fitTransform(
        self, X: pd.DataFrame, y: None | pd.DataFrame = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.originalfeatureNames = X.columns
        for name, step in self.steps.items():
            if hasattr(step, 'randomState'):
                step.randomState = self.randomState
            if name in self.fixed:
                X, y = step.transform(X, y)
            else:
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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
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
        
        return X_train, y_train, X_test, y_test
    
class DatasetPipeline(Pipeline):
    def __init__(
        self,
        feature_calculators: list[DescriptorSet] | None = None,
        steps: dict[str, Step] = {},
        fixed: list[str] = [],
    ):
        super().__init__(steps, fixed=fixed)
        self.feature_calculators = feature_calculators
        
    def apply(
        self,
        dataset: QSPRTable,
        split: DataSplit | None = None,
        fit: bool = True,
        seed: int | None = None,
        order: pd.Index | None = None, # FIXME: added to reproduce original behavior
    ) -> Generator[
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None],
        None,
        None,
    ]:
        """Apply the pipeline to the dataset
        
        Note. the random state of the dataset is used to randomize the pipeline
            when the seed of feature calculators, splits or steps is not set.
        
        Args:
            dataset (QSPRTable): dataset to apply the pipeline to
            split (DataSplit): split to apply to the dataset
            seed (int | None): seed to randomize the pipeline,
                if None, the random state of the dataset is used
            fit (bool): whether to fit the pipeline
        
        Yields:
            X_train (pd.DataFrame): transformed training data
            y_train (pd.DataFrame): transformed training targets
            X_test (pd.DataFrame | None): transformed test data if split is not None
            y_test (pd.DataFrame | None): transformed test targets if split is not None
        """
        self.randomState = dataset.randomState if seed is None else seed
        
        if self.feature_calculators is not None:
            for feature_calculator in self.feature_calculators:
                if hasattr(feature_calculator, 'randomState') and feature_calculator.randomState is None:
                    feature_calculator.randomState = self.randomState
            dataset.addDescriptors(self.feature_calculators)
        X = dataset.getDescriptors()
        if self.originalfeatureNames is not None:
            assert all(
                feature in X.columns for feature in self.originalfeatureNames
            ), "Some features are missing in the dataset, please check if any "
            "descriptors that were added to the dataset directly "
            "before fitting the pipeline are missing in the dataset."
        y = dataset.getTargets()
        if order is not None:  # FIXME: added to reproduce original behavior
            X = X.loc[order]  # FIXME: added to reproduce original behavior
            y = y.loc[order]  # FIXME: added to reproduce original behavior
        if split is None:
            X, y, _, _ = super().apply(X, y, fit = fit)
            yield X, y
        else:
            if isinstance(split, str):
                split = dataset.getSplit(split)
            if hasattr(split, 'randomState') and split.randomState is None:
                    split.randomState = self.randomState
            for train_index, test_index in dataset.split(split, X, y):  # FIXME: added to reproduce original behavior
                X_train, y_train, X_test, y_test = (
                    X.loc[train_index], y.loc[train_index], X.loc[test_index], y.loc[test_index]
                )
                yield super().apply(X_train, y_train, X_test, y_test, fit)

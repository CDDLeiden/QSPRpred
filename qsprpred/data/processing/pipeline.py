import pandas as pd
from .step import Step
from .feature_transformers import SklearnStep
from ...utils.serialization import JSONSerializable
from ..descriptors.sets import DescriptorSet
from qsprpred.data.sampling.splits import DataSplit
from typing import Generator
from qsprpred.data.tables.qspr import QSPRTable
from ...utils.interfaces.randomized import Randomized
from sklearn.base import BaseEstimator


class Pipeline(Randomized, JSONSerializable):
    """Pipeline class for for sequentially applying data preprocessing steps.
    
    Attributes:
        steps (dict[str, Step | BaseEstimator]): Dictionary of named steps in the 
            pipeline, if the step is a scikit-learn transformer, it will be wrapped in 
            a SklearnStep.
        fixed (list[str]): List of step names that should not be fitted, only 
            transformed
        fitOn (dict[str, str]): Settings for which data a step should be fitted on.
            Either 'train', 'test' or 'both', if not specified the step is fitted on
            the training data.
        applyTo (dict[str, str]): Settings for which data a step should be applied to.
            Either 'train', 'test' or 'both', if not specified the step is applied to 
            both.
        randomState (int | None): Random state for the pipeline
        skip (list[str]): List of step names to skip
        fitted (bool): Whether the pipeline is fitted
    """
    def __init__(
        self,
        steps: dict[str, Step | BaseEstimator] = {},
        fixed: list[str] = [],
        fit_on: dict[str, str] = {},
        apply_to: dict[str, str] = {},
        skip: list[str] = [],
        seed: int | None = None,
    ):
        """Initialize the Pipeline
        
        Args:
            steps (dict[str, Step | BaseEstimator]): Dictionary of named steps in the 
                pipeline, if the step is a scikit-learn transformer, it will be wrapped 
                in a SklearnStep.
            fixed (list[str]): List of step names that should not be fitted, only 
                transformed
            fit_on (dict[str, str]): Settings for which data a step should be fitted on.
                Either 'train', 'test' or 'both', if not specified the step is fitted on
                the training data.
            apply_to (dict[str, str]): Settings for which data a step should be applied 
                to. Either 'train', 'test' or 'both', if not specified the step is 
                applied to both.
            skip (list[str]): List of step names to skip
            seed (int | None): Random state for the pipeline
        """
        self.steps = steps
        self.fixed = fixed
        self.fitOn = fit_on
        self.applyTo = apply_to
        for name, step in steps.items():
            if not isinstance(step, Step):
                if hasattr(step, 'fit_transform'):
                    steps[name] = SklearnStep(step)
        self.randomState = seed
        self._skip = skip
        self._fitted = False

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
            
    def apply(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.DataFrame | None = None,
        fit: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """Apply the pipeline to the data
        
        If fit is True, the pipeline is fitted to the training data and 
        then applied to the train and test data. If fit is False, the pipeline is only
        applied to the data.

        Args:
            X_train (pd.DataFrame): training data to apply the pipeline to
            y_train (pd.DataFrame | None): training target data to apply the pipeline to
            X_test (pd.DataFrame | None): test data to apply the pipeline to
            y_test (pd.DataFrame | None): test target data to apply the pipeline to
            fit (bool): whether to fit the pipeline
        
        Returns:
            X_train (pd.DataFrame): transformed training data
            y_train (pd.DataFrame | None): transformed training targets
            X_test (pd.DataFrame | None): transformed test data
            y_test (pd.DataFrame | None): transformed test targets
        """
        if not self.fitted and not fit and len(self.steps) > 0:
            raise ValueError("Pipeline must be fitted before transforming data")
        
        for name, step in self.steps.items():
            if name in self.skip:
                continue
            if fit:
                if hasattr(step, 'randomState'):
                    step.randomState = self.randomState if step.randomState is None else step.randomState
                # Fit step on the specified data
                if name not in self.fixed:
                    if self.fitOn.get(name, 'train') == 'train':
                        step.fit(X_train, y_train)
                    elif self.fitOn.get(name, 'train') == 'both':
                        X_all, y_all = pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
                        step.fit(X_all, y_all)
                    elif self.fitOn.get(name, 'train') == 'test':
                        step.fit(X_test, y_test)
                    else:
                        raise ValueError(f"Unknown value for {name} fit_on: {self.fitOn.get(name)}")
                self._fitted = True
            # Apply step on the specified data
            if self.applyTo.get(name, 'both') in ['train', 'both']:
                X_train, y_train = step.transform(X_train, y_train)
            if self.applyTo.get(name, 'both') in ['test', 'both']:
                if X_test is not None:
                    X_test, y_test = step.transform(X_test, y_test)
            if self.applyTo.get(name, 'both') not in ['train', 'test', 'both']:
                raise ValueError(f"Unknown value for {name} apply_to: {self.applyTo.get(name)}")
            # Check number of features is still consistent between training and test data
            if X_test is not None:
                assert X_train.shape[1] == X_test.shape[1], f"Number of features in training and test data is not consistent after step {name}"
                assert all(X_train.columns == X_test.columns), f"Feature names in training and test data are not consistent after step {name}"
        
        return X_train, y_train, X_test, y_test
    
    def removeStep(self, name: str):
        """Remove a step from the pipeline
        
        Args:
            name (str): name of the step to remove
        """
        self.steps.pop(name)
        
    def addStep(self, name: str, step: Step, fit_on: str = 'train', apply_to: str = 'both', fixed: bool = False):
        """Add a step to the pipeline
            
        Args:
            name (str): name of the step
            step (Step): step to add to the pipeline
            fit_on (str): whether to fit the step on 'train', 'test' or 'both'
            apply_to (str): whether to apply the step on 'train', 'test' or 'both'
            fixed (bool): whether the step should be fixed and not fitted
        """
        self.steps[name] = step
        self.fitOn[name] = fit_on
        self.applyTo[name] = apply_to
        if fixed:
            self.fixed.append(name)
            
    def orderSteps(self, order: list[str]):
        """Order the steps in the pipeline
        
        Args:
            order (list[str]): list of step names in the desired order
        """
        assert set(order) == set(self.steps.keys()), "Order must contain all step names"
        self.steps = {name: self.steps[name] for name in order}
    
    @property
    def fitted(self) -> bool:
        """Check if the pipeline is fitted"""
        return self._fitted
    
    @property
    def skip(self) -> list[str]:
        """Get the steps to skip
        
        The steps to skip are not fitted or transformed, but
        are still present in the pipeline.
        
        Returns:
            list[str]: list of step names to skip
        """
        return self._skip
    
    def addSkip(self, name: str):
        """Add a step to the skip list
        
        Args:
            name (str): name of the step to skip
        """
        self._skip.append(name)
    
    def removeSkip(self, name: str):
        """Remove a step from the skip list
        
        Args:
            name (str): name of the step to remove from the skip list
        """
        self._skip.remove(name)
    
    def __str__(self):
        steps = []
        for name, obj in self.steps.items():
            step = f"{name} ({obj.__class__.__name__}): "
            step += f"fit_on={self.fitOn.get(name, 'train')}, "
            step += f"apply_to={self.applyTo.get(name, 'both')}, "
            step += f"fixed={name in self.fixed}"
            steps.append(step)
        return (
            f"{self.__class__.__name__}\n"
            f"steps:\n  " + 
            f"\n  ".join(steps) + 
            f"\nseed: {self.seed}"
            f"\nfitted: {self.fitted}"
        )
            
    
class DatasetPipeline(Pipeline):
    """Pipeline class for applying data preprocessing steps to a QSPRDataset.
    
    Attributes:
        feature_calculators (list[DescriptorSet] | None): List of feature calculators 
            to apply to the dataset. If None, no feature calculators are applied.
        originalfeatureNames (list[str] | None): Original feature names in the dataset 
            before applying the pipeline.
    """
    def __init__(
        self,
        feature_calculators: list[DescriptorSet] | None = None,
        steps: dict[str, Step | BaseEstimator] = {},
        fixed: list[str] = [],
        fit_on: dict[str, str] = {},
        apply_to: dict[str, str] = {},
        skip: list[str] = [],
        seed: int | None = None,
    ):
        """Initialize the DatasetPipeline
        
        Args:
            feature_calculators (list[DescriptorSet] | None): List of feature 
                calculators to apply to the dataset.
            steps (dict[str, Step | BaseEstimator]): Dictionary of named steps in the 
                pipeline, if the step is a scikit-learn transformer, it will be wrapped 
                in a SklearnStep.
            fixed (list[str]): List of step names that should not be fitted, only 
                transformed
            fit_on (dict[str, str]): Settings for which data a step should be fitted on.
                Either 'train', 'test' or 'both', if not specified the step is fitted on
                the training data.
            apply_to (dict[str, str]): Settings for which data a step should be applied 
                to. Either 'train', 'test' or 'both', if not specified the step is 
                applied to both.
            skip (list[str]): List of step names to skip
            seed (int | None): Random state for the pipeline
        """
        super().__init__(steps, fixed, fit_on, apply_to, skip, seed)
        self.originalfeatureNames = None
        self.feature_calculators = feature_calculators
        
    def apply(
        self,
        dataset: QSPRTable,
        split: DataSplit | None = None,
        fit: bool = True,
        seed: int | None = None,
        order: pd.Index | None = None, # FIXME: added to reproduce original behavior
    ) -> Generator[
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame],
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
        
        # prepare X and y from the dataset
        if self.feature_calculators is not None:
            for feature_calculator in self.feature_calculators:
                if hasattr(feature_calculator, 'randomState') and feature_calculator.randomState is None:
                    feature_calculator.randomState = self.randomState
            dataset.addDescriptors(self.feature_calculators)
        X = dataset.getDescriptors()
        if self.fitted and not fit:
            assert all(
                feature in X.columns for feature in self.originalfeatureNames
            ), "Some features are missing in the dataset, please check if any "
            "descriptors that were added to the dataset directly "
            "before fitting the pipeline are missing in the dataset."
        else:
            self.originalfeatureNames = X.columns
        y = dataset.getTargets()
        if order is not None:  # FIXME: added to reproduce original behavior
            X = X.loc[order]  # FIXME: added to reproduce original behavior
            y = y.loc[order]  # FIXME: added to reproduce original behavior
            
        # set the dataset for each step
        for step in self.steps.values():
            if hasattr(step, 'setDataSet'):
                step.setDataSet(dataset)
        
        # split the dataset and apply the pipeline
        if split is None:
            X, y, _, _ = super().apply(X, y, fit = fit)
            yield X, y
        else:
            if isinstance(split, str):
                split = dataset.getSplit(split)
            if hasattr(split, 'setDataSet'):
                split.setDataSet(dataset)
            if hasattr(split, 'randomState') and split.randomState is None:
                    split.randomState = self.randomState
            for train_index, test_index in dataset.split(split, X, y):  # FIXME: added to reproduce original behavior
                X_train, y_train, X_test, y_test = (
                    X.loc[train_index], y.loc[train_index], X.loc[test_index], y.loc[test_index]
                )
                yield super().apply(X_train, y_train, X_test, y_test, fit)
    
    def __str__(self):
        feature_calculators = ["None"] if self.feature_calculators is None else self.feature_calculators
        return (
            super().__str__() + 
            f"\nfeature_calculators: {', '.join([str(fc) for fc in feature_calculators])}")

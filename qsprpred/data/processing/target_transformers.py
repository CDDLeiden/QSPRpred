import pandas as pd
from .step import Step
from abc import abstractmethod
import numpy as np
from typing import Literal, ClassVar
from sklearn.preprocessing import LabelEncoder

class TargetTransformer(Step):

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Transform the target data.
        
        Args:
            X (pd.DataFrame): features
            y (pd.DataFrame): target data (to be transformed)
        
        Returns:
            pd.DataFrame: data
            pd.DataFrame: (transformed) target data
        """
        pass
    
class Discretizer(TargetTransformer):
    """Discretizes the target data into bins.
    
    Note. using this step in a pipeline may break the subsequent model training
    as the discretizer does not update the `targetProperties` of the dataset.
    It is recommended to use the `makeClassification` method of the dataset instead,
    see the documentation of the `QSPRDataSet` class.

    Attributes:
        target (str): name of the target property to be discretized
        th (list[float]): thresholds for the bins
        le (LabelEncoder): label encoder for multi-class discretization, only
            used if more than one threshold is provided.
    """
    
    def __init__(self, target: str, th: list[float] | float):
        """Initialize the discretizer.
        
        Args:
            target (str): name of the target property to be discretized
            th (list[float] | float): thresholds for the bins.
                If a single float is provided, it will be used as a single threshold.
                If a list is provided, it should contain at least one value.
        """
        self._fitted = False
        self.target = target
        if isinstance(th, float) or isinstance(th, int):
            th = [th]
        assert len(th) > 0, "Threshold list must contain at least one value."
        if len(th) > 1:
            assert len(th) > 3, (
                "For multi-class classification, set at least 4 thresholds. "
                "These define the lower and upper bounds of the bins, e.g. "
                "[1, 2, 3, 4] will create bins (1,2], (2,3], (3,4]."
            )
        
        self.th = th
    
    def transform(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Discretize the target data into bins.
        
        Args:
            X (pd.DataFrame): features
            y (pd.DataFrame | None): target data to be discretized
        
        Returns:
            pd.DataFrame: data
            pd.DataFrame | None: (discretized) target data
        """
        if y is None:
            return X, y
        if isinstance(y, pd.Series):
            y = pd.DataFrame(y)
        if self.target not in y.columns:
            raise ValueError(f"Target {self.target} not found in target data.")
        if y[self.target].isna().all():
            return X, y
        if len(self.th) > 1:
            assert max(y[self.target].dropna()) <= max(self.th), (
                "Make sure final threshold value is not smaller "
                "than largest value of property"
            )
            assert min(y[self.target].dropna()) >= min(self.th), (
                "Make sure first threshold value is not larger "
                "than smallest value of property"
            )
            y_intervals = pd.cut(
                y[self.target], bins=self.th, include_lowest=True
            ).astype(str)
            self.le = LabelEncoder()
            encoded_intervals = self.le.fit_transform(y_intervals)
            y_transformed = pd.DataFrame(
                np.where(
                    y[self.target].notna(), encoded_intervals, np.nan
                ).astype(float),
                columns=[self.target],
                index=y.index
            )         
        else:
            binary_target = y[self.target] > self.th[0]
            y_transformed = pd.DataFrame(
                binary_target.where(y[self.target].notna(), np.nan).astype(float),
                columns=[self.target],
                index=y.index
            )
        return X, y_transformed
    
    def getIntervals(self, discrete_values: pd.Series) -> pd.Series:
        """Transform the discretized values to intervals.
        
        Args:
            discrete_values (pd.Series): discretized values
        Returns:
            pd.Series: intervals corresponding to the discretized values
        """
        if not self.fitted:
            raise ValueError("Discretizer has not been fitted yet.")
        if not hasattr(self, "le"):
            raise ValueError(
                "No label encoder found, intervals can only be retrieved "
                "for multi-class discretization."
            )
        return pd.Series(
            self.le.inverse_transform(discrete_values.astype(int)),
            index=discrete_values.index
        )

    
class SimpleTargetTransformer(TargetTransformer):
    """Applies a simple transformation to the target data.

    Attributes:
        transform_dict (dict): dictionary of available transformations
        transformer (callable): numpy function
    """
    _notJSON: ClassVar = ["transform_dict", "inverse_transform_dict"]
    
    transform_dict = {
        "log10": lambda x: (__import__("numpy").log10(x)),
        "log2": lambda x: (__import__("numpy").log2(x)),
        "log": lambda x: (__import__("numpy").log(x)),
        "sqrt": lambda x: (__import__("numpy").sqrt(x)),
        "cbrt": lambda x: (__import__("numpy").cbrt(x)),
        "exp": lambda x: (__import__("numpy").exp(x)),
        "square": lambda x: __import__("numpy").power(x, 2),
        "cube": lambda x: __import__("numpy").power(x, 3),
        "reciprocal": lambda x: __import__("numpy").reciprocal(x),
    }
    inverse_transform_dict = {
        "log10": lambda x: (__import__("numpy").power(10, x)),
        "log2": lambda x: (__import__("numpy").power(2, x)),
        "log": lambda x: (__import__("numpy").exp(x)),
        "sqrt": lambda x: (__import__("numpy").power(x, 2)),
        "cbrt": lambda x: (__import__("numpy").power(x, 3)),
        "exp": lambda x: (__import__("numpy").log(x)),
        "square": lambda x: __import__("numpy").sqrt(x),
        "cube": lambda x: __import__("numpy").cbrt(x),
        "reciprocal": lambda x: __import__("numpy").reciprocal(x),
    }
    
    
    def __init__(
        self, 
        target: str,
        transformation: Literal["log10", "log2", "log", "sqrt", "cbrt", "exp", "square", "cube", "reciprocal"]
    ):
        """Initialize the SklearnStep
        
        Args:
            target (str): name of the target property to be transformed
            transformation (str): transformer function, should be 
            one of log10, log2, log, sqrt, cbrt, exp, square, cube, reciprocal
        """
        self._fitted = False
        self.target = target
        self.transformation = transformation
        if transformation not in self.transform_dict:
            raise ValueError(
                f"Transformation {transformation} not recognized. "
                f"Available transformations: {list(self.transform_dict.keys())}"
            )
        
    
    def transform(
        self, X: pd.DataFrame, y: None | pd.DataFrame = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Transform the data using the transformer
        
        Args:
            X (pd.DataFrame): data to be transformed
            y (pd.DataFrame | None): target data to be transformed

        Returns:
            pd.DataFrame: transformed data
            pd.DataFrame | None: (transformed) target data
        """
        if y is None:
            return X, None
        if self.target not in y.columns:
            raise ValueError(f"Target {self.target} not found in target data.")
        
        y[self.target] = self.getTransformer()(y[self.target])
        return X, y
    
    def inverseTransform(
        self, X: pd.DataFrame, y: None | pd.DataFrame = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Inverse transform the data using the inverse transformer
        
        Args:
            X (pd.DataFrame): data to be transformed
            y (pd.DataFrame | None): target data to be transformed

        Returns:
            pd.DataFrame: transformed data
            pd.DataFrame | None: (transformed) target data
        """
        if y is None:
            return X, None
        if self.target not in y.columns:
            raise ValueError(f"Target {self.target} not found in target data.")
        
        y[self.target] = self.getInverseTransformer()(y[self.target])
        return X, y
        
    def getTransformer(self) -> callable:
        """Get the transformer function

        Returns:
            callable: transformer function
        """
        return self.transform_dict[self.transformation]

    def getInverseTransformer(self) -> callable:
        """Get the inverse transformer function

        Returns:
            callable: inverse transformer function
        """
        return self.inverse_transform_dict[self.transformation]